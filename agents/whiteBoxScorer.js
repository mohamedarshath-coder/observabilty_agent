/**
 * White Box Scorer
 *
 * Evaluates internal token-level confidence signals.
 * Each output token carries a probability from the generating model;
 * low-probability tokens flag uncertain/hallucinated content.
 *
 * Real path (when `tokenLogprobs` is supplied):
 *   1. Use the actual per-token log-probabilities returned by the LLM API call
 *      (Together's chat completions endpoint, requested with `logprobs: true`)
 *   2. Confidence = geometric mean of real token probabilities (exp(logprob))
 *   3. Flag the lowest-probability tokens as hallucination hotspots
 *
 * Fallback path (no real logprobs available — e.g. provider/model didn't return
 * them): reconstructs a pseudo-probability per token from context grounding and
 * token shape, same as before. This keeps the scorer from crashing or stalling
 * the pipeline if the upstream API call didn't include usable logprob data.
 */

'use strict';

// Common English function words — assigned high probability by default (fallback path only)
const FUNCTION_WORDS = new Set([
  'the','a','an','is','are','was','were','be','been','being',
  'have','has','had','do','does','did','will','would','shall','should',
  'may','might','must','can','could','and','or','but','if','then',
  'that','this','these','those','it','its','in','on','at','to','of',
  'for','with','by','from','as','not','no','so','yet','nor','both',
]);

function tokenise(text) {
  return text.toLowerCase().match(/\b[a-z']+\b/g) || [];
}

function buildContextSet(context) {
  return new Set(tokenise(context));
}

function pseudoProbability(token, contextSet) {
  if (FUNCTION_WORDS.has(token))        return 0.95;  // common function word
  if (contextSet.has(token))            return 0.88;  // grounded in context
  if (token.length <= 3)                return 0.80;  // short token
  if (/^\d+$/.test(token))              return 0.45;  // bare number — risky
  if (token.length > 12)                return 0.40;  // long unusual token
  return 0.62;                                        // unknown content token
}

function geometricMean(probs) {
  if (probs.length === 0) return 0;
  const logSum = probs.reduce((s, p) => s + Math.log(Math.max(p, 1e-10)), 0);
  return Math.exp(logSum / probs.length);
}

function scoreFromRealLogprobs(tokenLogprobs, contextSet) {
  const tokenScores = tokenLogprobs.map(t => {
    const cleanToken = (t.token || '').trim().toLowerCase().replace(/^[▁Ġ]/, ''); // strip common SentencePiece/BPE markers
    return {
      token:       cleanToken || t.token,
      probability: parseFloat(Math.min(1, Math.exp(t.logprob)).toFixed(4)),
      grounded:    contextSet.has(cleanToken),
    };
  });
  return tokenScores;
}

async function runWhiteBoxScorer({ response, context = '', tokenLogprobs = null }) {
  const contextSet = buildContextSet(context);
  const usingRealLogprobs = Array.isArray(tokenLogprobs) && tokenLogprobs.length > 0;

  const tokenScores = usingRealLogprobs
    ? scoreFromRealLogprobs(tokenLogprobs, contextSet)
    : tokenise(response).map(tok => ({
        token:       tok,
        probability: parseFloat(pseudoProbability(tok, contextSet).toFixed(3)),
        grounded:    contextSet.has(tok),
      }));

  const probs      = tokenScores.map(t => t.probability);
  const confidence = parseFloat(geometricMean(probs).toFixed(3));

  // Hotspots: lowest 20% probability tokens
  const sorted   = [...tokenScores].sort((a, b) => a.probability - b.probability);
  const hotspots = sorted
    .filter(t => t.probability < 0.5 && !FUNCTION_WORDS.has(t.token))
    .slice(0, 8)
    .map(t => t.token);

  const chainOfThought = [
    usingRealLogprobs
      ? `Read ${tokenScores.length} real per-token log-probabilities from the LLM API response.`
      : `No real logprobs available — fell back to heuristic pseudo-probabilities over ${tokenScores.length} tokens.`,
    `Context coverage: ${tokenScores.filter(t => t.grounded).length}/${tokenScores.length} tokens grounded.`,
    `Geometric mean probability: ${confidence}.`,
    hotspots.length > 0
      ? `Low-confidence tokens detected: ${hotspots.join(', ')}.`
      : 'All tokens within acceptable confidence range.',
  ];

  return {
    scorer:         'WhiteBoxScorer',
    method:         usingRealLogprobs ? 'real_token_logprobs' : 'heuristic_fallback',
    token_count:    tokenScores.length,
    grounded_tokens: tokenScores.filter(t => t.grounded).length,
    hotspot_tokens:  hotspots,
    min_token_prob:  probs.length > 0 ? parseFloat(Math.min(...probs).toFixed(3)) : 0,
    max_token_prob:  probs.length > 0 ? parseFloat(Math.max(...probs).toFixed(3)) : 0,
    confidence,
    chain_of_thought: chainOfThought,
    interpretation: confidence >= 0.75
      ? 'High token certainty — well-grounded response'
      : confidence >= 0.55
        ? 'Moderate certainty — verify flagged tokens'
        : 'Low token certainty — likely hallucination',
  };
}

module.exports = { runWhiteBoxScorer };
