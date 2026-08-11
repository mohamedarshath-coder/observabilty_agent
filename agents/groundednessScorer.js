/**
 * Groundedness Scorer
 *
 * Measures how well the AI response is anchored in verifiable source data.
 *
 * Real path (when `nliScore` is supplied):
 *   - Each sentence of the response is checked against the retrieved context
 *     using a real local NLI (Natural Language Inference) cross-encoder model —
 *     the sentence is the hypothesis, the context is the premise, and the
 *     model's entailment probability is the groundedness score.
 *   - Context Relevance uses real sentence-embedding cosine similarity (when
 *     `embedText`/`cosineSimilarity` are supplied) between the question and
 *     the response, instead of keyword overlap.
 *
 * Fallback path (no NLI/embedder available): n-gram overlap heuristic, same
 * as before — keeps the scorer functional if the real dependencies aren't
 * wired up for a given call.
 */

'use strict';

const MAX_SENTENCES_FOR_NLI = 10; // bound real-model calls per request for latency

function sentences(text) {
  return text.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(s => s.length > 15);
}

function ngrams(text, n) {
  const tokens = text.toLowerCase().split(/\W+/).filter(Boolean);
  const grams  = new Set();
  for (let i = 0; i <= tokens.length - n; i++) {
    grams.add(tokens.slice(i, i + n).join(' '));
  }
  return grams;
}

function sentenceGroundednessNgram(sentence, contextText) {
  if (!contextText || contextText.trim().length === 0) return 0.5;

  const uni1  = ngrams(sentence, 1);
  const bi1   = ngrams(sentence, 2);
  const uniC  = ngrams(contextText, 1);
  const biC   = ngrams(contextText, 2);

  const uniMatch = [...uni1].filter(g => uniC.has(g)).length;
  const biMatch  = [...bi1].filter(g => biC.has(g)).length;

  const uniScore = uni1.size > 0 ? uniMatch / uni1.size : 0;
  const biScore  = bi1.size  > 0 ? biMatch  / bi1.size  : 0;

  return parseFloat((0.4 * uniScore + 0.6 * biScore).toFixed(3));
}

function contextRelevanceNgram(response, question) {
  if (!question || question.trim().length === 0) return 0.7;
  const qTokens = new Set(question.toLowerCase().split(/\W+/).filter(w => w.length > 3));
  const rTokens = response.toLowerCase().split(/\W+/).filter(w => w.length > 3);
  const hits    = rTokens.filter(t => qTokens.has(t)).length;
  return parseFloat(Math.min(0.98, hits / Math.max(qTokens.size, 1)).toFixed(3));
}

async function runGroundednessScorer({
  response, context = '', question = '', nliScore = null, embedText = null, cosineSimilarity = null,
}) {
  const sents = sentences(response).slice(0, MAX_SENTENCES_FOR_NLI);
  const usingNli = typeof nliScore === 'function' && context.trim().length > 0;

  let sentenceResults;
  if (usingNli) {
    sentenceResults = await Promise.all(sents.map(async s => {
      const nli = await nliScore(context, s);
      const score = nli ? parseFloat(nli.entailment.toFixed(3)) : sentenceGroundednessNgram(s, context);
      return {
        sentence:  s.slice(0, 100) + (s.length > 100 ? '…' : ''),
        score,
        supported: score >= 0.5,
        nli: nli ? { entailment: nli.entailment, neutral: nli.neutral, contradiction: nli.contradiction } : null,
      };
    }));
  } else {
    sentenceResults = sents.map(s => {
      const score = sentenceGroundednessNgram(s, context);
      return { sentence: s.slice(0, 100) + (s.length > 100 ? '…' : ''), score, supported: score >= 0.35, nli: null };
    });
  }

  const groundednessScore = sentenceResults.length > 0
    ? parseFloat((sentenceResults.reduce((a, b) => a + b.score, 0) / sentenceResults.length).toFixed(3))
    : 0.5;

  let contextRelevance;
  let relevanceMethod;
  if (typeof embedText === 'function' && typeof cosineSimilarity === 'function' && question.trim()) {
    const [qVec, rVec] = await Promise.all([embedText(question), embedText(response)]);
    if (qVec && rVec) {
      contextRelevance = parseFloat(Math.max(0, Math.min(0.98, cosineSimilarity(qVec, rVec))).toFixed(3));
      relevanceMethod = 'embedding_cosine_similarity';
    }
  }
  if (contextRelevance === undefined) {
    contextRelevance = contextRelevanceNgram(response, question);
    relevanceMethod = 'keyword_overlap_fallback';
  }

  const unsupportedSentences = sentenceResults
    .filter(s => !s.supported)
    .map(s => s.sentence);

  return {
    scorer:               'GroundednessScorer',
    method:               usingNli ? 'nli_entailment' : 'ngram_fallback',
    context_relevance_method: relevanceMethod,
    sentence_count:       sentenceResults.length,
    sentence_results:     sentenceResults,
    unsupported_sentences: unsupportedSentences,
    groundedness_score:   groundednessScore,
    context_relevance:    contextRelevance,
    confidence:           parseFloat(((groundednessScore * 0.7 + contextRelevance * 0.3)).toFixed(3)),
    interpretation: groundednessScore >= 0.6
      ? 'Response is well-grounded in provided source context'
      : groundednessScore >= 0.35
        ? 'Partial grounding — some claims lack context support'
        : 'Poor grounding — response not anchored in source material',
  };
}

module.exports = { runGroundednessScorer };
