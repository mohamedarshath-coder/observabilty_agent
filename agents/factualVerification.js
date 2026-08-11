/**
 * Factual Verification Agent
 *
 * Performs hallucination scoring via real NLI entailment when available:
 * each claim extracted from the response is checked against the retrieved
 * context using a local NLI cross-encoder model — the context is the premise,
 * the claim is the hypothesis, and entailment probability is the claim's
 * factuality confidence.
 *
 * Falls back to a lexical word-overlap heuristic if no `nliScore` function is
 * supplied, so the agent stays functional even without the real model wired up.
 *
 * PII detection (SSN / card / email / phone patterns) is a separate compliance
 * check, unrelated to hallucination scoring, and is unaffected by this change.
 */

const PII_PATTERNS = [
  { pattern: /\b\d{3}-\d{2}-\d{4}\b/, label: 'SSN' },
  { pattern: /\b4[0-9]{12}(?:[0-9]{3})?\b/, label: 'Visa card' },
  { pattern: /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/i, label: 'Email' },
  { pattern: /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/, label: 'Phone number' },
];

const MAX_CLAIMS_FOR_NLI = 6;

function extractClaims(text) {
  return text
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim())
    .filter(s => s.length > 20 && s.length < 300);
}

function scoreClaimAgainstContextLexical(claim, context) {
  if (!context || context.trim().length === 0) {
    return { supported: null, confidence: 0.5, source: 'No context provided' };
  }
  const claimWords = new Set(claim.toLowerCase().split(/\W+/).filter(w => w.length > 4));
  const contextLower = context.toLowerCase();
  let matches = 0;
  claimWords.forEach(w => { if (contextLower.includes(w)) matches++; });
  const ratio = claimWords.size > 0 ? matches / claimWords.size : 0;
  const supported = ratio > 0.4;
  return {
    supported,
    confidence: Math.min(0.95, 0.4 + ratio * 0.6),
    source: supported ? 'Matched against provided context (lexical overlap)' : 'Insufficient context support',
  };
}

async function scoreClaimAgainstContext(claim, context, nliScore) {
  if (!context || context.trim().length === 0) {
    return { supported: null, confidence: 0.5, source: 'No context provided' };
  }
  if (typeof nliScore === 'function') {
    const nli = await nliScore(context, claim);
    if (nli) {
      const supported = nli.entailment >= 0.5;
      const contradicted = nli.contradiction > nli.entailment && nli.contradiction > nli.neutral;
      return {
        supported: contradicted ? false : supported,
        confidence: parseFloat(nli.entailment.toFixed(3)),
        source: contradicted
          ? `NLI model detected contradiction (${(nli.contradiction * 100).toFixed(0)}%)`
          : supported
            ? `NLI entailment against retrieved context (${(nli.entailment * 100).toFixed(0)}%)`
            : `Insufficient NLI entailment (${(nli.entailment * 100).toFixed(0)}%)`,
      };
    }
  }
  return scoreClaimAgainstContextLexical(claim, context);
}

async function runFactualVerification({ response, context, nliScore = null }) {
  const claims = extractClaims(response).slice(0, MAX_CLAIMS_FOR_NLI);

  // Check for PII — unrelated compliance check, always runs regardless of NLI availability.
  const piiFound = PII_PATTERNS.filter(p => p.pattern.test(response)).map(p => p.label);

  const evidence = await Promise.all(claims.map(async claim => {
    const result = await scoreClaimAgainstContext(claim, context, nliScore);
    return { claim, ...result };
  }));

  const supportedCount = evidence.filter(e => e.supported !== false).length;
  const factualityScore = evidence.length > 0
    ? parseFloat((supportedCount / evidence.length).toFixed(2))
    : 0.8; // no claims to verify → assume OK

  return {
    agent: 'FactualVerificationAgent',
    method: typeof nliScore === 'function' ? 'nli_entailment' : 'lexical_overlap_fallback',
    factuality_score: factualityScore,
    evidence,
    pii_detected: piiFound,
    claim_count: claims.length,
  };
}

module.exports = { runFactualVerification };
