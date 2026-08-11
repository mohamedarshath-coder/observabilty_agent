/**
 * Reflection Scorer
 *
 * Self-evaluation: "Did I answer correctly?" — re-verifies each extracted
 * claim independently and checks agreement with the original response.
 *
 * Real path (when `nliScore` is supplied): each claim is checked against the
 * retrieved context with a local NLI cross-encoder model (same model used by
 * the Factual Verification and Groundedness scorers) — entailment probability
 * is the verification confidence.
 *
 * Fallback path (no NLI available): word-overlap ratio between claim and
 * context, same as before.
 */

'use strict';

const CLAIM_PATTERNS = [
  /(?:is|are|was|were|has|have|had)\s+[^.!?]{10,60}[.!?]/gi,
  /\b(?:the|a|an)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:is|was|are|were)\s+[^.!?]{5,50}/gi,
  /\b\d{4}\b[^.!?]{3,40}[.!?]/gi,                // year-based claims
  /\b(?:always|never|every|all|none)\b[^.!?]{5,50}/gi,  // absolute claims
];

const MAX_CLAIMS_FOR_NLI = 6;

function extractClaims(text) {
  const found = new Set();
  CLAIM_PATTERNS.forEach(pat => {
    const matches = text.match(new RegExp(pat.source, pat.flags)) || [];
    matches.forEach(m => found.add(m.trim().slice(0, 120)));
  });
  if (found.size === 0) {
    text.split(/(?<=[.!?])\s+/)
      .filter(s => s.length > 20)
      .slice(0, 5)
      .forEach(s => found.add(s.slice(0, 120)));
  }
  return [...found].slice(0, MAX_CLAIMS_FOR_NLI);
}

function verifyClaimLexical(claim, context) {
  if (!context || context.trim().length === 0) {
    return { verified: null, confidence: 0.5, note: 'No context to verify against' };
  }
  const claimTokens  = new Set(claim.toLowerCase().split(/\W+/).filter(w => w.length > 3));
  const ctxLower     = context.toLowerCase();
  let   matches      = 0;
  claimTokens.forEach(t => { if (ctxLower.includes(t)) matches++; });
  const ratio        = claimTokens.size > 0 ? matches / claimTokens.size : 0;
  const verified     = ratio >= 0.4;
  return {
    verified,
    confidence: parseFloat((0.3 + ratio * 0.7).toFixed(3)),
    note: verified ? 'Claim supported by context (lexical overlap)' : 'Claim not found in context',
  };
}

async function verifyClaimAgainstContext(claim, context, nliScore) {
  if (!context || context.trim().length === 0) {
    return { verified: null, confidence: 0.5, note: 'No context to verify against' };
  }
  if (typeof nliScore === 'function') {
    const nli = await nliScore(context, claim);
    if (nli) {
      const verified = nli.entailment >= 0.5;
      return {
        verified,
        confidence: parseFloat(nli.entailment.toFixed(3)),
        note: verified
          ? `NLI model entails this claim (${(nli.entailment * 100).toFixed(0)}%)`
          : `NLI model does not entail this claim (entailment ${(nli.entailment * 100).toFixed(0)}%, contradiction ${(nli.contradiction * 100).toFixed(0)}%)`,
      };
    }
  }
  return verifyClaimLexical(claim, context);
}

async function runReflectionScorer({ response, context = '', nliScore = null }) {
  const claims = extractClaims(response);

  const reflections = await Promise.all(claims.map(async claim => {
    const verification = await verifyClaimAgainstContext(claim, context, nliScore);
    return {
      claim:           claim.slice(0, 100),
      verified:        verification.verified,
      confidence:      verification.confidence,
      note:            verification.note,
      verification_q:  `Is it true that: "${claim.slice(0, 60)}…"?`,
    };
  }));

  const verifiedCount   = reflections.filter(r => r.verified === true).length;
  const unverifiedCount = reflections.filter(r => r.verified === false).length;
  const totalVerifiable = reflections.filter(r => r.verified !== null).length;

  const consistency = totalVerifiable > 0
    ? verifiedCount / totalVerifiable
    : 0.6; // no claims → neutral

  const confidence = parseFloat((0.2 + consistency * 0.8).toFixed(3));

  const selfEval = confidence >= 0.75
    ? 'Self-check passed: claims are consistent with available evidence.'
    : confidence >= 0.5
      ? 'Self-check partial: some claims could not be verified.'
      : 'Self-check failed: multiple claims contradict or lack evidence.';

  return {
    scorer:          'ReflectionScorer',
    method:          typeof nliScore === 'function' ? 'nli_entailment' : 'lexical_overlap_fallback',
    claims_extracted: claims.length,
    reflections,
    verified_claims:   verifiedCount,
    unverified_claims: unverifiedCount,
    consistency_ratio: parseFloat(consistency.toFixed(3)),
    confidence,
    self_evaluation:   selfEval,
    interpretation: confidence >= 0.7
      ? 'High self-consistency — response likely accurate'
      : confidence >= 0.45
        ? 'Moderate self-consistency — some claims unverified'
        : 'Low self-consistency — iterative improvement needed',
  };
}

module.exports = { runReflectionScorer };
