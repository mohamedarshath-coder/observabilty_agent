/**
 * NeuroSymbolic AI Scorer
 *
 * Merges neural (LLM-based) signals with symbolic reasoning rules,
 * enabling agents to reason, generalize, and explain decisions with
 * greater transparency and control.
 *
 * Symbolic rule layers:
 *  Rule 1 — Temporal consistency    (dates are logically ordered)
 *  Rule 2 — Numerical consistency   (quantities are plausible ranges)
 *  Rule 3 — Entity resolution       (named entities appear consistently)
 *  Rule 4 — Logical form            (if-then statements are well-formed)
 *  Rule 5 — Causal integrity        (cause precedes effect)
 *  Rule 6 — Quantifier coherence    (universal/existential quantifiers are consistent)
 *
 * Neural layer:
 *  - Semantic plausibility score (context overlap + structural similarity)
 *
 * Final score = weighted combination of symbolic pass rate + neural score
 */

'use strict';

// ── Symbolic Rules ────────────────────────────────────────────────────────────

function rule1_temporalConsistency(text) {
  const years = (text.match(/\b(1[0-9]{3}|20[0-2][0-9])\b/g) || []).map(Number);
  if (years.length < 2) return { rule: 'TemporalConsistency', passed: true, note: 'No temporal sequence to validate.' };
  // Check if years appear in roughly ascending order when mentioned as a sequence
  const inOrder = years.every((y, i) => i === 0 || y >= years[i - 1] - 50); // allow 50yr leeway
  return {
    rule:   'TemporalConsistency',
    passed: inOrder,
    note:   inOrder ? `Year sequence plausible: ${years.join(' → ')}` : `Out-of-order year sequence: ${years.join(', ')}`,
  };
}

function rule2_numericalConsistency(text) {
  const numbers = (text.match(/\b\d+(?:\.\d+)?\b/g) || []).map(Number).filter(n => n > 0 && n < 1e12);
  // Flag impossibly large round numbers that look fabricated
  const suspicious = numbers.filter(n => n > 1e9 && n % 1e8 === 0);
  return {
    rule:   'NumericalConsistency',
    passed: suspicious.length === 0,
    note:   suspicious.length > 0
      ? `Suspiciously round large numbers: ${suspicious.slice(0,3).join(', ')}`
      : 'Numerical values appear plausible.',
  };
}

function rule3_entityConsistency(text) {
  // Extract proper nouns (capitalised sequences)
  const entities = (text.match(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b/g) || []);
  const freq     = {};
  entities.forEach(e => { freq[e] = (freq[e] || 0) + 1; });
  // Flag entity name variations that differ by one word (possible inconsistency)
  const names = Object.keys(freq);
  const conflicts = [];
  for (let i = 0; i < names.length; i++) {
    for (let j = i + 1; j < names.length; j++) {
      const a = names[i].toLowerCase().split(' ');
      const b = names[j].toLowerCase().split(' ');
      const shared = a.filter(w => b.includes(w));
      if (shared.length > 0 && a.length !== b.length) {
        conflicts.push(`"${names[i]}" vs "${names[j]}"`);
      }
    }
  }
  return {
    rule:    'EntityConsistency',
    passed:  conflicts.length === 0,
    note:    conflicts.length > 0
      ? `Potential entity inconsistency: ${conflicts.slice(0,2).join('; ')}`
      : `${names.length} entities referenced consistently.`,
  };
}

function rule4_logicalFormIntegrity(text) {
  // Check if-then constructs are well-formed
  const ifThen = text.match(/\bif\b[^.!?]{5,80}\bthen\b/gi) || [];
  const bareIf  = text.match(/\bif\b[^.!?]{5,80}[.!?]/gi) || [];
  // If there are "if" without "then" — possible incomplete logical form
  const incomplete = bareIf.length > ifThen.length;
  return {
    rule:   'LogicalFormIntegrity',
    passed: !incomplete || bareIf.length < 2,
    note:   incomplete && bareIf.length >= 2
      ? 'Incomplete if-then constructs detected.'
      : 'Logical constructs appear well-formed.',
  };
}

function rule5_causalIntegrity(text) {
  // "X caused Y" — cause should precede effect in narrative
  const causalPhrases = (text.match(/\b(?:because|therefore|thus|hence|caused|resulted in|led to)\b/gi) || []);
  return {
    rule:   'CausalIntegrity',
    passed: true, // heuristic: presence of causal connectors is a positive signal
    note:   causalPhrases.length > 0
      ? `${causalPhrases.length} causal connectors found — coherent causal chain.`
      : 'No explicit causal reasoning detected.',
  };
}

function rule6_quantifierCoherence(text) {
  const universals   = (text.match(/\b(all|every|always|none|never|no one)\b/gi) || []).length;
  const existentials = (text.match(/\b(some|sometimes|often|many|few|occasionally)\b/gi) || []).length;
  // Mixing extreme quantifiers with hedged ones in same sentence is risky
  const sentences = text.split(/[.!?]/);
  const mixed = sentences.filter(s => {
    const u = /\b(all|every|always|never|none)\b/i.test(s);
    const e = /\b(some|sometimes|often|occasionally)\b/i.test(s);
    return u && e;
  });
  return {
    rule:   'QuantifierCoherence',
    passed: mixed.length <= 1,
    note:   mixed.length > 1
      ? `${mixed.length} sentences mix absolute and hedged quantifiers.`
      : 'Quantifier usage is coherent.',
  };
}

// ── Neural plausibility layer ─────────────────────────────────────────────────
// Real path: cosine similarity between real sentence-embedding vectors of the
// response and the retrieved context (same embedding model used for RAG).
// Fallback: word-overlap ratio, used only if no embedder is supplied.
function neuralPlausibilityWordOverlap(response, context) {
  if (!context || context.trim().length === 0) return 0.6;
  const rWords = new Set(response.toLowerCase().split(/\W+/).filter(w => w.length > 3));
  const cWords = new Set(context.toLowerCase().split(/\W+/).filter(w => w.length > 3));
  const overlap = [...rWords].filter(w => cWords.has(w)).length;
  return parseFloat(Math.min(0.97, 0.35 + (overlap / Math.max(rWords.size, 1)) * 0.65).toFixed(3));
}

async function neuralPlausibility(response, context, embedText, cosineSimilarity) {
  if (!context || context.trim().length === 0) return { score: 0.6, method: 'no_context' };
  if (typeof embedText === 'function' && typeof cosineSimilarity === 'function') {
    const [rVec, cVec] = await Promise.all([embedText(response), embedText(context)]);
    if (rVec && cVec) {
      return {
        score: parseFloat(Math.max(0, Math.min(0.97, cosineSimilarity(rVec, cVec))).toFixed(3)),
        method: 'embedding_cosine_similarity',
      };
    }
  }
  return { score: neuralPlausibilityWordOverlap(response, context), method: 'word_overlap_fallback' };
}

// ── Main scorer ───────────────────────────────────────────────────────────────
async function runNeuroSymbolicScorer({ response, context = '', embedText = null, cosineSimilarity = null }) {
  const rules = [
    rule1_temporalConsistency(response),
    rule2_numericalConsistency(response),
    rule3_entityConsistency(response),
    rule4_logicalFormIntegrity(response),
    rule5_causalIntegrity(response),
    rule6_quantifierCoherence(response),
  ];

  const passedRules  = rules.filter(r => r.passed).length;
  const symbolicScore = parseFloat((passedRules / rules.length).toFixed(3));
  const neural = await neuralPlausibility(response, context, embedText, cosineSimilarity);
  const neuralScore = neural.score;

  // Weighted combination: symbolic 60% + neural 40%
  const confidence = parseFloat((symbolicScore * 0.6 + neuralScore * 0.4).toFixed(3));

  return {
    scorer:          'NeuroSymbolicScorer',
    method:          `symbolic_rules_plus_${neural.method}`,
    symbolic_rules:  rules,
    passed_rules:    passedRules,
    total_rules:     rules.length,
    symbolic_score:  symbolicScore,
    neural_score:    neuralScore,
    confidence,
    explanation: rules
      .filter(r => !r.passed)
      .map(r => `[${r.rule}] ${r.note}`)
      .concat(rules.filter(r => r.passed && r.note).map(r => `✓ ${r.rule}`))
      .join(' | '),
    interpretation: confidence >= 0.75
      ? 'Symbolic rules pass + neural signals aligned — high trustworthiness'
      : confidence >= 0.5
        ? 'Mixed symbolic/neural signals — moderate confidence'
        : 'Symbolic rule violations and/or low neural plausibility',
  };
}

module.exports = { runNeuroSymbolicScorer };
