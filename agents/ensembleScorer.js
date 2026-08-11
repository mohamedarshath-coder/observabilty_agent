/**
 * Ensemble Scorer
 *
 * Combines all 8 scoring methods into one holistic final score.
 * Combines: relevance, groundedness, factuality for a more robust evaluation.
 *
 * Scorer weights (tunable via config):
 *   BlackBox          0.12  — consistency across samples
 *   WhiteBox          0.13  — token-level certainty
 *   LLM-as-a-Judge    0.20  — multi-judge vote
 *   Groundedness      0.18  — RAG/source anchoring
 *   NeuroSymbolic     0.12  — symbolic rule + neural
 *   Factual (legacy)  0.10  — NLI entailment from factualVerification
 *
 * Final verdict thresholds:
 *   ensemble_score >= 0.72 → pass
 *   ensemble_score >= 0.45 → warn
 *   ensemble_score <  0.45 → block
 */

'use strict';

const DEFAULT_WEIGHTS = {
  black_box:       0.12,
  white_box:       0.13,
  llm_judge:       0.20,
  groundedness:    0.18,
  neuro_symbolic:  0.12,
  factual:         0.10,
};

function ensembleScore(scores, weights = DEFAULT_WEIGHTS) {
  const totalWeight = Object.values(weights).reduce((a, b) => a + b, 0);
  let weighted = 0;
  Object.entries(weights).forEach(([key, w]) => {
    weighted += (scores[key] ?? 0.5) * w;
  });
  return parseFloat((weighted / totalWeight).toFixed(3));
}

function deriveVerdict(score) {
  if (score >= 0.72) return 'pass';
  if (score >= 0.45) return 'warn';
  return 'block';
}

function buildScoreBreakdown(results) {
  return {
    black_box:      results.blackBox?.confidence      ?? null,
    white_box:      results.whiteBox?.confidence      ?? null,
    llm_judge:      results.llmJudge?.confidence      ?? null,
    groundedness:   results.groundedness?.confidence  ?? null,
    neuro_symbolic: results.neuroSymbolic?.confidence ?? null,
    factual:        results.factual?.factuality_score ?? null,
  };
}

function topIssues(results) {
  const issues = [];
  if (results.blackBox?.confidence      < 0.5) issues.push('Low response consistency (Black Box)');
  if (results.whiteBox?.hotspot_tokens?.length > 3) issues.push(`${results.whiteBox.hotspot_tokens.length} low-confidence tokens (White Box)`);
  const judgeCount = results.llmJudge?.judge_count || 0;
  if (judgeCount > 0 && results.llmJudge.votes.incorrect >= Math.ceil(judgeCount / 2)) {
    issues.push(`${results.llmJudge.votes.incorrect}/${judgeCount} judges voted Incorrect`);
  }
  if (results.groundedness?.unsupported_sentences?.length > 2) issues.push('Multiple unsupported sentences (Groundedness)');
  if (results.neuroSymbolic?.passed_rules < 4) issues.push(`${results.neuroSymbolic.total_rules - results.neuroSymbolic.passed_rules} symbolic rules violated`);
  if (results.factual?.factuality_score < 0.5) issues.push('Low factuality score (NLI)');
  return issues;
}

async function runEnsembleScorer(results) {
  const breakdown   = buildScoreBreakdown(results);
  const score       = ensembleScore(breakdown);
  const verdict     = deriveVerdict(score);
  const issues      = topIssues(results);

  // Build per-scorer summary for the UI
  const scorerSummary = [
    { name: 'Black Box',       score: breakdown.black_box,      weight: DEFAULT_WEIGHTS.black_box,       method: 'Pairwise Similarity' },
    { name: 'White Box',       score: breakdown.white_box,      weight: DEFAULT_WEIGHTS.white_box,       method: 'Token Probability' },
    { name: 'LLM as a Judge',  score: breakdown.llm_judge,      weight: DEFAULT_WEIGHTS.llm_judge,       method: 'Multi-Judge Ensemble' },
    { name: 'Groundedness',    score: breakdown.groundedness,   weight: DEFAULT_WEIGHTS.groundedness,    method: 'Sentence Entailment' },
    { name: 'NeuroSymbolic',   score: breakdown.neuro_symbolic, weight: DEFAULT_WEIGHTS.neuro_symbolic,  method: 'Symbolic Rules + Neural' },
    { name: 'Factual (NLI)',   score: breakdown.factual,        weight: DEFAULT_WEIGHTS.factual,         method: 'NLI Entailment' },
  ].map(s => ({ ...s, score: s.score !== null ? parseFloat(s.score.toFixed(3)) : null }));

  return {
    scorer:          'EnsembleScorer',
    method:          'weighted_combination',
    scorer_summary:  scorerSummary,
    score_breakdown: breakdown,
    weights:         DEFAULT_WEIGHTS,
    ensemble_score:  score,
    verdict,
    top_issues:      issues,
    risk_level:      score >= 0.72 ? 'low' : score >= 0.45 ? 'medium' : 'high',
    interpretation:  verdict === 'pass'
      ? `All scorers converge on a trustworthy response (${(score*100).toFixed(0)}%)`
      : verdict === 'warn'
        ? `Mixed signals across scorers — proceed with caution (${(score*100).toFixed(0)}%)`
        : `Multiple scorers flag this response — block recommended (${(score*100).toFixed(0)}%)`,
  };
}

module.exports = { runEnsembleScorer, DEFAULT_WEIGHTS };
