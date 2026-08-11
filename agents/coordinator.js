/**
 * Hallucination Guard Coordinator  (v3 — real ML/LLM evaluation pipeline)
 *
 * Dispatches ALL scorers in parallel, feeds results to EnsembleScorer,
 * then applies mitigation based on the ensemble verdict.
 *
 * Scorer pipeline (all run concurrently via Promise.all):
 *   1. Factual Verification   — real local NLI entailment (Xenova/mobilebert-uncased-mnli) + PII detection
 *   2. Responsible AI Policy  — policy rule checks (rule-based by design)
 *   3. Black Box Scorer       — real independent LLM resample + embedding cosine similarity
 *   4. White Box Scorer       — real per-token logprobs from the generating LLM call
 *   5. LLM-as-a-Judge         — real 3-judge cross-provider ensemble (OpenAI + Anthropic + Together)
 *   6. Groundedness Scorer    — real NLI entailment per sentence + embedding-based context relevance
 *   7. NeuroSymbolic Scorer   — symbolic rules (rule-based by design) + real embedding cosine similarity
 *
 * Every scorer above falls back to its original lexical/heuristic method if the real
 * dependency (nliScore / embedText / secondSample / tokenLogprobs / API keys) isn't
 * available for a given call, so the pipeline degrades gracefully instead of failing.
 *
 * Then:
 *   EnsembleScorer            — weighted combination → final verdict
 *   Mitigation Agent          — block / rewrite / annotate
 */

'use strict';

const { runFactualVerification }  = require('./factualVerification');
const { runResponsibleAIPolicy }  = require('./responsibleAI');
const { runBlackBoxScorer }       = require('./blackBoxScorer');
const { runWhiteBoxScorer }       = require('./whiteBoxScorer');
const { runLLMJudge }             = require('./llmJudge');
const { runGroundednessScorer }   = require('./groundednessScorer');
const { runNeuroSymbolicScorer }  = require('./neuroSymbolicScorer');
const { runEnsembleScorer }       = require('./ensembleScorer');
const { runMitigation }           = require('./mitigation');

/**
 * Main entry point.
 *
 * @param {object} opts
 * @param {string} opts.response       – The incoming agent response text
 * @param {string} opts.context        – Source / KB context for grounding
 * @param {string} opts.question       – Original question / prompt (optional)
 * @param {Array}  opts.policies       – Active Policy documents
 * @param {string} opts.defaultAction  – 'annotate' | 'rewrite' | 'block'
 *
 * @returns {object} Full guardrail result with per-scorer breakdowns
 */
async function runGuardrailCheck({
  response, context = '', question = '', policies = [], defaultAction = 'annotate',
  // Real evaluation dependencies (all optional — each scorer falls back to a heuristic
  // if the dependency it needs isn't supplied, so this stays safe to call without them).
  nliScore = null, embedText = null, cosineSimilarity = null,
  rawResponse = null, secondSample = null, tokenLogprobs = null,
}) {
  const startTime = Date.now();

  // Content-verification scorers (does this answer's CONTENT match the source) compare
  // against the RAW, pre-masking response — [REDACTED ...] placeholders aren't part of
  // the original semantic content, and asking an NLI/LLM check to verify a placeholder
  // against the source text produces false-negative "unsupported claim" results. Confirmed
  // live: a legitimate, accurate answer with 2 masked dollar figures was hard-blocked
  // purely because the verification scorers couldn't confirm placeholder text against
  // context, not because anything was actually wrong with the answer.
  // NeuroSymbolic and ResponsibleAI intentionally keep using the MASKED response — their
  // job is checking the masking itself (e.g. "did a raw $ sign slip through"), which only
  // makes sense against what's actually shown to the user.
  const contentToVerify = rawResponse || response;

  // ── Dispatch all 8 scorers in parallel ──────────────────────────────────
  const [
    factualResult,
    policyResult,
    blackBoxResult,
    whiteBoxResult,
    judgeResult,
    groundednessResult,
    neuroSymbolicResult,
  ] = await Promise.all([
    runFactualVerification({ response: contentToVerify, context, nliScore }),
    runResponsibleAIPolicy({ response, policies }),
    runBlackBoxScorer({ response: contentToVerify, context, secondSample, embedText, cosineSimilarity }),
    runWhiteBoxScorer({ response, context, tokenLogprobs }),
    runLLMJudge({ response: contentToVerify, context, question }),
    runGroundednessScorer({ response: contentToVerify, context, question, nliScore, embedText, cosineSimilarity }),
    runNeuroSymbolicScorer({ response, context, embedText, cosineSimilarity }),
  ]);

  // ── Ensemble: combine all scorer confidences ─────────────────────────────
  const ensembleResult = await runEnsembleScorer({
    factual:       factualResult,
    blackBox:      blackBoxResult,
    whiteBox:      whiteBoxResult,
    llmJudge:      judgeResult,
    groundedness:  groundednessResult,
    neuroSymbolic: neuroSymbolicResult,
  });

  const { verdict, ensemble_score } = ensembleResult;

  // ── Mitigation ────────────────────────────────────────────────────────────
  const mitigationResult = await runMitigation({
    response,
    verdict,
    factualityScore:  ensemble_score,
    policyFlags:      policyResult.policy_flags,
    evidence:         factualResult.evidence,
    defaultAction,
  });

  return {
    // Top-level verdict (from ensemble)
    coordinator:        'HallucinationGuardCoordinator',
    verdict,
    ensemble_score,
    risk_level:         ensembleResult.risk_level,
    top_issues:         ensembleResult.top_issues,
    interpretation:     ensembleResult.interpretation,

    // Legacy fields (kept for backward compat with routes/sandbox.js)
    factuality_score:   factualResult.factuality_score,
    policy_flags:       policyResult.policy_flags,
    evidence:           factualResult.evidence,
    pii_detected:       factualResult.pii_detected,

    // Full scorer breakdowns
    scorers: {
      factual:        factualResult,
      policy:         policyResult,
      black_box:      blackBoxResult,
      white_box:      whiteBoxResult,
      llm_judge:      judgeResult,
      groundedness:   groundednessResult,
      neuro_symbolic: neuroSymbolicResult,
      ensemble:       ensembleResult,
    },

    // Mitigation
    mitigated_response:   mitigationResult.mitigated_response,
    mitigation_action:    mitigationResult.action,

    duration_ms:  Date.now() - startTime,
    sub_agents: [
      factualResult.agent,
      policyResult.agent,
      blackBoxResult.scorer,
      whiteBoxResult.scorer,
      judgeResult.scorer,
      groundednessResult.scorer,
      neuroSymbolicResult.scorer,
      ensembleResult.scorer,
      mitigationResult.agent,
    ],
  };
}

module.exports = { runGuardrailCheck };
