/**
 * LLM-as-a-Judge Scorer — real multi-provider ensemble
 *
 * Dispatches the SAME evaluation question to 2 independent LLMs across
 * different providers, in parallel:
 *   Judge 1 — Anthropic   (env ANTHROPIC_API_KEY, model env ANTHROPIC_JUDGE_MODEL, default "claude-sonnet-5")
 *   Judge 2 — Together AI (env TOGETHER_API_KEY,  model env TOGETHER_JUDGE_MODEL,  default "meta-llama/Llama-3.3-70B-Instruct-Turbo" — same model as chat generation, guaranteed available on the account)
 *
 * OpenAI was intentionally dropped from this ensemble per project decision — only
 * Anthropic and Together are used as judges now.
 *
 * Each judge is asked to return a strict JSON verdict: correct / partial / incorrect,
 * mapped to a 1.0 / 0.5 / 0.0 vote. Final confidence = mean of available judge votes.
 *
 * Graceful degradation: any judge whose API key is missing, or whose call fails
 * or returns an unparsable response, is silently dropped from the ensemble rather
 * than crashing the pipeline — confidence is averaged over whichever judges
 * actually responded. If none respond, a neutral 0.5 is returned with a note.
 */

'use strict';

const axios = require('axios');
const https = require('https');
// Free (non-method) startObservation — picks up whatever Langfuse trace is "active" via
// OpenTelemetry's async-context propagation at call time (server.js wraps the whole
// /api/chat handler in startActiveObservation('chat-turn', ...), so these two calls land
// as child generations of that trace with no explicit parent object threaded through
// coordinator.js/runGuardrailCheck). If no trace is active (e.g. this module is invoked
// outside that context), these silently start their own standalone trace instead of
// throwing — same graceful-degradation posture as the rest of this file.
const { startObservation } = require('@langfuse/tracing');

const httpsAgent = new https.Agent({ keepAlive: true, rejectUnauthorized: false });

const VERDICT_VOTES = { correct: 1.0, partial: 0.5, incorrect: 0.0 };

const JUDGE_SYSTEM_PROMPT =
  'You are an impartial evaluation judge for a RAG chatbot. You will be given a QUESTION, ' +
  'the CONTEXT documents the answer should be grounded in, and a RESPONSE to evaluate. ' +
  'Judge whether the RESPONSE is factually correct and fully supported by the CONTEXT ' +
  '(or, if CONTEXT is empty, whether it correctly declines to answer rather than inventing facts). ' +
  'Reply with STRICT JSON only, no other text, no markdown fences: ' +
  '{"verdict": "correct" | "partial" | "incorrect", "rationale": "<one concise sentence>"}';

function buildJudgeUserPrompt({ question, context, response }) {
  return `QUESTION:\n${question || '(none provided)'}\n\n` +
    `CONTEXT:\n${context?.trim() ? context.slice(0, 4000) : '(no context retrieved)'}\n\n` +
    `RESPONSE TO EVALUATE:\n${response}`;
}

function parseJudgeVerdict(rawText, provider, model) {
  let parsed = null;
  try {
    const jsonMatch = rawText.match(/\{[\s\S]*\}/);
    parsed = JSON.parse(jsonMatch ? jsonMatch[0] : rawText);
  } catch {
    const lower = rawText.toLowerCase();
    const verdict = lower.includes('incorrect') ? 'incorrect'
      : lower.includes('partial') ? 'partial'
        : lower.includes('correct') ? 'correct' : null;
    parsed = verdict ? { verdict, rationale: rawText.slice(0, 200) } : null;
  }
  if (!parsed || !(parsed.verdict in VERDICT_VOTES)) return null;
  return {
    judge:     `${provider} (${model})`,
    vote:      VERDICT_VOTES[parsed.verdict],
    verdict:   parsed.verdict,
    rationale: parsed.rationale || `Verdict: ${parsed.verdict}`,
  };
}

async function callAnthropicJudge(promptArgs) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return null;
  const model = process.env.ANTHROPIC_JUDGE_MODEL || 'claude-sonnet-5';
  const userPrompt = buildJudgeUserPrompt(promptArgs);
  const generation = startObservation(
    'llm-judge-anthropic',
    { model, input: [{ role: 'system', content: JUDGE_SYSTEM_PROMPT }, { role: 'user', content: userPrompt }] },
    { asType: 'generation' }
  );
  try {
    const r = await axios.post('https://api.anthropic.com/v1/messages', {
      model,
      max_tokens: 200,
      system: JUDGE_SYSTEM_PROMPT,
      messages: [{ role: 'user', content: userPrompt }],
    }, {
      headers: { 'x-api-key': apiKey, 'anthropic-version': '2023-06-01', 'Content-Type': 'application/json' },
      timeout: 60000, httpsAgent,
    });
    const text = (r.data.content || []).map(c => c.text || '').join('');
    const verdict = parseJudgeVerdict(text, 'Anthropic', model);
    generation.update({
      output: text,
      usageDetails: r.data.usage ? { promptTokens: r.data.usage.input_tokens, completionTokens: r.data.usage.output_tokens } : undefined,
    });
    generation.end();
    return verdict;
  } catch (err) {
    console.error('[LLM Judge — Anthropic Error]:', err.response?.data?.error?.message || err.message);
    generation.update({ level: 'ERROR', statusMessage: err.message });
    generation.end();
    return null;
  }
}

async function callTogetherJudge(promptArgs) {
  const apiKey = process.env.TOGETHER_API_KEY;
  if (!apiKey) return null;
  // Defaults to the same model already used for chat generation, since that one is
  // confirmed available on the account's plan. Override TOGETHER_JUDGE_MODEL with a
  // different serverless (non-dedicated-endpoint) model for more ensemble diversity —
  // check availability at https://api.together.ai/models first.
  const model = process.env.TOGETHER_JUDGE_MODEL || 'meta-llama/Llama-3.3-70B-Instruct-Turbo';
  const messages = [
    { role: 'system', content: JUDGE_SYSTEM_PROMPT },
    { role: 'user', content: buildJudgeUserPrompt(promptArgs) },
  ];
  const generation = startObservation(
    'llm-judge-together',
    { model, input: messages, modelParameters: { temperature: 0, maxTokens: 200 } },
    { asType: 'generation' }
  );
  try {
    const r = await axios.post('https://api.together.xyz/v1/chat/completions', {
      model,
      messages,
      temperature: 0,
      max_tokens: 200,
    }, { headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' }, timeout: 60000, httpsAgent });
    const text = r.data.choices?.[0]?.message?.content || '';
    const verdict = parseJudgeVerdict(text, 'Together', model);
    generation.update({
      output: text,
      usageDetails: r.data.usage ? {
        promptTokens: r.data.usage.prompt_tokens, completionTokens: r.data.usage.completion_tokens, totalTokens: r.data.usage.total_tokens,
      } : undefined,
    });
    generation.end();
    return verdict;
  } catch (err) {
    console.error('[LLM Judge — Together Error]:', err.response?.data?.error?.message || err.message);
    generation.update({ level: 'ERROR', statusMessage: err.message });
    generation.end();
    return null;
  }
}

// ── Main scorer ───────────────────────────────────────────────────────────────
async function runLLMJudge({ response, context = '', question = '' }) {
  const promptArgs = { question, context, response };

  const settled = await Promise.all([
    callAnthropicJudge(promptArgs),
    callTogetherJudge(promptArgs),
  ]);
  const judges = settled.filter(Boolean);

  if (judges.length === 0) {
    return {
      scorer:      'LLMasaJudge',
      method:      'no_providers_available',
      judge_count: 0,
      judges:      [],
      votes:       { correct: 0, partial: 0, incorrect: 0 },
      confidence:  0.5,
      interpretation: 'No judge LLM calls succeeded (missing API keys or all requests failed) — neutral default returned.',
    };
  }

  const confidence = parseFloat((judges.reduce((sum, j) => sum + j.vote, 0) / judges.length).toFixed(3));
  const correct   = judges.filter(j => j.vote === 1.0).length;
  const partial   = judges.filter(j => j.vote === 0.5).length;
  const incorrect = judges.filter(j => j.vote === 0.0).length;

  return {
    scorer:      'LLMasaJudge',
    method:      'real_multi_llm_judge',
    judge_count: judges.length,
    judges,
    votes:       { correct, partial, incorrect },
    confidence,
    interpretation: confidence >= 0.75
      ? 'Majority judges agree response is correct'
      : confidence >= 0.45
        ? 'Mixed verdict — partial agreement across judges'
        : 'Majority judges flag this response as incorrect',
  };
}

module.exports = { runLLMJudge };
