/**
 * LLM-as-a-Judge Scorer — real multi-provider ensemble
 *
 * Dispatches the SAME evaluation question to 3 independent LLMs across
 * different providers, in parallel:
 *   Judge 1 — OpenAI      (env OPENAI_API_KEY,   model env OPENAI_JUDGE_MODEL,   default "gpt-4o-mini")
 *   Judge 2 — Anthropic   (env ANTHROPIC_API_KEY, model env ANTHROPIC_JUDGE_MODEL, default "claude-sonnet-5")
 *   Judge 3 — Together AI (env TOGETHER_API_KEY,  model env TOGETHER_JUDGE_MODEL,  default "meta-llama/Llama-3.3-70B-Instruct-Turbo" — same model as chat generation, guaranteed available on the account)
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

async function callOpenAiJudge(promptArgs) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) return null;
  const model = process.env.OPENAI_JUDGE_MODEL || 'gpt-4o-mini';
  try {
    const r = await axios.post('https://api.openai.com/v1/chat/completions', {
      model,
      messages: [
        { role: 'system', content: JUDGE_SYSTEM_PROMPT },
        { role: 'user', content: buildJudgeUserPrompt(promptArgs) },
      ],
      temperature: 0,
      max_tokens: 200,
    }, { headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' }, timeout: 20000, httpsAgent });
    return parseJudgeVerdict(r.data.choices?.[0]?.message?.content || '', 'OpenAI', model);
  } catch (err) {
    console.error('[LLM Judge — OpenAI Error]:', err.response?.data?.error?.message || err.message);
    return null;
  }
}

async function callAnthropicJudge(promptArgs) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return null;
  const model = process.env.ANTHROPIC_JUDGE_MODEL || 'claude-sonnet-5';
  try {
    const r = await axios.post('https://api.anthropic.com/v1/messages', {
      model,
      max_tokens: 200,
      system: JUDGE_SYSTEM_PROMPT,
      messages: [{ role: 'user', content: buildJudgeUserPrompt(promptArgs) }],
    }, {
      headers: { 'x-api-key': apiKey, 'anthropic-version': '2023-06-01', 'Content-Type': 'application/json' },
      timeout: 20000, httpsAgent,
    });
    const text = (r.data.content || []).map(c => c.text || '').join('');
    return parseJudgeVerdict(text, 'Anthropic', model);
  } catch (err) {
    console.error('[LLM Judge — Anthropic Error]:', err.response?.data?.error?.message || err.message);
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
  try {
    const r = await axios.post('https://api.together.xyz/v1/chat/completions', {
      model,
      messages: [
        { role: 'system', content: JUDGE_SYSTEM_PROMPT },
        { role: 'user', content: buildJudgeUserPrompt(promptArgs) },
      ],
      temperature: 0,
      max_tokens: 200,
    }, { headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' }, timeout: 20000, httpsAgent });
    return parseJudgeVerdict(r.data.choices?.[0]?.message?.content || '', 'Together', model);
  } catch (err) {
    console.error('[LLM Judge — Together Error]:', err.response?.data?.error?.message || err.message);
    return null;
  }
}

// ── Main scorer ───────────────────────────────────────────────────────────────
async function runLLMJudge({ response, context = '', question = '' }) {
  const promptArgs = { question, context, response };

  const settled = await Promise.all([
    callOpenAiJudge(promptArgs),
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
