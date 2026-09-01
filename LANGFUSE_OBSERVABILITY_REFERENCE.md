# Langfuse Observability Reference — HM Chatbot (Hallucination Manager)

Last generated: 2026-09-01 · Owner: (unassigned — set an owner) · Review cadence: revisit on major drift re-scan, or quarterly

## Scope

This document covers LLM/agent-layer observability and evaluation via Langfuse for the RAG chatbot in this repo (`server.js` + `eval_service/`). It does **not** cover infrastructure metrics (CPU, memory, non-LLM request latency, Chroma/uvicorn process health) — Langfuse does not ingest those at all, and no separate OTel/Prometheus/Grafana path exists for them in this repo yet.

Deployment: **Langfuse Cloud**, `LANGFUSE_BASE_URL=https://cloud.langfuse.com` (see `app.yaml`). This was already the existing choice before this change; nobody has evaluated self-hosting for compliance/data-residency reasons. **Flag for follow-up**: given this app's stated purpose is redacting PII/financial figures, whether Langfuse Cloud is the right destination for *any* residual leakage risk (see Known gaps below) is a decision worth a deliberate compliance/security sign-off, not something this change decided on your behalf.

## What changed in this pass (2026-09-01)

Langfuse tracing (`instrumentation.js`, `@langfuse/otel`) and a real custom eval system (`eval_service/` — ragas, deepeval, GEval masking-completeness) already existed before this change. What was fixed/added:

1. **Critical: unmasked PII/financial data was reaching Langfuse Cloud on every chat turn.** The app's own end-user redaction ran *after* data was already sent to Langfuse generation spans, and retrieved context was never masked at all. Fixed by adding synchronous, export-time `mask` hooks on both SDKs (`instrumentation.js`'s `LangfuseSpanProcessor`, `eval_service/app.py`'s `Langfuse(mask_otel_spans=...)`), backed by a shared regex module (`lib/piiMasking.js` / `eval_service/pii_masking.py`) — see **Known gaps** below for what this does and doesn't catch.
2. **Cross-service trace linking.** `eval_service`'s `/evaluate` calls used to become their own disconnected Langfuse trace. `server.js` now passes its active trace/span IDs (`getActiveTraceId()`/`getActiveSpanId()`) to `eval_service`, which nests its `real-eval` span under the original chat-turn trace via `trace_context`.
3. **The guardrail ensemble's verdict now reaches Langfuse.** The 7-scorer pass/warn/block verdict (`agents/coordinator.js`) previously only appeared in this app's own UI. It's now pushed as `guardrail_verdict` + `guardrail_ensemble_score`, plus each of the 6 individual scorer confidences, via the Scores API.
4. **Session correlation.** `session_id` now propagates to every observation in a trace via `propagateAttributes` (there's no `userId` — this app has no end-user auth).
5. **Environment tag.** `LANGFUSE_TRACING_ENVIRONMENT` added (`app.yaml` sets `production`; set it to `development` in your local `.env`), so local dev runs don't mix with the deployed app's data in Langfuse's UI.
6. **Judge-prompt injection hardening.** `agents/llmJudge.js`'s judge prompt and `eval_service/app.py`'s two GEval criteria (masking-completeness, chunk-attribution) now explicitly frame user/document-derived content as data to evaluate, never instructions, and the XML-style delimiter values are escaped (`escapeForJudgeTag`) so a crafted `</context>` in retrieved document text can't forge tag structure — see `references/security-production.md` in the `scan-codebase-observability` skill for why this matters.
7. **Dataset seed script.** `scripts/seed_langfuse_dataset.js` — see **Dataset** below.
8. **Graceful shutdown flush.** `server.js` now handles `SIGTERM`/`SIGINT` (bounded to 5s) to flush `langfuseClient`'s queued scores and the OTel SDK's batched span exporter before the process exits on a Databricks Apps redeploy — previously both were dropped silently on every restart. **Only validated by direct function-level test on this Windows dev machine** — Windows doesn't deliver POSIX `SIGTERM` to a Node child process the same way the deployed Linux container does (`app.yaml`'s shell trap + `kill -TERM`), so confirm this actually fires on a real redeploy before fully trusting it.

An independent code review of this pass's diff also caught and this pass fixed: PII was leaking through Langfuse **score comments** (the masking-completeness/deepeval judges quote the leak they found verbatim in their `reason` text — a separate ingestion path the span-level `mask` hooks never touch, now masked explicitly at each `create_score` call site); the guardrail verdict/ensemble score were being pushed even on a network failure (now gated on `!networkBlocked`, matching the per-scorer push); and `eval_service/requirements.txt`'s `langfuse>=3.0.0` floor predated this pass's use of 4.x-only APIs (`mask_otel_spans`), raised to `>=4.14.4`.

## Where things live

- **Dashboards**: none yet — Langfuse's default Traces/Sessions/Scores views cover everything below today. Recommended follow-up: one custom dashboard combining cost/latency (native) with `guardrail_verdict`, `masking_check_passed`, and the ragas/deepeval scores side by side, organized around "is this turn safe and grounded," not a generic catch-all.
- **Dataset (golden/calibration set)**: `hm-chatbot-golden-seed` — created by running `node scripts/seed_langfuse_dataset.js` (not yet run as part of this change — it needs your real Langfuse credentials, which this change never touches). Starts with 4 hand-labeled cases migrated from `promptfoo-tests/promptfooconfig.yaml` (good-answer, correct-refusal, masking regression, jailbreak resistance). This is a **cold-start seed, not a mature calibration set** — grow it by curating real production disagreements/failures into it over time (see Known gaps).
- **CI/CD Experiments**: none wired in yet. Not recommended until the dataset above has grown well past 4 items — gating CI on 4 examples would be noise, not signal.
- **Masking rules applied**: financial figures (numeral `$1,200` and spelled-out `"one thousand two hundred dollars"`) and a fixed list of common/known person names are redacted from every trace/span's input, output, and metadata before export, via a synchronous regex pass (`lib/piiMasking.js` / `eval_service/pii_masking.py`). This runs **independently of** `MASKING_ENABLED` (the separate end-user-facing redaction flag in `server.js`, off by default in this repo right now).

## Known gaps — read before trusting this as complete

- **The Langfuse-side mask is regex-only — weaker than the full end-user pipeline.** `server.js`'s own masking (when `MASKING_ENABLED=true`) also runs an NER model (`Xenova/bert-base-NER`) to catch names never seen before. The Langfuse `mask` hooks can't do this — both SDKs require the hook to be synchronous, and NER requires an async model call. Concretely: a name like "John" (not in the common-names list, not corporate/location-excluded) will **not** be redacted before reaching Langfuse. Verified live: see the masking test in this PR's description.
- **`MASKING_ENABLED` is off by default in this repo** (a pre-existing, deliberate flag — see `server.js`'s own comment). This only affects what the *end user* sees; the Langfuse-side mask above is unconditional and always runs.
- **Historical exposure**: raw PII/financial data likely reached Langfuse Cloud on chat turns run before this fix. This change does not retroactively redact or delete anything already stored there — that's a data-retention/deletion decision for you to route through Langfuse's own retention settings or a compliance review, not something this change took action on.
- **No Langfuse-side masking test suite yet.** Recommended follow-up: extend `promptfoo-tests/` (or a small dedicated script) to fire a synthetic-PII request through the real traced path and assert the stored trace content in Langfuse doesn't contain it — this repo's `promptfoo-tests` currently test the end-user response, not what actually lands in Langfuse.
- **No cost cap on judge-model spend.** Every chat turn already triggers ~7 Claude calls (README's own known limitation) across the guardrail ensemble + eval_service, on 100% of traffic, uncapped. This is a real, exploitable cost surface (see the observability skill's cost-governance guidance) — not something this change added, but worth flagging given scores are now more visible/relied-upon.
- **Hardcoded prompts are not yet migrated into Langfuse Prompt Management** — the large RAG system prompt in `server.js`, the judge prompts in `llmJudge.js`, and the GEval criteria in `eval_service/app.py` are all still plain string literals. Recommended as a follow-up PR: without this, a future score regression can't be cleanly distinguished from "the prompt changed" vs. "the model changed" vs. real drift.

## Metrics

| Metric | Native or custom | Computed by | Langfuse score name | Where to find it | Maps to | Calibration status |
|---|---|---|---|---|---|---|
| Cost, tokens, latency | Native | Automatic from tracing | — | Traces view, per-trace columns; Sessions view for per-session rollup | Cost/latency budget | — |
| RAGAS Faithfulness | Custom | `eval_service/app.py::run_ragas` (judge: Claude Haiku) | `ragas_faithfulness` | Traces → filter by score name | Groundedness | Provisional — same judge model as the rest, no dedicated calibration set checked yet |
| RAGAS Answer Relevancy | Custom | `run_ragas` | `ragas_answer_relevancy` | Traces → filter by score name | Answer quality | Provisional |
| RAGAS Context Precision (reference-free) | Custom | `run_ragas` | `ragas_context_precision` | Traces → filter by score name | Retrieval quality | Provisional |
| DeepEval Faithfulness | Custom | `run_deepeval` | `deepeval_faithfulness` | Traces → filter by score name | Groundedness | Provisional |
| DeepEval Hallucination | Custom | `run_deepeval` | `deepeval_hallucination` | Traces → filter by score name | Groundedness | Provisional |
| DeepEval Contextual Relevancy | Custom | `run_deepeval` | `deepeval_contextual_relevancy` | Traces → filter by score name | Retrieval quality | Provisional |
| Hit Rate / MRR (LLM-judged proxy) | Custom | `run_deepeval` | `deepeval_hit_rate`, `deepeval_mrr` | Traces → filter by score name | Retrieval ranking | Provisional — not ground-truth-verified (would need a labeled "known relevant chunk" per query) |
| Masking Completeness (GEval) | Custom | `run_masking_check` | `masking_check_passed` | Traces → filter by score name | Safety/compliance | Provisional |
| Chunk Attribution Rate / Utilization | Custom | `run_chunk_attribution` | `chunk_attribution_rate`, `chunk_attribution_avg_utilization` | Traces → filter by score name | Retrieval efficiency | Provisional |
| Guardrail verdict (pass/warn/block) | Custom | `agents/ensembleScorer.js` via `agents/coordinator.js` | `guardrail_verdict` | Traces → filter by score name (categorical) | Safety gate | Provisional — newly wired to Langfuse this pass |
| Guardrail ensemble score | Custom | `agents/ensembleScorer.js` | `guardrail_ensemble_score` | Traces → filter by score name | Safety gate | Provisional |
| 6 individual guardrail scorers (factual NLI, black-box consistency, white-box confidence, LLM-judge, groundedness, neuro-symbolic) | Custom | `agents/*.js` (see `agents/coordinator.js`'s requires) | `guardrail_factual_nli`, `guardrail_black_box`, `guardrail_white_box`, `guardrail_llm_judge`, `guardrail_groundedness`, `guardrail_neurosymbolic` | Traces → filter by score name | Safety-gate breakdown | Provisional — newly wired to Langfuse this pass |

"Provisional" here means: these evaluators already ran in production before this change (this isn't a cold-start eval system), but none of them has been checked against `hm-chatbot-golden-seed` or any larger annotated set yet — do that before treating any of these as calibrated. Once checked, update this table's calibration column to "Calibrated against N dataset examples."

## What's intentionally not built

- **Tier 3 (human annotation queue for disagreement cases)** — not wired up. Recommended follow-up: route cases where the guardrail verdict and the ragas/deepeval scores disagree (e.g. guardrail says `pass` but `masking_check_passed` is false) to a Langfuse annotation queue.
- **Score-threshold alerting** — not a clearly documented native Langfuse feature as of this writing; not built here. If needed, poll the Metrics API and route to whatever alerting this org already uses.
- **New reference-free LLM-judge evaluators from the Phase D taxonomy** — deliberately not built. This app already has ragas, deepeval, a 7-scorer guardrail ensemble, and a GEval masking-completeness check covering faithfulness/hallucination/relevancy/safety/groundedness. Building a parallel set would duplicate an already-working system rather than integrate it, which the observability skill this was built from explicitly warns against.
- **CI/CD experiment gating** — not built; the seed dataset (4 items) is too small to gate on meaningfully yet.
