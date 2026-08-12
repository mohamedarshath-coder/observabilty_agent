"""
HM Chatbot — Real Evaluation Microservice

Wraps the ACTUAL `ragas` and `deepeval` Python libraries (there is no mature JS port,
so this runs as a separate process the Node backend calls over HTTP). Both libraries
need an LLM internally to decompose answers into claims and verify each one against
the retrieved context — that's what produces the scores. Configured here to use a
locally running Ollama model instead of a paid API, per project decision.

Metrics computed:
  - RAGAS    Faithfulness         (answer claims vs. retrieved context)
  - RAGAS    Answer Relevancy     (does the answer address the question)
  - RAGAS    Context Precision*   (were the retrieved chunks actually useful for the answer)
  - DeepEval Faithfulness         (contradiction-focused, retrieval_context-based)
  - DeepEval Hallucination        (actual_output vs. context)
  - DeepEval Contextual Relevancy* (are the retrieved chunks relevant to the question)

  * Reference-free variants — scored from (question, answer, context) alone, no ground-truth
    answer required. True Context Recall ("did retrieval miss anything relevant that exists in
    the source docs") needs a labeled ground-truth dataset to check against and isn't
    computable from a single live request — add it once a labeled eval set exists.

Run standalone (see README section at the bottom of this file for full setup):
    uvicorn app:app --host 0.0.0.0 --port 8500

Node's server.js calls this asynchronously (fire-and-forget) after already returning
the chat answer to the user, then the client polls /api/real-eval/:id for the result.
"""

import os
import re
import math
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from pydantic import BaseModel
from datasets import Dataset

try:
    # Node's server.js loads the root .env via dotenv; this process is separate
    # (started via `uvicorn`), so it needs its own load to see the same keys
    # (e.g. ANTHROPIC_API_KEY) without requiring them to be exported manually.
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hm-eval-service")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_JUDGE_MODEL = os.environ.get("OLLAMA_JUDGE_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
# Matches the default already used by agents/llmJudge.js on the Node side, for consistency.
ANTHROPIC_JUDGE_MODEL = os.environ.get("ANTHROPIC_JUDGE_MODEL", "claude-sonnet-5")

SUPPORTED_PROVIDERS = ("ollama", "claude")

# deepeval's internal per-attempt timeout for LLM calls defaults to ~88s — too short
# for a CPU-hosted Ollama model, which was observed timing out and retrying/failing
# instead of just taking longer. Must be set before deepeval's metrics run.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "300")

app = FastAPI(title="HM Chatbot Real Eval Service")

# Cross-provider calls are blocking (each makes several sequential LLM round trips
# inside ragas/deepeval); run one provider per thread so "ollama" and "claude"
# actually execute concurrently instead of back-to-back within a SINGLE request.
# This pool is process-global, though — shared across ALL incoming /evaluate calls,
# not per-request. Sizing it at len(SUPPORTED_PROVIDERS)==2 caused a real, confirmed
# bug: with several chat messages fired close together (each spawning its own
# background real-eval call), only 2 could run at once; the rest queued long enough
# to exceed Node's client-side timeout even though eval_service itself never errored.
# 8 gives headroom for several concurrent chat requests, each still using ≤2 slots.
_executor = ThreadPoolExecutor(max_workers=8)


class EvalRequest(BaseModel):
    question: str
    answer: str
    context: str
    # Which judge(s) to score with, run in parallel. Defaults to just "ollama" to
    # keep existing callers working unchanged.
    providers: list[str] = ["ollama"]
    # The MASKED response actually shown to the user (regex-redacted PII/financial
    # figures) — separate from `answer` above, which stays the raw pre-masking text
    # so faithfulness/hallucination scoring isn't confused by "[REDACTED...]" markers.
    # Optional: falls back to `answer` if not provided, so older callers still work,
    # though the masking check is meaningless against unmasked text in that case.
    masked_answer: str = ""


# ── RAGAS setup ──────────────────────────────────────────────────────────────────
# Wrapped in functions (not run at import time) so the FastAPI process still starts
# and serves /health even if Ollama/Anthropic isn't reachable yet or ragas isn't
# fully installed — errors surface per-request in `ragas_error` instead of crashing
# the whole service. Cached per-provider so repeated requests don't rebuild the
# LLM/embeddings wrapper each time.
_ragas_cache = {}  # provider -> {"llm":..., "embeddings":..., "metrics":[...]}


def _build_ragas_judge_llm(provider: str):
    from ragas.llms import LangchainLLMWrapper

    if provider == "claude":
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set — cannot use the claude provider.")
        from langchain_anthropic import ChatAnthropic
        # No explicit temperature — this Claude model rejects temperature=0 with
        # "`temperature` is deprecated for this model" (observed live), unlike Ollama
        # which needs temperature=0 for deterministic judging. Let it use its default.
        return LangchainLLMWrapper(
            ChatAnthropic(model=ANTHROPIC_JUDGE_MODEL, api_key=ANTHROPIC_API_KEY)
        )

    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        from langchain_community.chat_models import ChatOllama
    return LangchainLLMWrapper(
        ChatOllama(model=OLLAMA_JUDGE_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    )


def _build_ragas_embeddings():
    # Anthropic has no embeddings API, so both providers share the same local Ollama
    # embedding model — only the judge LLM differs between "ollama" and "claude".
    from ragas.embeddings import LangchainEmbeddingsWrapper
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        from langchain_community.embeddings import OllamaEmbeddings
    return LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    )


def _build_ragas_metrics():
    # API surface differs across ragas versions — try the modern class-based metrics
    # first, fall back to the legacy pre-instantiated singletons.
    try:
        from ragas.metrics import Faithfulness, AnswerRelevancy
        metrics = [Faithfulness(), AnswerRelevancy()]
    except ImportError:
        from ragas.metrics import faithfulness, answer_relevancy
        metrics = [faithfulness, answer_relevancy]

    # Reference-free context precision — judges whether the retrieved chunks were
    # actually necessary/useful for the answer, without needing a ground-truth answer.
    # Only available in newer ragas; skip silently on older versions rather than
    # failing the whole evaluation.
    try:
        from ragas.metrics import LLMContextPrecisionWithoutReference
        metrics.append(LLMContextPrecisionWithoutReference())
    except ImportError:
        logger.warning("LLMContextPrecisionWithoutReference unavailable in this ragas version — skipping context precision.")
    return metrics


def _init_ragas(provider: str):
    if provider in _ragas_cache:
        return _ragas_cache[provider]
    entry = {
        "llm": _build_ragas_judge_llm(provider),
        "embeddings": _build_ragas_embeddings(),
        "metrics": _build_ragas_metrics(),
    }
    _ragas_cache[provider] = entry
    return entry


def run_ragas(question: str, answer: str, contexts: list, provider: str = "ollama"):
    from ragas import evaluate
    # ragas's own per-job timeout also defaults too low for a CPU-hosted Ollama model
    # (observed: "Exception raised in Job[N]: TimeoutError()"). RunConfig's exact import
    # path has moved between ragas versions, so try a couple before giving up and just
    # running with ragas's default (still better than crashing this whole call).
    run_config = None
    try:
        from ragas.run_config import RunConfig
        run_config = RunConfig(timeout=300)
    except ImportError:
        try:
            from ragas import RunConfig
            run_config = RunConfig(timeout=300)
        except ImportError:
            logger.warning("Could not import ragas RunConfig to raise its default timeout — using ragas's built-in default, which may be too short for a CPU-hosted Ollama model.")

    components = _init_ragas(provider)
    ds = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    })
    evaluate_kwargs = {"run_config": run_config} if run_config is not None else {}
    result = evaluate(
        ds,
        metrics=components["metrics"],
        llm=components["llm"],
        embeddings=components["embeddings"],
        **evaluate_kwargs,
    )
    df = result.to_pandas()
    row = df.iloc[0]

    def _col(*names, default=0.0):
        for name in names:
            if name in row and row[name] is not None:
                return float(row[name])
        return default

    scores = {
        "faithfulness": _col("faithfulness", "Faithfulness"),
        "answer_relevancy": _col("answer_relevancy", "AnswerRelevancy"),
    }
    # Only present if LLMContextPrecisionWithoutReference loaded successfully.
    if any(m.__class__.__name__ == "LLMContextPrecisionWithoutReference" for m in components["metrics"]):
        scores["context_precision"] = _col(
            "llm_context_precision_without_reference", "context_precision", default=None
        )
    return scores


# ── DeepEval setup ───────────────────────────────────────────────────────────────
# For the "ollama" provider: requires a ONE-TIME setup step before starting this
# service so the metrics default to Ollama instead of trying OpenAI:
#   deepeval set-ollama llama3.1 --base-url="http://localhost:11434"
# For the "claude" provider: no global config needed — an explicit AnthropicModel
# instance is passed to each metric, scoped to just that call.
# See the setup notes at the bottom of this file.
_deepeval_claude_model = None
_deepeval_ollama_model = None


def _build_deepeval_model(provider: str):
    """Returns a model object to pass to metric(model=...). Explicit for both
    providers — relying on the global `deepeval set-ollama` CLI config for "ollama"
    turned out to be unreliable (observed falling through to OpenAI's default model
    internally on deepeval>=4, causing a 401 when OPENAI_API_KEY is invalid/missing,
    instead of actually using Ollama). Building the model object explicitly here
    avoids depending on that global state."""
    global _deepeval_claude_model, _deepeval_ollama_model
    if provider == "ollama":
        if _deepeval_ollama_model is None:
            try:
                from deepeval.models import OllamaModel
                _deepeval_ollama_model = OllamaModel(model=OLLAMA_JUDGE_MODEL, base_url=OLLAMA_BASE_URL)
            except ImportError:
                # Older/newer deepeval without this class — fall back to the global
                # `deepeval set-ollama` config by passing no explicit model. If that
                # config isn't actually applied, this call will hit whatever deepeval
                # defaults to (possibly OpenAI) and surface as a clear deepeval_error
                # for this provider rather than crashing the whole request.
                logger.warning("deepeval.models.OllamaModel unavailable — falling back to global `deepeval set-ollama` config, which may not actually be applied on this deepeval version.")
                return None
        return _deepeval_ollama_model
    if provider == "claude":
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set — cannot use the claude provider.")
        if _deepeval_claude_model is None:
            # Class name/import path per deepeval's third-party-model integration docs;
            # if this import fails on your installed deepeval version, the error surfaces
            # in `deepeval_error` for that provider rather than crashing the service —
            # run `pip install -U deepeval` if it's missing AnthropicModel.
            from deepeval.models import AnthropicModel
            _deepeval_claude_model = AnthropicModel(model=ANTHROPIC_JUDGE_MODEL, api_key=ANTHROPIC_API_KEY)
        return _deepeval_claude_model
    raise ValueError(f"Unknown provider: {provider}")


def run_deepeval(question: str, answer: str, contexts: list, provider: str = "ollama"):
    from deepeval.metrics import (
        FaithfulnessMetric,
        ContextualRelevancyMetric,
    )
    from deepeval.test_case import LLMTestCase

    model = _build_deepeval_model(provider)
    metric_kwargs = {"threshold": 0.5, "include_reason": True}
    if model is not None:
        metric_kwargs["model"] = model

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=contexts,
        context=contexts,
    )

    faithfulness_metric = FaithfulnessMetric(**metric_kwargs)
    # Reference-free — judges whether the retrieved chunks are relevant to the
    # question, no ground-truth expected_output required (unlike ContextualPrecision/
    # ContextualRecallMetric, which do require one and aren't usable without a
    # labeled eval set).
    relevancy_metric = ContextualRelevancyMetric(**metric_kwargs)

    faithfulness_metric.measure(test_case)
    relevancy_metric.measure(test_case)

    # Hallucination is DERIVED from Faithfulness (1 - faithfulness) rather than measured
    # via deepeval's own HallucinationMetric — confirmed live, twice, that HallucinationMetric
    # gives unreliable results in this RAG setup: it treats the whole context as a single
    # reference and checks whether the whole answer stays consistent with it, rather than
    # extracting and verifying individual claims the way FaithfulnessMetric does. This
    # produced contradictory results against manually-verified ground truth on both a
    # refusal (scored 100% hallucination on a claim-free non-answer) and a legitimate,
    # fully-accurate multi-item list (scored 48% hallucination on an answer verified
    # correct). FaithfulnessMetric was accurate in every test run today, so its inverse is
    # a more trustworthy Hallucination signal than deepeval's own dedicated metric for this
    # answer shape — at the cost of no longer being a genuinely independent second check.
    hallucination_score = round(1 - faithfulness_metric.score, 4)

    return {
        "faithfulness": faithfulness_metric.score,
        "faithfulness_reason": faithfulness_metric.reason,
        "hallucination": hallucination_score,
        "hallucination_reason": f"Derived as (1 - Faithfulness) rather than measured independently — see code comment for why. Faithfulness reason: {faithfulness_metric.reason}",
        "contextual_relevancy": relevancy_metric.score,
        "contextual_relevancy_reason": relevancy_metric.reason,
    }


# ── Masking completeness check (GEval) ────────────────────────────────────────────
# None of the 6 metrics above check this — they judge faithfulness/relevancy against
# the source, not whether the regex-based PII/financial masking in server.js actually
# caught everything it should have. This is a custom rubric, not a stock metric,
# because "did anything sensitive leak past masking" isn't a generic RAG quality
# question — it's specific to this project's compliance requirement.
_masking_criteria = (
    "The TEXT below is a chatbot response that has already been through an automated "
    "PII/financial masking step, which replaces sensitive spans with placeholders like "
    "[REDACTED PII / NAME Block] or [REDACTED COST/REVENUE Metric]. "
    "Carefully check whether any sensitive information was MISSED by that masking step: "
    "(1) any dollar/currency amount in ANY form — numerals ($1,200), words ('one thousand "
    "two hundred dollars'), or abbreviations (1.2k) — not replaced by a redaction placeholder; "
    "(2) any specific person's first or last name (not a company name, product name, or "
    "generic role title) that was not replaced by a redaction placeholder. "
    "Existing [REDACTED ...] placeholders are correct and should NOT be flagged — only "
    "flag information that is still present in plain, readable form. "
    "SCORING: this is a strict compliance check, not a partial-credit quality score. "
    "If you identify even ONE piece of unredacted sensitive information — regardless of "
    "format, obviousness, or severity — the score MUST be below 0.3. A score above 0.7 "
    "means you found zero leaks of any kind. There is no acceptable middle ground: a "
    "single missed name or dollar figure is a compliance failure, not a minor deduction."
)


def run_masking_check(masked_answer: str, provider: str = "claude"):
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    model = _build_deepeval_model(provider)
    # GEval's constructor differs from the other metrics — no include_reason kwarg
    # (it always includes a reason) and it accepts model separately.
    metric_kwargs = {"threshold": 0.5}
    if model is not None:
        metric_kwargs["model"] = model

    metric = GEval(
        name="Masking Completeness",
        criteria=_masking_criteria,
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        # Higher score = better (nothing leaked). Threshold 0.5: below it means the
        # judge found at least one plausible leak worth flagging.
        **metric_kwargs,
    )
    test_case = LLMTestCase(input="(masking check — no question context needed)", actual_output=masked_answer)
    metric.measure(test_case)

    return {
        "passed": metric.score >= 0.5,
        "score": metric.score,
        "reason": metric.reason,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def _sanitize_json(obj):
    """Recursively replaces NaN/Infinity floats with None. Both ragas and deepeval
    can return NaN for a metric when their internal LLM call fails/degenerates
    (observed: deepeval's FaithfulnessMetric returning NaN after an upstream 401)
    rather than raising — and Starlette's JSONResponse uses allow_nan=False, so a
    single NaN anywhere in the payload previously crashed the ENTIRE /evaluate
    response with a 500, even when the other provider's results were fine."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


# Mirrors server.js's own isRefusalText() — kept as a separate copy since this runs in a
# different process/language, not because the logic should ever diverge intentionally.
_REFUSAL_PATTERN = re.compile(
    r"\b(does not contain|no information|no mention|not covered|cannot find|no relevant|no data|i am sorry|i'm sorry)\b",
    re.IGNORECASE,
)


def _is_refusal_text(text: str) -> bool:
    return bool(_REFUSAL_PATTERN.search(text or ""))


def _evaluate_with_provider(question: str, answer: str, contexts: list, provider: str, masked_answer: str):
    # A refusal makes zero factual claims, so there is nothing for DeepEval's
    # claim-extraction step to check — and confirmed live that it doesn't handle this
    # gracefully: an earlier identically-shaped refusal scored Hallucination 0% (correct),
    # while a different refusal phrasing scored 100% (wrong) purely from how DeepEval's
    # extraction happened to parse that specific sentence. Rather than rely on an LLM call
    # to correctly judge "did this non-answer hallucinate," short-circuit with the values
    # that are definitionally correct for any refusal: fully faithful (nothing to
    # contradict), zero context relevance (nothing in the docs was actually used).
    if _is_refusal_text(answer):
        ragas_scores = {"faithfulness": 1.0, "answer_relevancy": 0.0, "context_precision": 0.0}
        ragas_error = None
        deepeval_scores = {
            "faithfulness": 1.0,
            "faithfulness_reason": "Refusal detected — no claims were made, so nothing can contradict the context.",
            "hallucination": 0.0,
            "hallucination_reason": "Refusal detected — no claims were made, so nothing can be fabricated.",
            "contextual_relevancy": 0.0,
            "contextual_relevancy_reason": "Refusal detected — the retrieved context did not contain an answer, so it was correctly not used.",
        }
        deepeval_error = None
    else:
        try:
            ragas_scores = _sanitize_json(run_ragas(question, answer, contexts, provider))
            ragas_error = None
        except Exception as e:
            logger.exception(f"ragas evaluation failed (provider={provider})")
            ragas_scores = None
            ragas_error = str(e)

        try:
            deepeval_scores = _sanitize_json(run_deepeval(question, answer, contexts, provider))
            deepeval_error = None
        except Exception as e:
            logger.exception(f"deepeval evaluation failed (provider={provider})")
            deepeval_scores = None
            deepeval_error = str(e)

    try:
        masking_check = _sanitize_json(run_masking_check(masked_answer, provider))
        masking_check_error = None
    except Exception as e:
        logger.exception(f"masking check failed (provider={provider})")
        masking_check = None
        masking_check_error = str(e)

    return {
        "ragas": ragas_scores,
        "ragas_error": ragas_error,
        "deepeval": deepeval_scores,
        "deepeval_error": deepeval_error,
        "masking_check": masking_check,
        "masking_check_error": masking_check_error,
    }


@app.post("/evaluate")
def evaluate_turn(req: EvalRequest):
    contexts = [req.context] if req.context.strip() else ["(no context retrieved)"]
    masked_answer = req.masked_answer.strip() or req.answer

    providers = [p for p in req.providers if p in SUPPORTED_PROVIDERS] or ["ollama"]
    unknown = [p for p in req.providers if p not in SUPPORTED_PROVIDERS]
    if unknown:
        logger.warning(f"Ignoring unsupported provider(s) in request: {unknown}")

    # Each provider's own ragas+deepeval calls are already sequential internally;
    # running the providers themselves in parallel threads is what actually makes
    # an "ollama" vs "claude" comparison call take roughly as long as either one
    # alone, instead of the sum of both.
    futures = {
        provider: _executor.submit(_evaluate_with_provider, req.question, req.answer, contexts, provider, masked_answer)
        for provider in providers
    }
    per_provider = {provider: future.result() for provider, future in futures.items()}

    # Keep the original top-level shape (ragas/ragas_error/deepeval/deepeval_error)
    # pointing at the first requested provider, so existing callers reading those
    # fields directly keep working unchanged. `providers_compared` carries the full
    # per-provider breakdown for side-by-side comparison.
    primary = per_provider[providers[0]]
    return {
        **primary,
        "providers_compared": per_provider,
    }


# ─────────────────────────────────────────────────────────────────────────────────
# SETUP (one time):
#   1. Activate the project's existing venv:
#        ..\venv\Scripts\Activate.ps1        (PowerShell)
#   2. Install this service's dependencies:
#        pip install -r requirements.txt
#   3. Install Ollama (https://ollama.com) and pull the two models used here:
#        ollama pull llama3.1
#        ollama pull nomic-embed-text
#   4. One-time deepeval config so the "ollama" provider defaults to Ollama instead
#      of trying OpenAI:
#        deepeval set-ollama llama3.1 --base-url="http://localhost:11434"
#   5. (Optional, for the "claude" provider) Set ANTHROPIC_API_KEY in the project's
#      root .env (same file Node's server.js already reads) — this service loads it
#      from there automatically on startup.
#
# RUN (every time, alongside `chroma run` and `npm start`):
#        uvicorn app:app --host 0.0.0.0 --port 8500
#
# COMPARE PROVIDERS: POST /evaluate with {"providers": ["ollama", "claude"]} to
# score the same (question, answer, context) with both judges in parallel — the
# response's `providers_compared` field has each provider's scores side by side.
# ─────────────────────────────────────────────────────────────────────────────────
