# Hallucination Manager (HM) Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that answers questions **only** from your uploaded documents, with a multi-layer safety system that:

- Automatically redacts PII and financial figures before any answer is shown.
- Runs an 8-part guardrail ensemble to catch hallucinations, policy violations, and low-confidence answers — and can hard-block a response if it fails.
- Gets independently scored by real `ragas`/`deepeval` metrics via Claude, plus a dedicated masking-completeness check, on every turn.
- Ships with an automated regression suite (`promptfoo`) covering good-answer, refusal, masking, and jailbreak scenarios.

## Architecture

```
Browser (client/index.html)
   │
   ▼
Node/Express backend (server.js)
   │  ├─ Local embedding + reranker + NLI + NER models (@xenova/transformers)
   │  ├─ ChromaDB (vector store — chroma_data/)
   │  ├─ Together AI (Llama 3.3 70B) — generation
   │  ├─ agents/ — 8-scorer guardrail ensemble + policy checks + mitigation
   │  └─ fire-and-forget →
   ▼
Python eval_service (eval_service/app.py)
   │  ├─ ragas (Faithfulness, Answer Relevancy, Context Precision)
   │  ├─ deepeval (Faithfulness, Hallucination, Contextual Relevancy)
   │  ├─ GEval masking-completeness check
   │  └─ judge model: Claude (claude-haiku-4-5-20251001) via ANTHROPIC_API_KEY
```

### Request flow

1. **Retrieval** — question embedded locally → Chroma top-10 → cross-encoder rerank → top-3 chunks.
2. **Generation** — Together AI answers strictly from the retrieved context (or gives a fixed refusal if nothing relevant was found).
3. **Masking** — three layers run in sequence:
   - Regex for numeral currency (`$1,200`)
   - Regex for spelled-out currency (`"one thousand two hundred dollars"`)
   - NER model (`Xenova/bert-base-NER`) for person names, with a hardcoded-list regex as a fallback layer
4. **Guardrail ensemble** (`agents/coordinator.js`) — 8 scorers run in parallel (factual/NLI, responsible-AI policy, black-box self-consistency, white-box logprob confidence, multi-provider LLM-judge, groundedness, reflection, neuro-symbolic rules) → weighted ensemble score → verdict (`pass` / `warn` / `block`).
5. **Mitigation** — a `block` verdict swaps the response for a safe refusal message and excludes the exchange from conversation history.
6. **Background evaluation** — fires a request to `eval_service`, which returns real ragas/deepeval scores plus the masking-completeness check, polled by the client and swapped into the dashboard once ready.

## Setup

### Prerequisites
- Node.js, npm
- Python 3.10+ with `pip`
- [Chroma](https://docs.trychroma.com) CLI (`pip install chromadb`)
- An Anthropic API key (used both for the LLM-judge scorer and the real evaluation service)
- A Together AI API key (used for the main chat model)

### 1. Environment variables (`.env`, project root)

| Variable | Required | Purpose |
|---|---|---|
| `TOGETHER_API_KEY` | Yes | Main chat generation (Llama 3.3 70B) |
| `EMBEDDING_MODEL_PATH` | Yes | Absolute path to the `Sentance_Transformer/` folder on **this machine** |
| `ANTHROPIC_API_KEY` | Yes | Real evaluation service judge model + LLM-judge scorer |
| `ANTHROPIC_JUDGE_MODEL` | No (default: see note below) | Must be a model that doesn't return extended-thinking blocks — `claude-haiku-4-5-20251001` confirmed working. Avoid `claude-sonnet-5` here (known incompatible, see Known Limitations). |
| `PORT` | No (default `3001`) | Node server port |
| `OPENAI_API_KEY` | No | Optional third LLM-judge provider |
| `EVAL_SERVICE_URL` | No (default `http://localhost:8500`) | Python eval service address |

> ⚠️ `EMBEDDING_MODEL_PATH` is an absolute filesystem path — it will **not** work if copied from another machine. Update it to match wherever `Sentance_Transformer/` actually lives on your setup.

### 2. Start Chroma

```powershell
chroma run --host 127.0.0.1 --port 8000 --path ./chroma_data
```

Run this from the project root specifically — running it from the wrong directory silently creates a new, empty vector store instead of connecting to your real one.

### 3. Start the Node backend

```powershell
npm install
npm start
```

Watch for `[Chroma RAG] ✓ ChromaDB connected and collections synchronized.` in the console. First boot will download several ONNX models (embedding, reranker, NLI, NER) — this can take a few minutes.

### 4. Start the Python eval service

```powershell
cd eval_service
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8500
```

### 5. Open the app

Go to `http://localhost:3001`. Make sure the mode toggle shows **"📄 PDF Mode"** (not "Knowledge Mode" — despite the label, "Knowledge Mode" actually *disables* document retrieval; "PDF Mode" is the one that uses your uploaded documents).

## Testing

An automated regression suite covers 4 scenarios: a normal grounded answer, a correct refusal, a masking-leak check, and a jailbreak-resistance check.

```powershell
cd promptfoo-tests
set ANTHROPIC_API_KEY=<your key>
npx promptfoo@latest eval -c promptfooconfig.yaml --no-cache
npx promptfoo@latest view    # visual results in browser
```

## Known limitations

- **Masking over-redacts some legitimate business terms** (e.g. "Ad Serving," "Media Partner" can get flagged as if they were person names). Safer-than-not failure direction, not yet tuned.
- **The "Clear" chat button also wipes the entire document store** (`clearChat()` calls `clearVectorDB()` internally) — this is a real, unpatched bug. Avoid clicking "Clear" if you need to keep your ingested documents; re-upload via `/api/upload` if triggered.
- **The local/free Ollama evaluation path is disabled by default** — this machine's available RAM was insufficient to run it reliably; the code path still exists in `eval_service/app.py` (`providers: ["ollama"]`) for revisiting on better hardware.
- **`ragas` is incompatible with `claude-sonnet-5`** specifically (and possibly other extended-thinking-capable Claude models) — it injects a `temperature` parameter the API rejects for those models. Use a non-thinking model like `claude-haiku-4-5-20251001` for `ANTHROPIC_JUDGE_MODEL`.
- **True RAGAS Context Recall is not implemented** — it requires a labeled ground-truth dataset that doesn't currently exist; only the reference-free Context Precision variant is available.
- **No per-request API cost limiting** — each chat message triggers ~7 Claude API calls for evaluation. Monitor usage if running at any real volume.
- **`OPENAI_API_KEY` is expected to be invalid in most setups** unless you provide your own — the LLM-judge scorer degrades gracefully to the remaining providers.
- The "Ingested Docs Library" file list reflects what's on disk in `source/`, **not** what's actually embedded in Chroma — these can drift out of sync if files are added/removed outside the normal upload flow.
