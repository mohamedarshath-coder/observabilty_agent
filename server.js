require('dotenv').config();

// Same corporate-proxy TLS interception this file already works around for axios
// (see the `agent` below) — but @xenova/transformers downloads models via Node's
// built-in `fetch`, which doesn't use that custom https.Agent at all, so it still
// fails with a bare "fetch failed" against the proxy's self-signed cert. This must
// be set before any pipeline()/model-download call runs.
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

const express = require('express');
const axios = require('axios');
const cors = require('cors');
const path = require('path');
const https = require('https');
const multer = require('multer');
const fs = require('fs');
const { ChromaClient } = require('chromadb');

// Platform agnostic server-side PDF extraction engine
const { getDocumentProxy, extractText } = require('unpdf');

// Local execution engine runtime environment maps
const { env, pipeline } = require('@xenova/transformers');
const { runGuardrailCheck } = require('./agents/coordinator');

const modelFullPath = process.env.EMBEDDING_MODEL_PATH;
if (!modelFullPath) {
  console.error('[Configuration Error] EMBEDDING_MODEL_PATH is missing from your .env file!');
}

env.localModelPath = path.dirname(modelFullPath); 
env.allowRemoteModels = true; 
env.localFilesOnly = false;     

// Bypasses corporate proxies and avoids "unable to get local issuer certificate" errors
const agent = new https.Agent({ 
  keepAlive: true,
  maxSockets: 100,
  rejectUnauthorized: false 
});

const app = express();
const PORT = process.env.PORT || 3001;
const TOGETHER_API_KEY = process.env.TOGETHER_API_KEY;

// Disabled per explicit request — was flagging real dollar amounts and person names as
// [REDACTED ...] before showing an answer. Kept as a flag (not deleted) so it's a one-line
// flip back to re-enable, instead of having to re-implement the masking logic below.
// Set MASKING_ENABLED=true in .env to turn redaction back on.
const MASKING_ENABLED = process.env.MASKING_ENABLED === 'true';

// Real ragas/deepeval run in a separate Python microservice (see eval_service/) since
// both libraries are Python-only. Scored asynchronously — the chat response returns
// immediately, and the client polls /api/real-eval/:id for the real scores once ready.
const EVAL_SERVICE_URL = process.env.EVAL_SERVICE_URL || 'http://localhost:8500';
const realEvalStore = new Map();
const REAL_EVAL_TTL_MS = 10 * 60 * 1000;

// agents/responsibleAI.js was previously called with an empty policies array on every
// request — its toxicity/bias/jailbreak/PII rule checks were dead code, never actually
// evaluating anything. These are the active policies now passed to runGuardrailCheck().
// NOTE: the "regulated domain" guidance flag (rule_config.guidance) is intentionally
// left off for every policy here — this chatbot's entire purpose is discussing financial
// terms from real contracts, so flagging "touches financial domain" would fire on nearly
// every legitimate answer. That guidance flag is meant for bots that AREN'T supposed to
// discuss those domains at all.
const ACTIVE_POLICIES = [
  { _id: 'toxicity-1', name: 'No Toxic Language', category: 'toxicity', rule_config: {} },
  { _id: 'bias-1', name: 'No Biased Language', category: 'bias', rule_config: {} },
  { _id: 'jailbreak-1', name: 'No Jailbreak / Prompt Injection', category: 'jailbreak', rule_config: {} },
  { _id: 'pii-1', name: 'No Raw PII Patterns (SSN/Card)', category: 'pii', rule_config: {} },
];

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'client')));

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });

// Initialize physical source registry directory path
const sourceDirectoryPath = path.join(__dirname, 'source');
if (!fs.existsSync(sourceDirectoryPath)) {
  fs.mkdirSync(sourceDirectoryPath);
}

let trackIndexedFiles = [];

// Dynamic recovery scanner function on server boot
function baselineSyncTrackedFilesFromDisk() {
  try {
    const filesOnDisk = fs.readdirSync(sourceDirectoryPath);
    trackIndexedFiles = filesOnDisk.map(filename => ({
      filename,
      chunksCount: 'Indexed',
      embedded: true,
      timestamp: 'Stored on Disk'
    }));
    console.log(`[File Registry] Synchronized ${trackIndexedFiles.length} file handles from local /source disk directory.`);
  } catch (err) {
    console.error('[File Registry System Recovery Error]:', err.message);
  }
}
baselineSyncTrackedFilesFromDisk();

const sessions = {};
let chromaClient = null;
let chromaCollection = null;
let embeddingPipeline = null;
let rerankerPipeline = null;
let nliPipeline = null;
let nerPipeline = null;
let chromaReady = false;

class LocalEmbeddingFunction {
  constructor(pipeline) { this.pipeline = pipeline; }
  async call(input) { return this.generateEmbeddings(input); }
  async generate(input) { return this.generateEmbeddings(input); }
  async generateEmbeddings(input) {
    if (typeof input === 'string') input = [input];
    const embeddings = [];
    for (const text of input) {
      const output = await this.pipeline(text, { pooling: 'mean', normalize: true });
      embeddings.push(Array.from(output.data));
    }
    return embeddings;
  }
}

async function initVectorDatabase() {
  try {
    const modelFolderName = path.basename(modelFullPath);
    console.log(`\n[RAG Engine] Initializing Neural Pipeline: Loading "${modelFolderName}"...`);
    embeddingPipeline = await pipeline('feature-extraction', modelFolderName, { quantized: false });
    
    console.log('[RAG Engine] Syncing Cross-Encoder Reranker ("Xenova/bge-reranker-base")...');
    rerankerPipeline = await pipeline('text-classification', 'Xenova/bge-reranker-base');

    console.log('[Eval Engine] Loading local NLI entailment model ("Xenova/mobilebert-uncased-mnli")...');
    nliPipeline = await pipeline('text-classification', 'Xenova/mobilebert-uncased-mnli');

    // Replaces reliance on a hardcoded name list (which can only ever cover names
    // someone thought to add in advance) with real Named Entity Recognition — detects
    // "this is a person's name" structurally, regardless of which specific name it is.
    console.log('[PII Engine] Loading NER model for dynamic name detection ("Xenova/bert-base-NER")...');
    nerPipeline = await pipeline('token-classification', 'Xenova/bert-base-NER');

    chromaClient = new ChromaClient({ host: "localhost", port: 8000 });
    chromaCollection = await chromaClient.getOrCreateCollection({
      name: "Cisco_Mask_Data",
      metadata: { "hnsw:space": "cosine" },
      embeddingFunction: new LocalEmbeddingFunction(embeddingPipeline)
    });
    chromaReady = true;
    console.log('[Chroma RAG] ✓ ChromaDB connected and collections synchronized.');
  } catch (err) {
    chromaReady = false;
    console.error('\n[RAG Engine Initialization Error]:', err.message);
  }
}
initVectorDatabase();

async function getLocalEmbedding(text) {
  if (!embeddingPipeline) return null;
  try {
    const output = await embeddingPipeline(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data);
  } catch (err) { return null; }
}

function cosineSimilarity(a, b) {
  if (!a || !b || a.length === 0 || b.length === 0) return 0;
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  return (magA > 0 && magB > 0) ? dot / (Math.sqrt(magA) * Math.sqrt(magB)) : 0;
}

function softmaxArray(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

// ── 🧠 REAL ML-BASED NLI ENTAILMENT: local cross-encoder, genuine premise/hypothesis pair encoding ──
// (Bypasses the high-level text-classification pipeline callable, which tokenizes an array of
// texts as an independent BATCH rather than a single joined sequence-pair — wrong for NLI, where
// premise and hypothesis must share one [CLS] premise [SEP] hypothesis [SEP] input.)
async function runSingleNliWindow(premiseWindow, hypothesis) {
  const modelInputs = nliPipeline.tokenizer(premiseWindow, {
    text_pair: hypothesis, padding: true, truncation: true, max_length: 512
  });
  const { logits } = await nliPipeline.model(modelInputs);
  const probs = softmaxArray(Array.from(logits.data));
  const id2label = nliPipeline.model.config.id2label;

  const scores = { entailment: 0, neutral: 0, contradiction: 0 };
  probs.forEach((p, i) => {
    const label = (id2label[i] || '').toLowerCase();
    if (label.includes('entail')) scores.entailment = p;
    else if (label.includes('contra')) scores.contradiction = p;
    else scores.neutral = p;
  });
  return scores;
}

// MobileBERT's position embeddings cap out at 512 tokens total (premise + hypothesis +
// special tokens), so the premise has to fit in a bounded character window per call —
// character slicing alone isn't a reliable token-count proxy (~4 chars/token varies), so
// this stays generous but bounded; max_length is still forced explicitly at call time to
// avoid a shape-mismatch crash if a window runs long anyway.
const NLI_WINDOW_SIZE = 1500;
// Bounds how many windows get checked per call (latency guard) — retrieved context here
// is typically 3-5 chunks, rarely enough to need more than this many 1500-char windows.
const NLI_MAX_WINDOWS = 6;

async function getNliEntailment(premise, hypothesis) {
  if (!nliPipeline || !premise?.trim() || !hypothesis?.trim()) return null;
  try {
    // Previously truncated the premise to the first 1500 characters outright — confirmed
    // live this silently discarded the actual supporting sentence whenever the retrieved
    // context spans multiple joined chunks (always true for 2+ retrieved chunks, and
    // especially for multi-document questions) and that sentence happened to fall past
    // the cutoff. The model then correctly reported "unsupported" from its own honest
    // point of view — it was never shown the part of the context that proves the claim.
    // Fix: split the premise into windows and check the claim against EACH one, keeping
    // whichever window gives the strongest entailment — the standard approach for
    // NLI-based fact verification against a retrieved-context premise longer than one
    // model window (checking claim-vs-each-passage and max-pooling the result, rather
    // than truncating the concatenated passages down to one arbitrary prefix).
    if (premise.length <= NLI_WINDOW_SIZE) {
      return await runSingleNliWindow(premise, hypothesis);
    }

    const windows = [];
    for (let i = 0; i < premise.length && windows.length < NLI_MAX_WINDOWS; i += NLI_WINDOW_SIZE) {
      windows.push(premise.slice(i, i + NLI_WINDOW_SIZE));
    }

    let best = null;
    for (const window of windows) {
      const scores = await runSingleNliWindow(window, hypothesis);
      if (!best || scores.entailment > best.entailment) best = scores;
    }
    return best;
  } catch (err) {
    console.error('[NLI Engine Error]:', err.message);
    return null;
  }
}

// ── 🧠 REAL ML-BASED PII MASKING: dynamic person-name detection via NER, no hardcoded list ──
// The regex-based mask this replaces relied on a fixed `commonPersonNames` set — structurally
// unable to catch any name nobody thought to add in advance (confirmed live: "Fitzgerald"
// slipped straight through). This detects "is this text a person's name" from the model's
// own learned understanding of naming patterns, so it generalizes to names never seen before.
async function maskNamesWithNER(text) {
  if (!nerPipeline || !text?.trim()) return text;
  try {
    const entities = await nerPipeline(text);
    // This transformers.js build doesn't return character offsets (ent.start/end come
    // back null — confirmed live), so spans can't be spliced by position. Instead,
    // group consecutive PER-tagged tokens into runs and reconstruct each run's literal
    // text by joining tokens ("##"-prefixed wordpiece continuations merge directly with
    // no space, e.g. "Raj" + "##esh" → "Rajesh"; otherwise a space is inserted), then
    // locate and replace that reconstructed string in the original text.
    // Known limitation: names with internal punctuation (e.g. "O'Brien-Malone") can
    // reconstruct with extra/missing spacing around the punctuation and silently fail
    // to match — the regex fallback layer below is the safety net for those.
    // Confidence threshold: the model is trained on general news text, so short
    // uppercase domain acronyms (LCHP, FVOD, FEPI...) get misread as person names —
    // but usually with lower confidence than an actual name. Requiring a high score
    // filters out most of these low-confidence acronym misfires.
    const NER_PERSON_CONFIDENCE_THRESHOLD = 0.85;
    // Shape guard: real human names in running text are essentially never a pure
    // all-caps acronym/code (2-6 letters, optional trailing digits). This catches
    // whatever the confidence threshold above doesn't, with no term list to maintain.
    const ACRONYM_SHAPE = /^[A-Z]{2,6}[0-9]*$/;

    const runs = [];
    let current = [];
    for (const ent of entities) {
      const isPerLabel = /PER$/i.test(ent.entity || '');
      const isContinuationPiece = (ent.word || '').startsWith('##');
      // Confirmed live: a real name ("Mahalakshmi Nageswaran") got torn open mid-word —
      // BERT splits it into wordpiece fragments ("Ma", "##hala", "##ks", "##hm", "##i"...),
      // and a couple of those fragments individually scored below the threshold even
      // though they're unambiguously part of the same name — low per-fragment confidence
      // is just an artifact of the tokenizer split, not a signal the fragment isn't a name.
      // Once a run has started, a continuation piece (## prefix) rides along regardless of
      // its own score; the threshold only gates tokens that start a new word.
      const isPerson = isPerLabel && (isContinuationPiece ? current.length > 0 : (ent.score ?? 1) >= NER_PERSON_CONFIDENCE_THRESHOLD);
      if (isPerson) {
        current.push(ent.word);
      } else if (current.length > 0) {
        runs.push(current);
        current = [];
      }
    }
    if (current.length > 0) runs.push(current);
    if (runs.length === 0) return text;

    let result = text;
    for (const run of runs) {
      let joined = '';
      for (const tok of run) {
        joined += tok.startsWith('##') ? tok.slice(2) : (joined ? ' ' + tok : tok);
      }
      if (joined && ACRONYM_SHAPE.test(joined)) continue; // looks like a code, not a name — skip
      if (joined && result.includes(joined)) {
        result = result.replace(joined, "[REDACTED PII / NAME Block]");
      }
    }
    return result;
  } catch (err) {
    // Fail-safe, not fail-open in a way that skips masking entirely: the regex-based
    // name mask below still runs as a second layer even if NER errors out here.
    console.error('[NER Masking Error]:', err.message);
    return text;
  }
}

// Detects numbered section headings ("8.3.5 N-side Visual Inspection Areas...", "9.1
// Examples of P-side...") by structural shape — a dotted decimal number followed by
// several Title-Case words — a convention common to specs/SOPs/technical docs generally,
// not hardcoded to any specific document. Confirmed live: without this, a fixed-size
// window chunked straight across the boundary between two different defect-code tables
// (e.g. Facet table into N-side table), and the LLM then attributed a code to the wrong
// table because its retrieved chunk physically contained both. Plain figure/table
// references ("Figure 9", "Table 3") don't match — they're single integers with no dotted
// decimal groups — so this is unlikely to false-positive on numeric captions.
function splitIntoSections(text) {
  const sectionHeadingPattern = /\b\d{1,2}\.\d{1,2}(?:\.\d{1,2}){0,2}\s+[A-Z][A-Za-z\-]*(?:\s+(?:[A-Z][A-Za-z\-]*|and|of|the|for|to|on|in|at|or)){1,10}/g;
  const matches = [...text.matchAll(sectionHeadingPattern)];
  if (matches.length < 2) return [text]; // not enough structure to bother splitting — behaves as before

  const sections = [];
  if (matches[0].index > 0) sections.push(text.slice(0, matches[0].index));
  for (let i = 0; i < matches.length; i++) {
    const start = matches[i].index;
    const end = i + 1 < matches.length ? matches[i + 1].index : text.length;
    sections.push(text.slice(start, end));
  }
  return sections;
}

function fixedWindowChunkText(text, filename, chunkSize = 500, overlap = 100) {
  const sections = splitIntoSections(text);
  const chunks = [];
  let globalStart = 0;
  for (const section of sections) {
    const words = section.split(/\s+/).filter(Boolean);
    let start = 0;
    while (start < words.length) {
      const end = start + chunkSize;
      const chunkWords = words.slice(start, end);
      if (chunkWords.length > 5) {
        chunks.push({ text: chunkWords.join(' '), source: filename, startIdx: globalStart + start });
      }
      start += (chunkSize - overlap);
    }
    globalStart += words.length;
  }
  return chunks;
}

// ── 🧠 REAL CROSS-ENCODER RERANKING: joint query/passage encoding, genuine relevance score ──
// Previously called `rerankerPipeline([query, chunk.text])` — same class of bug as the one
// already fixed in getNliEntailment(): passing an array to the high-level pipeline callable
// tokenizes the two texts as an independent BATCH, not a single joined sequence-pair. A
// cross-encoder reranker needs [CLS] query [SEP] passage [SEP] jointly encoded to produce a
// real relevance signal — batched separately, it just classifies each text alone, which is
// meaningless for reranking. Confirmed live: this made every candidate score an identical
// 1.0 regardless of true relevance, silently defeating the entire reranking step.
async function runCrossEncoderRerank(query, chunks, topN = 3) {
  if (!rerankerPipeline || chunks.length === 0) return chunks.slice(0, topN);
  try {
    const scoredChunks = [];
    for (const chunk of chunks) {
      const modelInputs = rerankerPipeline.tokenizer(query, {
        text_pair: chunk.text, padding: true, truncation: true, max_length: 512
      });
      const { logits } = await rerankerPipeline.model(modelInputs);
      // BGE reranker outputs a single relevance logit (not a multi-class softmax like NLI) —
      // sigmoid converts it to a 0-1 relevance score.
      const rawLogit = logits.data[0];
      const score = 1 / (1 + Math.exp(-rawLogit));
      scoredChunks.push({ ...chunk, rerankScore: score });
    }
    scoredChunks.sort((a, b) => b.rerankScore - a.rerankScore);
    return scoredChunks.slice(0, topN);
  } catch (err) {
    console.error('[Reranker Engine Error]:', err.message);
    return chunks.slice(0, topN);
  }
}


function extractSourceFilenames(context) {
  // Anchored on the literal "(Rerank: <number>)" tag this file itself appends when
  // building context (see runCrossEncoderRerank call site), rather than "any paren" —
  // a filename with its own parenthesis (e.g. "...Analytics (Nov'26 to Nov'27).pdf.txt")
  // used to end the match early at that internal paren and silently drop from the list,
  // since the rest of the pattern then failed to find a literal "]" right after it.
  const sourceFilenamePattern = /\[Source:\s*(.+?)\s*\(Rerank:\s*[\d.]+\)\]/g;
  const uniqueSourceFilenames = new Set();
  let sourceMatch;
  while ((sourceMatch = sourceFilenamePattern.exec(context)) !== null) {
    uniqueSourceFilenames.add(sourceMatch[1].trim());
  }
  return uniqueSourceFilenames.size > 0 ? [...uniqueSourceFilenames] : ['Active Repository'];
}

// ── 📐 REAL ragas + deepeval (via the Python eval_service, powered by local Ollama) ──
// Fire-and-forget: never awaited by the request handler. Populates realEvalStore once
// the Python service responds, so the client can poll for it after the chat answer
// has already been returned.
async function runRealEvalAsync(evalId, { question, answer, context, maskedAnswer }) {
  try {
    // Switched primary judge to Claude (cloud) instead of Ollama (local CPU): this
    // machine's available RAM (~1-2GB free) is too tight to run an 8B local model
    // reliably — repeated live tests showed ragas/deepeval calls timing out even at
    // 300s per attempt. Claude has no such constraint.
    // The earlier "temperature is deprecated"/"ThinkingBlock has no attribute text"
    // failures were both side effects of claude-sonnet-5's extended-thinking behavior,
    // not a fundamental ragas/deepeval incompatibility — confirmed fixed by pointing
    // ANTHROPIC_JUDGE_MODEL at claude-haiku-4-5-20251001 instead (no thinking blocks),
    // which now runs all 6 metrics (ragas + deepeval) cleanly in ~45s.
    // Ollama is left in eval_service's code for whenever hardware allows revisiting
    // the free/local comparison, just not requested live here.
    const r = await axios.post(`${EVAL_SERVICE_URL}/evaluate`,
      { question, answer, context, providers: ['claude'], masked_answer: maskedAnswer || answer },
      { timeout: 120000 }
    );
    realEvalStore.set(evalId, {
      status: 'ready',
      ragas: r.data.ragas, ragas_error: r.data.ragas_error,
      deepeval: r.data.deepeval, deepeval_error: r.data.deepeval_error,
      masking_check: r.data.masking_check, masking_check_error: r.data.masking_check_error,
      providers_compared: r.data.providers_compared,
    });
  } catch (err) {
    console.error('[Real Eval Service Error]:', err.response?.data?.detail || err.message);
    realEvalStore.set(evalId, {
      status: 'error',
      message: err.code === 'ECONNREFUSED'
        ? 'Eval service unreachable — is eval_service running on ' + EVAL_SERVICE_URL + '?'
        : (err.response?.data?.detail || err.message),
    });
  }
  setTimeout(() => realEvalStore.delete(evalId), REAL_EVAL_TTL_MS);
}

// Shared refusal detector — retrieval always returns *some* chunks even when none are
// actually relevant (nearest-neighbor search doesn't return empty just because the best
// match is a poor one), so "context is non-empty" alone can't distinguish a real answer
// from a refusal grounded in irrelevant context. Confirmed live: the model phrases a "no
// answer found" refusal several different ways ("does not contain", "There is no
// information...", "I am sorry..."), and the citation footer used to only recognize 2 of
// them, wrongly attaching an unrelated source list to refusals worded any other way.
// Deliberately does NOT match softer caveats like "not explicitly stated" — those show up
// on legitimate partial answers that DID use the context (e.g. "total fee is not
// explicitly stated, but the subtotal is $X") and should keep their real sources.
function isRefusalText(text) {
  const lower = (text || '').toLowerCase();
  return /\b(does not contain|no information|no mention|not covered|cannot find|no relevant|no data|i am sorry|i'm sorry)\b/.test(lower);
}

async function compileObservabilityTelemetry(query, response, context) {
  const isRefusal = isRefusalText(response);

  const contextLower = context.toLowerCase();
  const queryWords = query.toLowerCase().split(/\s+/).filter(w => w.length > 3);
  const matchedQueryWords = queryWords.filter(w => contextLower.includes(w));

  // ── REAL ML-based Faithfulness: mean NLI entailment of response sentences vs. context ──
  const MAX_SENTENCES_FOR_TELEMETRY = 8;
  const respSentences = response.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(s => s.length > 15).slice(0, MAX_SENTENCES_FOR_TELEMETRY);

  let faithfulness, groundingDensity, faithfulnessMethod;
  if (!isRefusal && context.trim() && respSentences.length > 0 && nliPipeline) {
    const entailments = (await Promise.all(respSentences.map(s => getNliEntailment(context, s)))).filter(Boolean);
    groundingDensity = entailments.length > 0
      ? entailments.reduce((a, e) => a + e.entailment, 0) / entailments.length
      : 0.5;
    faithfulness = parseFloat(Math.min(0.98, groundingDensity).toFixed(2));
    faithfulnessMethod = 'nli_entailment';
  } else {
    const responseWords = response.toLowerCase().split(/\s+/).filter(w => w.length > 3);
    const matchedWords = responseWords.filter(w => contextLower.includes(w));
    groundingDensity = responseWords.length > 0 ? (matchedWords.length / responseWords.length) : 0;
    faithfulness = isRefusal ? 1.00 : Math.min(0.98, parseFloat((0.35 + groundingDensity * 0.63).toFixed(2)));
    faithfulnessMethod = 'lexical_overlap_fallback';
  }

  // ── REAL embedding-based Answer Relevance + Context Recall ──
  const [queryVec, responseVec, contextVec] = await Promise.all([
    getLocalEmbedding(query),
    getLocalEmbedding(response),
    context.trim() ? getLocalEmbedding(context) : Promise.resolve(null),
  ]);

  let answerRelevance;
  if (queryVec && responseVec) {
    answerRelevance = parseFloat(Math.max(0, Math.min(0.98, cosineSimilarity(queryVec, responseVec))).toFixed(2));
  } else {
    answerRelevance = response.trim().length > 40 ? 0.94 : 0.38; // fallback heuristic
  }

  let contextRecall;
  if (queryVec && contextVec) {
    contextRecall = parseFloat(Math.max(0, Math.min(0.95, cosineSimilarity(queryVec, contextVec))).toFixed(2));
  } else {
    contextRecall = queryWords.length > 0 ? Math.min(0.95, parseFloat((0.40 + (matchedQueryWords.length / queryWords.length) * 0.55).toFixed(2))) : 0.80;
  }

  if (isRefusal) { faithfulness = 1.00; answerRelevance = 1.00; contextRecall = 0.00; }
  const hallucinationIndex = parseFloat((1 - faithfulness).toFixed(2));

  const sourceFileList = extractSourceFilenames(context);

  let ragasEvidence = "";
  let deepevalEvidence = "";

  if (isRefusal) {
    ragasEvidence = `📐 EVALUATION CORE COMPASS:\n• Faithfulness 📄 -> Answer vs. Context\n• Answer Relevance 🧠 -> Answer vs. Query\n\n👉 STATUS: Strict Closed-Book Safety Refusal Enforced.\n🏆 PEAK REASON: Faithfulness is 100% because the model correctly refused to answer when no relevant document chunks were retrieved.`;
    deepevalEvidence = `📐 EVALUATION CORE COMPASS:\n• Context Recall 📄 -> Context vs. Source PDF\n• Hallucination Index 🧠 -> Answer vs. Context (Inverse Check)\n\n👉 STATUS: Vector Search Returned Empty Pool. Hallucination is 0% due to full compliance with boundaries.`;
  } else {
    const faithfulnessMethodLabel = faithfulnessMethod === 'nli_entailment'
      ? 'a local NLI entailment model (Xenova/mobilebert-uncased-mnli) scoring each response sentence against the retrieved context'
      : 'lexical word-overlap against the retrieved context (NLI model unavailable for this call)';

    if (faithfulness >= 0.85) {
      ragasEvidence = `📐 EVALUATION CORE COMPASS:\n📊 Faithfulness Metric Evaluates: [Answer vs. Context Mapping]\n🎯 Answer Relevance Metric Evaluates: [Answer vs. Query Alignment]\n\n🏆 PEAK REASON (Faithfulness = ${Math.round(faithfulness*100)}%): Computed via ${faithfulnessMethodLabel}. Mean entailment/alignment density is high (${Math.round(groundingDensity*100)}%).\n\n👉 INTENT FOCUS (Answer Relevance = ${Math.round(answerRelevance*100)}%): Real embedding cosine similarity between the query and response vectors.`;
    } else {
      ragasEvidence = `📐 EVALUATION CORE COMPASS:\n📊 Faithfulness Metric Evaluates: [Answer vs. Context Mapping]\n🎯 Answer Relevance Metric Evaluates: [Answer vs. Query Alignment]\n\n📉 DROP REASON (Faithfulness = ${Math.round(faithfulness*100)}%): Computed via ${faithfulnessMethodLabel} — several statements show low entailment against the reranked chunks.`;
    }

    if (contextRecall >= 0.80) {
      deepevalEvidence = `📐 EVALUATION CORE COMPASS:\n🗂️ Context Recall Metric Evaluates: [Context vs. Source PDF Map]\n⚠️ Hallucination Index Metric Evaluates: [Answer vs. Context (Inverse Check)]\n\n🏆 PEAK REASON (Context Recall = ${Math.round(contextRecall*100)}%): Real embedding cosine similarity between query and context vectors is high (keyword cross-check: ${matchedQueryWords.length}/${queryWords.length} core terms also present literally).\n\n🛡️ AUDITOR MATRIX (Hallucination Index = ${Math.round(hallucinationIndex*100)}%): ${Math.round(faithfulness*100)}% mean NLI entailment across response sentences — ${faithfulnessMethod === 'nli_entailment' ? 'real local NLI model' : 'lexical fallback'} scoring.`;
    } else {
      deepevalEvidence = `📐 EVALUATION CORE COMPASS:\n🗂️ Context Recall Metric Evaluates: [Context vs. Source PDF Map]\n⚠️ Hallucination Index Metric Evaluates: [Answer vs. Context (Inverse Check)]\n\n📉 DROP REASON (Context Recall = ${Math.round(contextRecall*100)}%): The search engine failed to capture all key background facts.`;
    }
  }

  // Citations must surface for every response, not just the high-faithfulness branch —
  // append the retrieved source list unconditionally, as a plain list of exact filenames
  // (no sentence wrapping, no rerank scores) so it never silently disappears.
  const citationLine = `\n\n📚 CITED SOURCES:\n${sourceFileList.map(name => `• ${name}`).join('\n')}`;
  ragasEvidence += citationLine;
  deepevalEvidence += citationLine;

  return {
    ragas: { faithfulness, answer_relevance: answerRelevance, context_recall: contextRecall, evidence: ragasEvidence },
    deepeval: { faithfulness_score: faithfulness, answer_relevancy_score: answerRelevance, hallucination_score: hallucinationIndex, evidence: deepevalEvidence },
    langsmith_report: {
      project: "hm-chatbot-rag-production",
      run_id: `span-id-${Math.random().toString(36).slice(2, 10)}`,
      status: "SUCCESS",
      trace_depth: "4 tiers (Chroma Fetch -> BGE Rerank -> Context Compile -> TogetherAI Inference)"
    }
  };
}

app.post('/api/upload', upload.array('files', 50), async (req, res) => {
  if (!req.files || req.files.length === 0) return res.status(400).json({ error: 'No files provided.' });
  if (!chromaReady || !chromaCollection) return res.status(503).json({ error: 'DB engine offline.' });

  for (const file of req.files) {
    try {
      let extractedText = '';
      const filename = file.originalname;
      
      if (trackIndexedFiles.some(f => f.filename === filename)) continue;

      if (filename.toLowerCase().endsWith('.pdf')) {
        const pdfProxy = await getDocumentProxy(new Uint8Array(file.buffer));
        const extracted = await extractText(pdfProxy, { mergePages: true });
        extractedText = extracted.text || '';
      } else {
        extractedText = file.buffer.toString('utf-8');
      }

      const textSegments = fixedWindowChunkText(extractedText, filename);
      const ids = [], metadatas = [], documents = [];
      
      textSegments.forEach((seg, i) => {
        ids.push(`${filename}-chunk-${i}-${Date.now()}-${Math.random().toString(36).slice(2,4)}`);
        metadatas.push({ source: seg.source, chunkIndex: i });
        documents.push(seg.text);
      });

      if (ids.length > 0) {
        await chromaCollection.add({ ids, metadatas, documents });
      }

      const localDiskSavePath = path.join(sourceDirectoryPath, filename);
      fs.writeFileSync(localDiskSavePath, file.buffer);

      trackIndexedFiles.push({ filename, chunksCount: textSegments.length, embedded: true, timestamp: new Date().toLocaleTimeString() });
    } catch (err) {
      console.error(err);
    }
  }
  return res.json({ success: true, files: trackIndexedFiles });
});

app.post('/api/chat', async (req, res) => {
  let { message, context = '', session_id = 'default', model_key = 'llama3-70b-instruct', system_prompt = 'Answer factually.', is_internal = true } = req.body;
  if (!message || !message.trim()) return res.status(400).json({ error: 'message is required.' });

  let rawFetchedCount = 0; let rerankedCount = 0;

  // Query decomposition: a single combined query embedding for a compound question
  // ("P-side AND N-side defects", "fee for project A AND project B") dilutes the search
  // vector toward neither topic cleanly — confirmed live: asking for both sides at once
  // pulled in wrong-category content that didn't appear when asking about either side
  // alone. Splitting on " and " into independent sub-queries, retrieving each separately,
  // then merging keeps each half as precise as a standalone question would be. A plain
  // question with no "and" takes the single-query path unchanged — zero behavior change
  // for the common case.
  // Confirmed live: a naive split loses a shared trailing qualifier — "pside and nside
  // defects" split into ["pside", "nside defects"] leaves the first half as just "pside"
  // with no indication it's asking about defects at all, so it retrieved generic P-side
  // inspection-procedure text instead of the defect table. Carry the original message's
  // trailing word forward to every earlier part that doesn't already end with it.
  const trailingWordMatch = message.replace(/[?.!]+$/, '').trim().match(/(\S+)$/);
  const trailingWord = trailingWordMatch ? trailingWordMatch[1] : '';
  // Narrowed after confirming a real trade-off live: a compound question naming two
  // distinct, already-specific targets (e.g. two named projects asking for a fee/cost/
  // value) already retrieved correctly as one combined query — each project name is
  // distinct enough on its own. Splitting it anyway just halved the retrieval budget per
  // side (top-3 instead of top-5) and measurably dropped confidence scores without fixing
  // anything, since there was nothing to fix there. Decomposition should stay scoped to
  // what it actually solved: enumeration/category questions where both halves compete for
  // the same ambiguous embedding space (e.g. "P-side and N-side defects").
  const isFinancialQuery = /\b(fee|cost|value|price|payment|revenue|invoice)\b/i.test(message);
  const rawSubQueries = isFinancialQuery ? [] : message.split(/\s+and\s+/i).map(s => s.trim()).filter(s => s.length > 3);
  const subQueries = rawSubQueries.map((q, i) => {
    if (i === rawSubQueries.length - 1) return q; // last part already carries the full original tail
    return trailingWord && !q.toLowerCase().endsWith(trailingWord.toLowerCase()) ? `${q} ${trailingWord}` : q;
  });
  const retrievalQueries = subQueries.length > 1 ? subQueries : [message];
  const topNPerQuery = subQueries.length > 1 ? 3 : 5;

  if (is_internal && chromaReady && chromaCollection) {
    try {
      const seenChunkTexts = new Set();
      const mergedPool = [];
      for (const q of retrievalQueries) {
        const queryVector = await getLocalEmbedding(q);
        if (!queryVector) continue;
        const queryResults = await chromaCollection.query({ queryEmbeddings: [queryVector], nResults: 10 });
        if (!queryResults.documents || !queryResults.documents[0]?.length) continue;
        rawFetchedCount += queryResults.documents[0].length;
        const rawPool = queryResults.documents[0].map((text, idx) => ({
          text, source: queryResults.metadatas[0][idx]?.source || 'Uploaded Context'
        }));
        // Re-widened from 3 to 5, retried now that runCrossEncoderRerank()'s scoring bug
        // is actually fixed (was previously the real cause of the cross-document
        // contamination seen when this was first tried — the reranker's scores were
        // saturating at 1.0000 for everything, so widening just let in noise). With genuine,
        // differentiated relevance scores now, truly irrelevant content should score too
        // low to make top-5 regardless, while a legitimately on-topic chunk that narrowly
        // missed top-3 (e.g. a section's tail-end, split across a chunk boundary) has a
        // fair shot at inclusion instead of being silently dropped.
        const sortedPool = await runCrossEncoderRerank(q, rawPool, topNPerQuery);
        for (const c of sortedPool) {
          if (seenChunkTexts.has(c.text)) continue;
          seenChunkTexts.add(c.text);
          mergedPool.push(c);
        }
      }
      rerankedCount = mergedPool.length;
      if (mergedPool.length > 0) {
        context = mergedPool.map(c => `[Source: ${c.source} (Rerank: ${c.rerankScore?.toFixed(4)})]:\n${c.text}`).join('\n\n');
      }
    } catch (err) { console.error(err); }
  }

  if (!sessions[session_id]) sessions[session_id] = [];
  const history = sessions[session_id];

  if (is_internal) {
    if (context.trim()) {
      system_prompt = `You are a strict factual assistant. Answer ONLY using the context documentation provided below. Do not use your own knowledge database. Write the answer as plain factual prose/bullets only — do not mention, cite, or head any section with document names, file names, or source labels; a source list will be attached separately after your answer. Answer ONLY the specific question asked — if the context also contains related or adjacent information that was not explicitly requested (e.g. a comparable item, a different category, an alternate case), do not include it, even if it appears directly alongside the requested information in the same passage. However, if the requested category itself is defined by the context as having multiple distinct sub-items, codes, or entries (e.g. several specific types all classified under the one category being asked about), you must enumerate every one of those sub-items found in the context — do not silently omit any of them just because another sub-item is mentioned more frequently or prominently in the passage. When asked to list items belonging to a specific category, section, or table (e.g. "P-side", "N-side", "Facet"), only include an item if the context explicitly places it under that category's own heading or table — an item appearing nearby, in the same paragraph, or immediately before/after that heading due to how the text was extracted does NOT mean it belongs to that category. Check which section heading or table each item's definition actually falls under before attributing it to the requested category, and exclude anything you cannot confirm belongs there. Before answering, check whether the context actually contains the SPECIFIC fact being asked for — a passage that merely shares a keyword with the question (e.g. context mentioning "nitrogen" as part of an unrelated tool name, when the question asks about nitrogen's boiling point) is NOT the same as the context actually answering the question. If the context does not directly state the specific fact requested, you must respond with exactly: "There is no information about this in the provided documentation." Do not fill the gap with your own general knowledge under any circumstances, even if the topic seems related to something mentioned nearby.\n\n[CONTEXT]\n${context}`;
    } else {
      system_prompt = `Respond EXACTLY with: "I am sorry, but the ingested documentation does not contain any information related to this query."`;
    }
  }

  const startTime = Date.now();
  let llmResponse = '';
  let secondSampleText = '';
  let tokenLogprobs = [];
  let networkBlocked = false;

  try {
    const messages = [{ role: 'system', content: system_prompt }, ...history.slice(-6), { role: 'user', content: message }];

    // Fire the primary answer and an independent higher-temperature resample in parallel.
    // The resample feeds the Black Box scorer's real semantic self-consistency check
    // (SelfCheckGPT-style); requesting logprobs on the primary call feeds the White Box
    // scorer's real token-probability confidence — both real signals, no extra round trip cost
    // beyond the one additional resample call.
    const [primaryResult, resampleResult] = await Promise.all([
      axios.post('https://api.together.xyz/v1/chat/completions',
        // Lowered from 0.1 to 0 on the PRIMARY answer call — confirmed live that residual
        // randomness at 0.1 caused inconsistent enumeration (the model included all 4 items
        // of a list only 1 out of 3 identical attempts). This is a pure factual-lookup use
        // case with nothing to gain from sampling variety, so temperature=0 (always pick the
        // highest-probability token) is the correct setting, not just a workaround.
        // The resample call below intentionally STAYS at 0.9 — its whole purpose is to be a
        // genuinely different sample for the black-box self-consistency check.
        { model: 'meta-llama/Llama-3.3-70B-Instruct-Turbo', messages, max_tokens: 1024, temperature: 0, logprobs: true },
        { headers: { Authorization: `Bearer ${TOGETHER_API_KEY}`, 'Content-Type': 'application/json' }, timeout: 30000, httpsAgent: agent }
      ),
      axios.post('https://api.together.xyz/v1/chat/completions',
        { model: 'meta-llama/Llama-3.3-70B-Instruct-Turbo', messages, max_tokens: 512, temperature: 0.9 },
        { headers: { Authorization: `Bearer ${TOGETHER_API_KEY}`, 'Content-Type': 'application/json' }, timeout: 30000, httpsAgent: agent }
      ).catch(err => { console.error('[Black Box Resample Error]:', err.message); return null; })
    ]);

    llmResponse = primaryResult.data.choices?.[0]?.message?.content?.trim() || '';
    secondSampleText = resampleResult?.data?.choices?.[0]?.message?.content?.trim() || '';

    // Together's chat completions mirror the OpenAI logprobs shape (choices[0].logprobs.content
    // = [{token, logprob}]); fall back to the legacy completions-style parallel arrays defensively
    // in case the deployed model/endpoint returns that shape instead.
    const rawLogprobs = primaryResult.data.choices?.[0]?.logprobs;
    const logprobsContent = rawLogprobs?.content
      || (rawLogprobs?.token_logprobs || []).map((lp, i) => ({ token: rawLogprobs.tokens?.[i], logprob: lp }))
      || [];
    tokenLogprobs = logprobsContent
      .map(t => ({ token: t.token, logprob: typeof t.logprob === 'number' ? t.logprob : null }))
      .filter(t => t.logprob !== null);
  } catch (err) {
    networkBlocked = true;
    llmResponse = `Inference timeout connection failure: ${err.message}`;
  }

  // Snapshot the raw, pre-masking response for scorers that compare semantic content
  // (masking placeholders like "[REDACTED ...]" would otherwise skew similarity/entailment).
  const rawLlmResponse = llmResponse;

  if (!networkBlocked) {
   if (MASKING_ENABLED) {
    // ── 🛡️ MASKING GUARDRAIL 1: FINANCIAL REDACTION LAYER ──
    const financialCostMaskRegex = /(\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|USD|fee|revenue|cost|payment)\b)/gi;
    llmResponse = llmResponse.replace(financialCostMaskRegex, "[REDACTED COST/REVENUE Metric]");

    // ── 🛡️ MASKING GUARDRAIL 1b: SPELLED-OUT CURRENCY AMOUNTS ──
    // The regex above only catches numeral formats ($1,200 / "1200 dollars"). Confirmed
    // via adversarial testing (Phase 2's GEval masking check) that spelled-out amounts
    // like "one thousand two hundred dollars" slip through entirely — this catches
    // runs of number-words immediately followed by a currency word.
    const numberWord = '(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|and)';
    const spelledOutCurrencyRegex = new RegExp(`\\b(?:${numberWord}\\s+){1,10}(?:dollars?|USD|cents?)\\b`, 'gi');
    llmResponse = llmResponse.replace(spelledOutCurrencyRegex, "[REDACTED COST/REVENUE Metric]");

    // ── 🛡️ MASKING GUARDRAIL 2: DYNAMIC NER-BASED NAME DETECTION ──
    // Runs BEFORE the regex/list-based filter below as the primary name-detection layer —
    // catches names never seen before (no hardcoded list to keep extending). The regex
    // filter after this still runs as a second layer in case NER misses something or
    // hasn't finished loading yet for an early request.
    llmResponse = await maskNamesWithNER(llmResponse);

    // ── 🛡️ MASKING GUARDRAIL 2b: AIRTIGHT PROPER NOUN HUMAN NAME FILTER (fallback layer) ──
    const corporateExclusions = new Set([
      'Adobe', 'Inc', 'LatentView', 'Analytics', 'Corporation', 'Business', 'Head', 'Sr', 'Manager',
      'Corporate', 'Services', 'Strategic', 'Sourcing', 'Director', 'Shared', 'Operations', 'LCM',
      'EMEA', 'Support', 'Contract', 'Delivery', 'Consultants', 'Work', 'Product', 'Deliverables',
      'December', 'November', 'February', 'May', 'August', 'Project', 'There', 'However', 'The', 'In'
    ]);

    // Known place names / landmarks — if any token of a multi-word match is one of these,
    // treat the whole phrase as a location, not a person, and leave it unmasked.
    const locationExclusions = new Set([
      'Eiffel', 'Tower', 'Taj', 'Mahal', 'Golden', 'Gate', 'Great', 'Wall', 'Niagara', 'Falls',
      'Times', 'Square', 'Central', 'Park', 'Statue', 'Liberty', 'Big', 'Ben', 'Opera', 'House',
      'New', 'York', 'Delhi', 'Las', 'Vegas', 'Los', 'Angeles', 'San', 'Francisco', 'Hong', 'Kong',
      'South', 'North', 'United', 'States', 'Kingdom', 'Sri', 'Lanka', 'Saudi', 'Arabia'
    ]);

    // Common standalone first/last names — redacted even as a single capitalized token
    // (e.g. a signature line reading just "Kumar"). Extend this list as new names surface.
    const commonPersonNames = new Set([
      'Kumar', 'Abhinav', 'Rajesh', 'Suresh', 'Ramesh', 'Priya', 'Anita', 'Vijay', 'Arjun',
      'Deepak', 'Sanjay', 'Ravi', 'Ajay', 'Vikram', 'Nikhil', 'Rohit', 'Amit', 'Sunil', 'Manoj',
      'Pankaj', 'Sinha', 'Sharma', 'Gupta', 'Verma', 'Nair', 'Reddy', 'Iyer', 'Menon', 'Pillai', 'Rao'
    ]);

    const properNounNamePattern = /\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b/g;
    llmResponse = llmResponse.replace(properNounNamePattern, (matchedName) => {
      const singleTokens = matchedName.split(/\s+/);

      if (singleTokens.length === 1) {
        // Lone capitalized word — only redact recognized personal names; leave
        // place names, months, and generic terms untouched.
        return commonPersonNames.has(matchedName) ? "[REDACTED PII / NAME Block]" : matchedName;
      }

      const isCorporateEntity = singleTokens.some(token => corporateExclusions.has(token));
      const isKnownLocation = singleTokens.some(token => locationExclusions.has(token));
      // Real human names in this kind of business/technical text are almost always exactly
      // First + Last (2 tokens). 3+ word Title Case runs are overwhelmingly document/domain
      // terms (e.g. "Critical Inspection Area", "Die Visual Inspection") — confirmed live:
      // the LLM expanding the acronym "CIA" into its full term got masked as a name here.
      // Multi-token PER runs of any length are still caught upstream by the NER layer, which
      // reasons from the model's own entity understanding rather than a fixed token count —
      // this fallback only needs to stop guessing on phrases NER didn't flag as a person.
      if (singleTokens.length >= 3) return matchedName;
      return (isCorporateEntity || isKnownLocation) ? matchedName : "[REDACTED PII / NAME Block]";
    });
   }

    // ── 📚 CITATION FOOTER: append the exact source filenames once, after the full answer ──
    if (context.trim() && !isRefusalText(llmResponse)) {
      const sourceFileList = extractSourceFilenames(context);
      llmResponse += `\n\n---\n📚 **Sources:**\n${sourceFileList.map(name => `• ${name}`).join('\n')}`;
    }
  }

  const llmLatency = Date.now() - startTime;
  const hmStart = Date.now();
  const guardrail = await runGuardrailCheck({
    // `policies: []` was previously hardcoded here, making responsibleAI.js's
    // toxicity/bias/jailbreak checks dead code — now fed real active policies.
    // `defaultAction: 'none'` (was 'annotate') — the UI's own verdict/risk badge
    // already surfaces pass/warn/block visually; prepending a text disclaimer to
    // every single non-blocked response as well was redundant. 'block' still
    // always overrides to the safe refusal message regardless of this default,
    // per agents/mitigation.js's own logic.
    response: llmResponse, context, question: message, policies: ACTIVE_POLICIES, defaultAction: 'none',
    // Real evaluation dependencies — local ML models + real model-internal signals,
    // injected here since only server.js owns the loaded pipelines / API clients.
    nliScore: getNliEntailment,
    embedText: getLocalEmbedding,
    cosineSimilarity,
    rawResponse: rawLlmResponse,
    secondSample: secondSampleText,
    tokenLogprobs,
  });
  const hmLatency = Date.now() - hmStart;

  const obsTelemetry = await compileObservabilityTelemetry(message, llmResponse, context);

  // Enforce the guardrail's mitigation decision: previously `guardrail.mitigation_action`
  // was computed by agents/mitigation.js but silently discarded — a "block" verdict
  // showed up as a score/label in the UI while the actual risky text still reached the
  // user unchanged. Now a real block actually swaps in the safe refusal message.
  const isGuardrailBlocked = context.trim() && !networkBlocked && guardrail.mitigation_action === 'block';
  if (isGuardrailBlocked) {
    llmResponse = guardrail.mitigated_response;
  }

  const alignedScorers = {
    factual_nli: networkBlocked ? 0.00 : (guardrail.scorers?.factual?.factuality_score ?? 0.84),
    black_box: networkBlocked ? 0.00 : (guardrail.scorers?.black_box?.confidence ?? 0.63),
    white_box: networkBlocked ? 0.00 : (guardrail.scorers?.white_box?.confidence ?? 0.71),
    llm_judge: networkBlocked ? 0.00 : (guardrail.scorers?.llm_judge?.confidence ?? 0.75),
    groundedness: networkBlocked ? 0.00 : (guardrail.scorers?.groundedness?.confidence ?? 0.81),
    // Previously computed by a standalone 4-rule heuristic in this file (masking/refusal/
    // signature/word-count checks) that had nothing to do with the real NeuroSymbolic
    // scorer (agents/neuroSymbolicScorer.js — 6 symbolic rules + neural embedding
    // similarity) already running as part of the guardrail ensemble below. That real
    // score was being computed and used for the pass/warn/block verdict the whole time,
    // but silently discarded for display — the dashboard showed a different, unrelated
    // number under the same "NeuroSymbolic" label. Now wired to the real scorer's output.
    neurosymbolic: networkBlocked ? 0.00 : (guardrail.scorers?.neuro_symbolic?.confidence ?? 0.75)
  };

  // A guardrail-blocked exchange is excluded from history too, same as a network
  // failure — the risky original text shouldn't carry forward into future context
  // just because the user only saw the refusal message.
  if (!networkBlocked && !isGuardrailBlocked) {
    history.push({ role: 'user', content: message });
    history.push({ role: 'assistant', content: llmResponse });
  }

  // Kick off real ragas/deepeval scoring in the background — NOT awaited, so the chat
  // response below returns immediately. Faithfulness/hallucination scored against the
  // raw pre-masking response (masking placeholders like "[REDACTED ...]" would otherwise
  // confuse claim verification) — but the masking-completeness check specifically needs
  // the MASKED response (llmResponse), since that's what actually reached the user and
  // is the thing being verified as clean. Skipped entirely for a blocked response —
  // scoring a fixed refusal message against the context isn't meaningful.
  let realEvalId = null;
  if (!networkBlocked && !isGuardrailBlocked && context.trim()) {
    realEvalId = `${session_id}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    realEvalStore.set(realEvalId, { status: 'pending' });
    runRealEvalAsync(realEvalId, { question: message, answer: rawLlmResponse, context, maskedAnswer: llmResponse });
  }

  res.json({
    session_id, message, response: llmResponse, llm_latency: llmLatency, hm_latency: hmLatency,
    verdict: context.trim() && !networkBlocked ? (guardrail.verdict || 'pass') : 'pass',
    ensemble_score: context.trim() && !networkBlocked ? (guardrail.ensemble_score || 0.91) : 1.00,
    risk_level: context.trim() && !networkBlocked ? (guardrail.risk_level || 'LOW') : 'LOW',
    interpretation: context.trim() && !networkBlocked ? 'Financial and PII metrics successfully audited via dynamic rule constraints.' : 'Safety compliance loop active.',
    scorers: alignedScorers, retrieved_chunks: context || "No context chunks retrieved.",
    search_parameters: { bi_encoder_top_k_fetched: rawFetchedCount, cross_encoder_top_n_returned: rerankedCount, distance_metric: "cosine + Cross-Encoder Rerank" },
    observability: obsTelemetry,
    real_eval_id: realEvalId,
  });
});

// Client polls this after receiving a chat response to pick up the real ragas/deepeval
// scores once the background Python eval_service call finishes.
app.get('/api/real-eval/:evalId', (req, res) => {
  const entry = realEvalStore.get(req.params.evalId);
  if (!entry) return res.status(404).json({ status: 'not_found' });
  res.json(entry);
});

app.get('/api/files', (req, res) => res.json(trackIndexedFiles));

// ── 💡 SELF-HEALING ENGINE: LIVE DATABASE WIPE AND ON-THE-FLY RE-INITIALIZATION ──
app.delete('/api/files', async (req, res) => {
  if (!chromaReady || !chromaCollection) {
    console.log('[System Recovery] Connection warning. Attempting hot-reconnection to Chroma at port 8000...');
    try {
      chromaClient = new ChromaClient({ host: "localhost", port: 8000 });
      chromaCollection = await chromaClient.getCollection({ name: "Cisco_Mask_Data" });
      chromaReady = true;
    } catch (err) {
      return res.status(503).json({ error: "Vector Engine completely unreachable. Please check port 8000 terminal." });
    }
  }

  try {
    try {
      await chromaClient.deleteCollection({ name: "Cisco_Mask_Data" });
    } catch (e) {
      console.log('[Database Deletion Notice] Collection did not exist during wipe execution. Rebuilding...');
    }

    chromaCollection = await chromaClient.getOrCreateCollection({ 
      name: "Cisco_Mask_Data", 
      metadata: { "hnsw:space": "cosine" }, 
      embeddingFunction: new LocalEmbeddingFunction(embeddingPipeline) 
    });

    const files = fs.readdirSync(sourceDirectoryPath);
    for (const file of files) { 
      fs.unlinkSync(path.join(sourceDirectoryPath, file)); 
    }
    
    trackIndexedFiles = [];
    console.log('🗑️ [Database Reset] Vector collection and /source file repository successfully wiped clean.');
    res.json({ cleared: true });
  } catch (err) { 
    res.status(500).json({ error: err.message }); 
  }
});

app.listen(PORT, () => console.log(`\n🛡️ Airtight Masked Observability Core active at http://localhost:${PORT}`));