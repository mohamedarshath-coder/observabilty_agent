require('dotenv').config();
const express  = require('express');
const axios    = require('axios');
const cors     = require('cors');
const path     = require('path');
const https    = require('https');
const multer   = require('multer');
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

// Set the base parent directory to the folder containing your model directory
env.localModelPath = path.dirname(modelFullPath); 
env.allowRemoteModels = false; 
env.localFilesOnly = true;     

const agent = new https.Agent({ 
  keepAlive: true,
  maxSockets: 100
});
const app  = express();
const PORT = process.env.PORT || 3001;
const TOGETHER_API_KEY = process.env.TOGETHER_API_KEY;
const GEMINI_API_KEY   = process.env.GEMINI_API_KEY;

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'client')));

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });

const trackIndexedFiles = [];
const sessions = {};

let chromaClient = null;
let chromaCollection = null;
let embeddingPipeline = null;
let chromaReady = false;

// Custom embedding function wrapper for Chroma
class LocalEmbeddingFunction {
  constructor(pipeline) {
    this.pipeline = pipeline;
  }

  async call(input) {
    if (typeof input === 'string') {
      input = [input];
    }
    
    const embeddings = [];
    for (const text of input) {
      const output = await this.pipeline(text, { pooling: 'mean', normalize: true });
      embeddings.push(Array.from(output.data));
    }
    return embeddings;
  }
}

// Initialize Database & Model Pipeline with proper error handling
async function initVectorDatabase() {
  try {
    const modelFolderName = path.basename(modelFullPath);
    console.log(`\n[RAG Engine] Initializing Neural Pipeline: Loading "${modelFolderName}"...`);
    embeddingPipeline = await pipeline('feature-extraction', modelFolderName, { quantized: false });
    console.log('[RAG Engine] Local Transformer Engine Loaded and Active.');

    console.log('[Chroma RAG] Attempting connection to Chroma DB at 127.0.0.1:8000...');
    chromaClient = new ChromaClient({ 
      host: "127.0.0.1", 
      port: 8000
    });

    const embeddingFunction = new LocalEmbeddingFunction(embeddingPipeline);

    chromaCollection = await chromaClient.getOrCreateCollection({
      name: "Cisco_Mask_Data",
      metadata: { "hnsw:space": "cosine" },
      embeddingFunction: embeddingFunction
    });

    chromaReady = true;
    console.log('[Chroma RAG] ✓ ChromaDB connected and collection ready.');
    console.log('[RAG Engine] Ingestion Engine fully armed and ready to process files.\n');
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
  } catch (err) {
    console.error('[Local Inference Error]:', err.message);
    return null;
  }
}

function fixedWindowChunkText(text, filename, chunkSize = 500, overlap = 100) {
  const words = text.split(/\s+/);
  const chunks = [];
  let start = 0;
  
  while (start < words.length) {
    const end = start + chunkSize;
    const chunkWords = words.slice(start, end);
    if (chunkWords.length > 5) {
      chunks.push({
        text: chunkWords.join(' '),
        source: filename,
        startIdx: start
      });
    }
    start += (chunkSize - overlap);
  }
  return chunks;
}

const TOGETHER_MODELS = {
  'llama3-8b':            'meta-llama/Llama-3-8b-chat-hf',
  'llama3-70b':           'meta-llama/Llama-3-70b-chat-hf',
  'llama3-70b-instruct':  'meta-llama/Llama-3.3-70B-Instruct-Turbo',
  'mistral-7b':           'mistralai/Mistral-7B-Instruct-v0.3',
  'mixtral-8x7':          'mistralai/Mixtral-8x7B-Instruct-v0.1',
  'qwen2-72b':            'Qwen/Qwen2-72B-Instruct',
  'qwen2.5-7b':          'Qwen/Qwen2.5-7B-Instruct-Turbo',
};

const GEMINI_MODELS = {
  'gemini-2.0-flash': 'gemini-2.0-flash',
  'gemini-1.5-flash': 'gemini-1.5-flash',
  'gemini-1.5-pro':   'gemini-1.5-pro',
};

const ALL_MODELS = { ...TOGETHER_MODELS, ...GEMINI_MODELS };

function isGeminiModel(model_key) {
  return model_key in GEMINI_MODELS;
}

async function callGemini(model_key, system_prompt, history, message) {
  if (!GEMINI_API_KEY) throw new Error('GEMINI_API_KEY not set in .env');
  const modelId = GEMINI_MODELS[model_key];
  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${modelId}:generateContent?key=${GEMINI_API_KEY}`;

  const contents = [];
  history.slice(-10).forEach(m => {
    contents.push({ role: m.role === 'assistant' ? 'model' : 'user', parts: [{ text: m.content }] });
  });
  contents.push({ role: 'user', parts: [{ text: message }] });

  const body = {
    systemInstruction: { parts: [{ text: system_prompt }] },
    contents,
    generationConfig: { maxOutputTokens: 1024, temperature: 0.2, topP: 0.9 },
  };

  const r = await axios.post(endpoint, body, { headers: { 'Content-Type': 'application/json' }, timeout: 30000, httpsAgent: agent });
  return r.data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || '';
}

// ── RESTORED: POST /api/upload - Document Ingestion Workflow ──
app.post('/api/upload', upload.single('file'), async (req, res) => {
  if (!req.file || !chromaReady || !chromaCollection) {
    return res.status(400).json({ error: 'File payload missing or Vector DB not initialized.' });
  }

  try {
    let extractedText = '';
    const filename = req.file.originalname;

    if (filename.toLowerCase().endsWith('.pdf')) {
      const pdfProxy = await getDocumentProxy(new Uint8Array(req.file.buffer));
      const extracted = await extractText(pdfProxy, { mergePages: true });
      extractedText = extracted.text || '';
    } else {
      extractedText = req.file.buffer.toString('utf-8');
    }

    if (!extractedText.trim()) {
      return res.status(400).json({ error: 'Extracted text space is empty.' });
    }

    const textSegments = fixedWindowChunkText(extractedText, filename);
    console.log(`[Chroma RAG] Segmented "${filename}" into ${textSegments.length} text blocks.`);

    const ids = [];
    const metadatas = [];
    const documents = [];

    textSegments.forEach((seg, i) => {
      ids.push(`${filename}-chunk-${i}-${Date.now()}`);
      metadatas.push({ source: seg.source, chunkIndex: i });
      documents.push(seg.text);
    });

    if (ids.length > 0) {
      // Automatically uses the registered LocalEmbeddingFunction instance
      await chromaCollection.add({ ids, metadatas, documents });
      console.log(`[Chroma RAG] Successfully stored vector indices for "${filename}".`);
    }

    const fileMeta = { filename, chunksCount: textSegments.length, embedded: true };
    trackIndexedFiles.push(fileMeta);
    return res.json(fileMeta);
  } catch (err) {
    console.error('[Ingestion Failure]:', err.message);
    return res.status(500).json({ error: err.message });
  }
});

// ── FIXED: MERGED UNIFIED POST /api/chat - RAG Semantic Router ──
// POST /api/chat - RAG Semantic Search Retrieval and Hybrid Toggle Router
// POST /api/chat - RAG Semantic Search Retrieval and Hybrid Toggle Router
app.post('/api/chat', async (req, res) => {
  // 💡 DIAGNOSTIC LOG: Let's look exactly at what the frontend network request contains
  console.log('[DEBUG ENGINE] Incoming request payload body:', req.body);

  let {
    message,
    context       = '',
    session_id    = 'default',
    model_key     = 'gemini-2.0-flash',
    system_prompt = 'You are a helpful, knowledgeable assistant. Answer clearly and factually.',
    is_internal   
  } = req.body;
  if (!message || !message.trim()) {
    return res.status(400).json({ error: 'message is required.' });
  }

  // 💡 BULLETPROOF NORMALIZATION: Ensure string "true"/"false" or boolean states resolve correctly
  // This safeguards against payload mapping or naming mismatches from the UI
  let shouldRunRAG = false;
  if (is_internal === true || is_internal === 'true') {
    shouldRunRAG = true;
  }
  
  // If the frontend used camelCase by accident, catch it here
  if (req.body.isInternal === true || req.body.isInternal === 'true') {
    shouldRunRAG = true;
  }

  // ── STEP 1: CONDITIONAL LOCAL PDF COLLECTION LOOKUP ──────────────────
  // ── STEP 1: CONDITIONAL LOCAL PDF COLLECTION LOOKUP ──────────────────
  // 💡 REMOVED trackIndexedFiles.length > 0 check so it can read past server restarts!
  if (shouldRunRAG && chromaReady && chromaCollection) {
    console.log(`[Router Mode] INTERNAL Active: Running semantic search inside "Cisco_Mask_Data"...`);
    try {
      const queryVector = await getLocalEmbedding(message);
      
      if (queryVector) {
        const queryResults = await chromaCollection.query({
          queryEmbeddings: [queryVector],
          nResults: 3
        });

        if (queryResults.documents && queryResults.documents[0]?.length > 0) {
          const matchingChunks = queryResults.documents[0];
          const matchingMeta = queryResults.metadatas[0];
          
          const documentContext = matchingChunks.map((text, i) => 
            `[Source Document Reference: ${matchingMeta[i]?.source || 'Uploaded Context'}]:\n${text}`
          ).join('\n\n');

          context = context ? `${context}\n\n${documentContext}` : documentContext;
          console.log(`[Chroma RAG] Retrieved ${matchingChunks.length} matching text chunks.`);
        } else {
          console.log('[Chroma RAG] Similarity search executed, but collection contains no entries matching this query.');
        }
      }
    } catch (err) {
      console.error('[Chroma Query Routing Error]:', err.message);
    }
  } else {
    console.log(`[Router Mode] EXTERNAL Active: Bypassing local vectors. Using vanilla model parameters.`);
    console.log(`   -> Reason: shouldRunRAG=${shouldRunRAG}, chromaReady=${chromaReady}`);
  }
  if (!sessions[session_id]) sessions[session_id] = [];
  const history = sessions[session_id];

  if (context && is_internal) {
    system_prompt = `${system_prompt}\n\n[CRITICAL GROUNDING CONTEXT DOCUMENTATION - ANSWER ONLY FROM THIS]\n${context}`;
  }

  const startTime = Date.now();
  let llmResponse = '';
  let llmError    = null;
  let modelId     = ALL_MODELS[model_key] || GEMINI_MODELS['gemini-2.0-flash'];

  if (isGeminiModel(model_key)) {
    try {
      llmResponse = await callGemini(model_key, system_prompt, history, message);
    } catch (err) {
      llmError = err.message;
      console.error('[Gemini Route Error Handled]:', llmError);
    }
  } else {
    modelId = TOGETHER_MODELS[model_key] || TOGETHER_MODELS['llama3-8b'];
    const messages = [
      { role: 'system', content: system_prompt },
      ...history.slice(-10),
      { role: 'user', content: message },
    ];
    const ENDPOINTS = [
      'https://api.together.ai/v1/chat/completions',
      'https://api.together.xyz/v1/chat/completions',
    ];
    for (const endpoint of ENDPOINTS) {
      try {
        const r = await axios.post(endpoint,
          { model: modelId, messages, max_tokens: 1024, temperature: 0.2, top_p: 0.9 },
          { headers: { Authorization: `Bearer ${TOGETHER_API_KEY}`, 'Content-Type': 'application/json' }, timeout: 30000, httpsAgent: agent }
        );
        llmResponse = r.data.choices?.[0]?.message?.content?.trim() || '';
        llmError    = null;
        break;
      } catch (err) {
        llmError    = err.response?.data?.error?.message || err.message;
      }
    }
  }

  if (!llmResponse) {
    llmResponse = "System runtime connection timeout. Verify API keys routing configurations.";
    llmError    = 'LLM API execution halt.';
  }

  const llmLatency = Date.now() - startTime;

  const hmStart = Date.now();
  const guardrail = await runGuardrailCheck({
    response:      llmResponse,
    context:       context || history.map(m => m.content).join('\n'),
    question:      message,
    policies:      [],
    defaultAction: 'annotate',
  });
  const hmLatency = Date.now() - hmStart;

  history.push({ role: 'user',      content: message     });
  history.push({ role: 'assistant', content: llmResponse });
  if (history.length > 20) history.splice(0, history.length - 20);

  // Place this right inside the res.json block at the end of app.post('/api/chat')
  res.json({
    session_id,
    model: modelId,
    model_key,
    message,
    response:           llmResponse,
    llm_error:          llmError,
    llm_latency:        llmLatency,
    hm_latency:         hmLatency,
    verdict:            guardrail.verdict,
    ensemble_score:     guardrail.ensemble_score,
    risk_level:         guardrail.risk_level,
    pii_detected:       guardrail.pii_detected,
    top_issues:         guardrail.top_issues,
    policy_flags:       guardrail.policy_flags,
    mitigation_action:  guardrail.mitigation_action,
    mitigated_response: guardrail.mitigated_response,
    interpretation:     guardrail.interpretation,
    scorers:            guardrail.scorers,
    
    // 💡 NEW DATA FIELDS: Expose the chunks and search parameters to the UI
    retrieved_chunks:   context ? context : "No chunks retrieved (External mode or no matches).",
    search_parameters:  {
      collection_name: "Cisco_Mask_Data",
      is_internal_toggle: is_internal,
      n_results_requested: 3,
      distance_metric: "cosine (HNSW)",
      input_query: message
    }
  });
});

app.get('/api/files', (req, res) => res.json(trackIndexedFiles));

app.delete('/api/files', async (req, res) => {
  try {
    if (!chromaReady || !chromaClient) {
      return res.status(503).json({ error: 'ChromaDB not connected.' });
    }
    await chromaClient.deleteCollection({ name: "Cisco_Mask_Data" });
    const embeddingFunction = new LocalEmbeddingFunction(embeddingPipeline);
    chromaCollection = await chromaClient.getOrCreateCollection({ 
      name: "Cisco_Mask_Data", 
      metadata: { "hnsw:space": "cosine" },
      embeddingFunction: embeddingFunction
    });
    trackIndexedFiles.length = 0;
    res.json({ cleared: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/models', (req, res) => res.json(ALL_MODELS));

app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    chromaConnected: chromaReady,
    embeddingModelLoaded: embeddingPipeline !== null
  });
});

app.get('/api/session/:id', (req, res) => res.json({ history: sessions[req.params.id] || [] }));
app.delete('/api/session/:id', (req, res) => { delete sessions[req.params.id]; res.json({ cleared: true }); });
app.get('*', (req, res) => { res.sendFile(path.join(__dirname, 'client', 'index.html')); });

app.listen(PORT, () => {
  console.log('\n========================================================================');
  console.log(` 🛡️  HM Chatbot active at http://localhost:${PORT}`);
  console.log('========================================================================\n');
});