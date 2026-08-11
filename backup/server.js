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

// ── LOCAL OFFLINE MODEL PATH CONFIGURATIONS ─────────────────────────────
// Dynamically reads the folder directory mapping out of your active .env file
// ── LOCAL OFFLINE MODEL PATH CONFIGURATIONS ─────────────────────────────
// Reads the exact absolute folder path straight out of your .env file
// ── 1. UPDATE YOUR LOCAL OFFLINE MODEL PATH CONFIGURATIONS ────────────────
// Reads the path straight out of your .env file
const modelFullPath = process.env.EMBEDDING_MODEL_PATH;

if (!modelFullPath) {
  console.error('[Configuration Error] EMBEDDING_MODEL_PATH is missing from your .env file!');
}

// Set the base parent directory to the folder containing your model directory
env.localModelPath = path.dirname(modelFullPath); 
env.allowRemoteModels = false; // Block downstream remote lookup network attempts
env.localFilesOnly = true;     // Restrict execution strictly to local disk assets

const agent = new https.Agent({ rejectUnauthorized: false });

const app  = express();
const PORT = process.env.PORT || 3001;
const TOGETHER_API_KEY = process.env.TOGETHER_API_KEY;
const GEMINI_API_KEY   = process.env.GEMINI_API_KEY;

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'client')));

// Ingest files into multipart temporary buffer streams
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });

// System tracker tracking variables
const trackIndexedFiles = [];
const sessions = {};

// Initialize ChromaDB External Core Client Connection
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

    // Initialize ChromaDB Client
    console.log('[Chroma RAG] Attempting connection to Chroma DB at 127.0.0.1:8000...');
    chromaClient = new ChromaClient({ 
      host: "127.0.0.1", 
      port: 8000
    });

    // Create custom embedding function
    const embeddingFunction = new LocalEmbeddingFunction(embeddingPipeline);

    // Test connection and get/create collection with custom embedding function
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
    console.error('[Chroma RAG] Could not connect to Chroma DB. Make sure Chroma server is running:');
    console.error('   Run: chroma run --host 127.0.0.1 --port 8000');
  }
}
initVectorDatabase();

// Local Model Feature Extraction Phase
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

// Fixed Window Text chunk segmentation strategy layout
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

// Together AI models preserved exactly
const TOGETHER_MODELS = {
  'llama3-8b':            'meta-llama/Llama-3-8b-chat-hf',
  'llama3-70b':           'meta-llama/Llama-3-70b-chat-hf',
  'llama3-70b-instruct':  'meta-llama/Llama-3.3-70B-Instruct-Turbo',
  'mistral-7b':           'mistralai/Mistral-7B-Instruct-v0.3',
  'mixtral-8x7':          'mistralai/Mixtral-8x7B-Instruct-v0.1',
  'qwen2-72b':            'Qwen/Qwen2-72B-Instruct',
  'qwen2.5-7b':          'Qwen/Qwen2.5-7B-Instruct-Turbo',
};

// Gemini models preserved exactly
const GEMINI_MODELS = {
  'gemini-2.0-flash': 'gemini-2.0-flash',
  'gemini-1.5-flash': 'gemini-1.5-flash',
  'gemini-1.5-pro':   'gemini-1.5-pro',
};

const ALL_MODELS = { ...TOGETHER_MODELS, ...GEMINI_MODELS };

function isGeminiModel(model_key) {
  return model_key in GEMINI_MODELS;
}

// Gemini Core downstream API routing handler
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

// POST /api/upload - Document Ingestion Workflow using ChromaDB
app.post('/api/upload', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded.' });
  }
  
  if (!chromaReady || !chromaCollection) {
    return res.status(503).json({ error: 'ChromaDB not connected. Start Chroma server first: chroma run --host 127.0.0.1 --port 8000' });
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
      return res.status(400).json({ error: 'No text extracted from file.' });
    }

    const textSegments = fixedWindowChunkText(extractedText, filename);
    console.log(`[Chroma RAG] Chunking "${filename}" into ${textSegments.length} segments...`);

    const ids = [];
    const embeddings = [];
    const metadatas = [];
    const documents = [];

    for (let i = 0; i < textSegments.length; i++) {
      const seg = textSegments[i];
      const vector = await getLocalEmbedding(seg.text);
      if (vector) {
        ids.push(`${filename}-chunk-${i}-${Date.now()}`);
        embeddings.push(vector);
        metadatas.push({ source: filename, chunkIndex: i });
        documents.push(seg.text);
      }
    }

    if (ids.length > 0) {
      // Add documents to ChromaDB using proper ChromaClient API
      await chromaCollection.add({
        ids,
        embeddings,
        documents,
        metadatas
      });
      console.log(`[Chroma RAG] ✓ Saved ${ids.length} embeddings to Cisco_Mask_Data collection.`);
    }

    trackIndexedFiles.push({ filename, chunksCount: textSegments.length, embedded: ids.length > 0 });
    return res.json({ success: true, filename, totalChunks: textSegments.length, embeddingsAdded: ids.length });
    
  } catch (err) {
    console.error('[Chroma Ingestion Error]:', err.message);
    return res.status(500).json({ error: 'Upload failed: ' + err.message });
  }
});

// GET /api/files - Get working visual file representation tree mappings
app.get('/api/files', (req, res) => res.json(trackIndexedFiles));

// DELETE /api/files - Empty current working context vectors cleanly
app.delete('/api/files', async (req, res) => {
  try {
    if (!chromaReady || !chromaClient) {
      return res.status(503).json({ error: 'ChromaDB not connected.' });
    }

    // Delete and recreate collection
    await chromaClient.deleteCollection({ name: "Cisco_Mask_Data" });
    
    const embeddingFunction = new LocalEmbeddingFunction(embeddingPipeline);
    chromaCollection = await chromaClient.getOrCreateCollection({ 
      name: "Cisco_Mask_Data", 
      metadata: { "hnsw:space": "cosine" },
      embeddingFunction: embeddingFunction
    });
    
    trackIndexedFiles.length = 0;
    console.log('[Chroma RAG] Collection cleared and reset.');
    res.json({ cleared: true });
  } catch (err) {
    console.error('[Chroma Clear Error]:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// POST /api/chat - RAG Semantic Search Retrieval Router Route
app.post('/api/chat', async (req, res) => {
  let {
    message,
    context       = '',
    session_id    = 'default',
    model_key     = 'gemini-2.0-flash',
    system_prompt = 'You are a helpful, knowledgeable assistant. Answer clearly and factually.',
  } = req.body;

  if (!message || !message.trim()) {
    return res.status(400).json({ error: 'message is required.' });
  }

  // Semantic query lookup inside Cisco_Mask_Data collection
  if (chromaReady && chromaCollection && trackIndexedFiles.length > 0) {
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
            `[Source: ${matchingMeta[i].source}]:\n${text}`
          ).join('\n\n');

          context = context ? `${context}\n\n${documentContext}` : documentContext;
          console.log(`[Chroma RAG] Retrieved ${matchingChunks.length} relevant document chunks.`);
        }
      }
    } catch (err) {
      console.error('[Chroma Query Error]:', err.message);
    }
  }

  if (!sessions[session_id]) sessions[session_id] = [];
  const history = sessions[session_id];

  if (context) {
    system_prompt = `${system_prompt}\n\n[CRITICAL GROUNDING CONTEXT DOCUMENTATION]\n${context}`;
  }

  const startTime = Date.now();
  let llmResponse = '';
  let llmError    = null;
  let modelId     = ALL_MODELS[model_key] || GEMINI_MODELS['gemini-2.0-flash'];

  if (isGeminiModel(model_key)) {
    try {
      llmResponse = await callGemini(model_key, system_prompt, history, message);
      console.log('[Gemini Model Router Execution] OK, model:', modelId);
    } catch (err) {
      llmError = err.message;
      console.error('[Gemini Route Exception Error Handler]:', llmError);
    }
  }

  if (!isGeminiModel(model_key) && !llmResponse) {
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

  // Run Hallucination Manager Telemetry Score Checks
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

  // Return full dynamic metrics tracking data back to front end layout
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
  });
});

app.get('/api/models', (req, res) => res.json(ALL_MODELS));

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    chromaConnected: chromaReady,
    embeddingModelLoaded: embeddingPipeline !== null,
    chromaMessage: chromaReady ? 'Connected to Cisco_Mask_Data collection' : 'ChromaDB connection failed. Run: chroma run --host 127.0.0.1 --port 8000'
  });
});

app.get('/api/session/:id', (req, res) => res.json({ history: sessions[req.params.id] || [] }));
app.delete('/api/session/:id', (req, res) => { delete sessions[req.params.id]; res.json({ cleared: true }); });

app.get('*', (req, res) => { res.sendFile(path.join(__dirname, 'client', 'index.html')); });

app.listen(PORT, () => {
  console.log('\\n========================================================================');
  console.log(` 🛡️  HM Chatbot active at http://localhost:${PORT}`);
  console.log(' 📦 ChromaDB Vector Store Status:', chromaReady ? '✓ Connected' : '✗ Not connected');
  console.log(' 🧬 Embedding Engine Source: Local Pathing Configurations');
  console.log('\\n📋 Getting Started:');
  console.log('   1. Start Chroma: chroma run --host 127.0.0.1 --port 8000');
  console.log('   2. Upload PDF: POST /api/upload');
  console.log('   3. Ask questions: POST /api/chat');
  console.log('   4. Check status: GET /api/health');
  console.log('========================================================================\\n');
});