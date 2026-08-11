# ChromaDB + PDF RAG Setup Guide

## ✅ What's Fixed

Your server.js now has:
- ✓ Proper async ChromaClient initialization
- ✓ Connection validation with error handling
- ✓ Health check endpoint (`/api/health`)
- ✓ Correct ChromaDB API usage (no more raw HTTP v2 API calls)
- ✓ Vector embedding pipeline integration
- ✓ PDF extraction and chunking

## 🚀 Quick Start (3 Steps)

### Step 1: Start Chroma Server
Open a **new terminal** and run:
```bash
chroma run --host 127.0.0.1 --port 8000
```

You should see:
```
Chroma server is running...
Listening on 127.0.0.1:8000
```

### Step 2: Start the HM Chatbot Server
In **another terminal**, run:
```bash
npm start
```

You should see:
```
[Chroma RAG] ✓ ChromaDB connected and collection ready.
[RAG Engine] Ingestion Engine fully armed and ready to process files.

🛡️  HM Chatbot active at http://localhost:3001
📦 ChromaDB Vector Store Status: ✓ Connected
```

### Step 3: Verify Connection
Visit in your browser:
```
http://localhost:3001/api/health
```

Should return:
```json
{
  "status": "ok",
  "chromaConnected": true,
  "embeddingModelLoaded": true,
  "chromaMessage": "Connected to Cisco_Mask_Data collection"
}
```

---

## 📋 API Endpoints

### 1. Upload PDF
**POST** `/api/upload`
```bash
curl -X POST -F "file=@document.pdf" http://localhost:3001/api/upload
```

Response:
```json
{
  "success": true,
  "filename": "document.pdf",
  "totalChunks": 45,
  "embeddingsAdded": 45
}
```

### 2. Ask Questions About PDF
**POST** `/api/chat`
```bash
curl -X POST http://localhost:3001/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is mentioned about X in the document?",
    "model_key": "gemini-2.0-flash"
  }'
```

Response includes:
- `response`: Answer grounded in your PDF
- `verdict`: Hallucination detection result
- `ensemble_score`: Confidence score

### 3. View Uploaded Files
**GET** `/api/files`

### 4. Clear All Documents
**DELETE** `/api/files`

### 5. Health Check
**GET** `/api/health`

---

## ⚠️ Troubleshooting

### Issue: "ChromaDB not connected"

**Cause:** Chroma server is not running

**Fix:**
```bash
# Terminal 1: Start Chroma
chroma run --host 127.0.0.1 --port 8000

# Terminal 2: Start your app
npm start
```

---

### Issue: "Rust compilation error" or "Error: ENOENT"

**Cause:** Old Chroma dependency or corrupted node_modules

**Fix:**
```bash
# Clear everything and reinstall
rm -r node_modules package-lock.json
npm install
```

---

### Issue: "Failed to download model" or "EMBEDDING_MODEL_PATH is missing"

**Check:**
- Your `.env` file has `EMBEDDING_MODEL_PATH` pointing to the correct folder
- The folder exists and contains `pytorch_model.bin` or `model.safetensors`

```bash
# Verify path
ls "C:\Users\shree.sr.lv\Downloads\hm-chatbot\hm-chatbot\Sentance_Transformer"
```

---

### Issue: PDF Upload Returns 503 Error

**Cause:** ChromaDB not ready when file uploaded

**Fix:**
1. Verify Chroma server is running: `curl http://127.0.0.1:8000/api/heartbeat`
2. Check server logs for connection status
3. Wait 5-10 seconds after starting for full initialization

---

## 🔧 How It Works

```
PDF Upload
    ↓
Extract Text (unpdf library)
    ↓
Split into Chunks (500 word chunks with 100 word overlap)
    ↓
Generate Embeddings (Local Sentence Transformers model)
    ↓
Store in ChromaDB (Cisco_Mask_Data collection)
    ↓
User Question
    ↓
Generate Query Embedding
    ↓
Search ChromaDB (Find 3 most relevant chunks)
    ↓
Add Context to System Prompt
    ↓
Send to Gemini/Together AI
    ↓
Run through Hallucination Manager
    ↓
Return Response to User
```

---

## 📦 Key Libraries

- **chromadb**: Vector database
- **@xenova/transformers**: Local embedding model
- **unpdf**: PDF extraction
- **multer**: File upload handling
- **express**: REST API framework

---

## ✨ Features

✅ Upload PDFs via web UI or API  
✅ Automatic text extraction and chunking  
✅ Semantic search using local embeddings  
✅ Grounded answers (from your PDF context)  
✅ Hallucination detection on all responses  
✅ Session management  
✅ Multi-model support (Gemini, Together AI)

---

## 🛑 Start Fresh

If you want to reset everything:

```bash
# 1. Stop the servers (Ctrl+C in both terminals)

# 2. Clear ChromaDB data (optional)
rm -r chroma_data

# 3. Clear uploaded files tracking
# (automatic on server restart)

# 4. Start fresh
chroma run --host 127.0.0.1 --port 8000
npm start
```

---

Need help? Check the logs in both terminal windows for detailed error messages.
