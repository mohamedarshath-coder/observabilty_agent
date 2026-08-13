"""
Local ONNX-based embeddings for ragas — no network call, no Ollama.

Why this exists: ragas's AnswerRelevancy metric needs a real embedding model
regardless of which LLM is used as the judge. This project originally always
routed that through Ollama, even when Claude is the judge — but Ollama's model
registry (registry.ollama.ai) is blocked by this organization's Zscaler proxy
for large binary downloads (confirmed live: the "download" silently came back
as a Zscaler interception page instead of the model weights). Anthropic has no
embeddings API at all, so Claude can't substitute either.

This reuses the SAME Sentance_Transformer model the Node backend already loads
for retrieval (see server.js's getLocalEmbedding) — same mean-pooling + L2-
normalize pipeline, confirmed against Sentance_Transformer/1_Pooling/config.json
(pooling_mode_mean_tokens) and modules.json (Transformer -> Pooling -> Normalize),
so ragas's embedding space stays consistent with what the app's own retrieval
already uses. Loaded directly via onnxruntime + tokenizers (small, ~15MB
combined) instead of the heavier sentence-transformers/torch stack, and
entirely offline once these two small packages are installed from PyPI.
"""

import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from langchain_core.embeddings import Embeddings

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "Sentance_Transformer")
_MAX_SEQ_LENGTH = 256  # matches Sentance_Transformer/sentence_bert_config.json


class LocalSentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible Embeddings implementation (embed_documents/embed_query
    are the only methods required — the async aembed_* variants come for free from
    langchain_core.embeddings.Embeddings' default executor-based implementation)."""

    _session = None
    _tokenizer = None

    def __init__(self):
        if LocalSentenceTransformerEmbeddings._session is None:
            onnx_path = os.path.join(_MODEL_DIR, "onnx", "model.onnx")
            LocalSentenceTransformerEmbeddings._session = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )
            tok = Tokenizer.from_file(os.path.join(_MODEL_DIR, "tokenizer.json"))
            tok.enable_truncation(max_length=_MAX_SEQ_LENGTH)
            LocalSentenceTransformerEmbeddings._tokenizer = tok

    def _embed(self, texts):
        tok = LocalSentenceTransformerEmbeddings._tokenizer
        session = LocalSentenceTransformerEmbeddings._session

        encodings = tok.encode_batch(list(texts))
        max_len = max(len(e.ids) for e in encodings)
        input_ids = np.array(
            [e.ids + [0] * (max_len - len(e.ids)) for e in encodings], dtype=np.int64
        )
        attention_mask = np.array(
            [e.attention_mask + [0] * (max_len - len(e.attention_mask)) for e in encodings],
            dtype=np.int64,
        )
        token_type_ids = np.zeros_like(input_ids)

        # Confirmed via onnxruntime introspection: this model takes exactly these
        # three inputs and returns one output, last_hidden_state (batch, seq, 384) —
        # no built-in pooling, so mean-pooling + normalize below matches Node's
        # `{ pooling: 'mean', normalize: true }` call on the same model.
        (token_embeddings,) = session.run(
            ["last_hidden_state"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )

        mask = attention_mask[..., None].astype(np.float32)
        summed = (token_embeddings * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        mean_pooled = summed / counts

        norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        normalized = mean_pooled / np.clip(norms, a_min=1e-9, a_max=None)
        return normalized.tolist()

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]
