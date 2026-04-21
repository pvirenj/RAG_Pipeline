# 📚 StudyAI — Modular RAG Pipeline

An end-to-end, production-ready Retrieval-Augmented Generation (RAG) pipeline built for academic research. This project parses textbooks locally, stores semantic embeddings in a containerized Qdrant vector database, and generates highly detailed, cited answers using Google Gemini with graceful LLM fallbacks and Advanced RAG retrieval (Multi-Query + Re-ranking).

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          StudyAI RAG Pipeline                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐                                      │
│  │ Streamlit UI  │     │   CLI        │                                      │
│  │ (future)      │     │  main.py     │                                      │
│  └──────┬───────┘     └──────┬───────┘                                      │
│         │                    │                                               │
│         ▼                    ▼                                               │
│  ┌──────────────┐     ┌──────────────┐                                      │
│  │  api/         │     │  Direct      │                                      │
│  │  FastAPI      │     │  Function    │                                      │
│  │  server.py    │     │  Calls       │                                      │
│  │  routes/      │     │              │                                      │
│  └──────┬───────┘     └──────┬───────┘                                      │
│         │                    │                                               │
│         └────────┬───────────┘                                               │
│                  ▼                                                           │
│  ┌───────────────────────────────────────────────────────────────────┐       │
│  │                    src/ (Core Business Logic)                     │       │
│  │                                                                   │       │
│  │  INGESTION PIPELINE                                               │       │
│  │  ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐    │       │
│  │  │ Docling   │──▶│ Chunking │──▶│ Embeddings │──▶│  Qdrant  │    │       │
│  │  │ Parser    │   │ 2-Pass   │   │ Qwen3-VL   │   │ Vector DB│    │       │
│  │  └──────────┘   └──────────┘   └────────────┘   └────┬─────┘    │       │
│  │                                                       │          │       │
│  │  QUERY PIPELINE (Advanced)                            │          │       │
│  │  ┌──────────────┐                                     │          │       │
│  │  │ Multi-Query   │ Gemini rewrites into 3 queries     │          │       │
│  │  └──────┬───────┘                                     │          │       │
│  │         ▼                                             │          │       │
│  │  ┌──────────────┐                                     │          │       │
│  │  │ Qdrant Search │◀───────────────────────────────────┘          │       │
│  │  └──────┬───────┘ (k=30 chunks)                                  │       │
│  │         ▼                                                         │       │
│  │  ┌──────────────┐                                                 │       │
│  │  │ BGE Re-ranker │ Cross-encoder reads all 30 → keeps top 5      │       │
│  │  └──────┬───────┘                                                 │       │
│  │         ▼                                                         │       │
│  │  ┌──────────────┐                                                 │       │
│  │  │ Gemini LLM   │ 2.5-Flash (fallback: 1.5-Pro)                  │       │
│  │  │ + Citations  │ Generates textbook-style answer                 │       │
│  │  └──────────────┘                                                 │       │
│  └───────────────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧱 Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| **Data Parsing** | [Docling](https://github.com/DS4SD/docling) (IBM) | Converts PDF → structured Markdown |
| **Chunking** | LangChain `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` | 2-pass splitting: logical (chapters) then physical (token limits) |
| **Embeddings** | `Qwen/Qwen3-VL-Embedding-2B` (Local, Apple Silicon MPS) | Converts text chunks → 2048-dim dense vectors |
| **Vector Database** | Qdrant (Dockerized, API Key auth) | Stores & searches vector embeddings with payload indexing |
| **Retrieval** | MMR / Advanced (Multi-Query + BGE Re-ranking) | Configurable retrieval strategies |
| **Re-ranker** | `BAAI/bge-reranker-base` (278M params, Local) | Cross-encoder that "reads" chunks for true relevance scoring |
| **LLM** | Google Gemini 2.5-Flash (Primary) + Gemini 1.5-Pro (Fallback) | Answer generation with automatic graceful degradation |
| **API Server** | FastAPI + Uvicorn | REST API with auto-generated Swagger docs |

---

## 📁 Project Structure

```
RAG_Pipeline/
├── main.py                    # CLI entry point (ingest / query / serve commands)
├── docker-compose.yaml        # Qdrant vector database container
├── requirements.txt           # Python dependencies
├── pyproject.toml             # UV project configuration
├── .env                       # API keys (GEMINI_API_KEY, QDRANT_API_KEY)
├── data/                      # PDF textbooks go here
│   └── *.pdf
├── api/                       # REST API layer (FastAPI)
│   ├── __init__.py
│   ├── server.py              # FastAPI app instance & router registration
│   ├── models.py              # Pydantic request/response schemas
│   └── routes/
│       ├── __init__.py
│       ├── health.py          # GET  /health
│       ├── collections.py     # GET  /collections
│       ├── ingest.py          # POST /ingest
│       └── query.py           # POST /query
└── src/                       # Core business logic
    ├── config.py              # Centralized configuration & environment variables
    ├── data_ingestion.py      # PDF → Markdown conversion using Docling
    ├── chunking.py            # 2-pass text splitting with metadata preservation
    ├── vector_embeddings.py   # Qdrant collection management, embedding & retrieval
    └── generation.py          # LLM chain, Multi-Query, Re-ranking & answer generation
```

---

## 🔧 Module Documentation

### `src/config.py` — Configuration Hub
Centralizes all settings. Loads API keys from `.env` and defines model names, chunk sizes, and retrieval parameters.

| Setting | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | `.env` | Google Gemini API key |
| `QDRANT_API_KEY` | `.env` | Qdrant authentication key |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server address |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-VL-Embedding-2B` | Local embedding model |
| `LLM_MODEL` | `gemini-2.5-flash` | Primary LLM |
| `FALLBACK_MODEL` | `gemini-1.5-pro` | Fallback LLM (on server errors) |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between adjacent chunks |
| `RETRIEVAL_MODE` | `advanced` | Default retrieval strategy |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder re-ranking model |
| `API_HOST` | `0.0.0.0` | FastAPI server bind address |
| `API_PORT` | `8000` | FastAPI server port |

---

### `src/data_ingestion.py` — Document Parser
Uses IBM's Docling to visually parse PDFs (including tables, figures, and multi-column layouts) into clean, structured Markdown. Includes front-matter cleaning to strip Table of Contents, Copyright pages, and Index sections.

---

### `src/chunking.py` — 2-Pass Text Splitter
**Pass 1 (Logical):** `MarkdownHeaderTextSplitter` splits by `#`, `##`, `###` headers, preserving chapter/section metadata in each chunk.

**Pass 2 (Physical):** `RecursiveCharacterTextSplitter` ensures each chunk fits within the embedding model's token window (1000 chars with 100 overlap).

Each chunk's metadata includes: `Header_1`, `Header_2`, `Header_3`, and `source` (filename).

---

### `src/vector_embeddings.py` — Vector Database Manager
Manages the full lifecycle of the Qdrant collection:

- **Collection Creation:** Auto-creates collections with correct vector dimensions and payload indexes on `metadata.Header_1` and `metadata.source`.
- **Smart Upsert:** Generates deterministic UUIDs (MD5 hash of content) to prevent duplicate vectors. Deletes old versions of a file before re-ingesting.
- **Batch Upload:** Uploads in batches of 20 with a `tqdm` progress bar.
- **Configurable Retrieval:** Supports `mmr` (diversity-focused, k=10) and `advanced` (broad fetch, k=30 for re-ranking).

---

### `src/generation.py` — RAG Generation Engine
The brain of the system. Handles:

1. **LLM Initialization:** Primary (Gemini 2.5-Flash) + Fallback (Gemini 1.5-Pro) using `.with_fallbacks()` for automatic graceful degradation.
2. **Advanced Retrieval Pipeline:**
   - **Layer 1 — Multi-Query:** Uses Gemini to rewrite the user's question into 3 variations, maximizing recall.
   - **Layer 2 — Re-ranking:** `BAAI/bge-reranker-base` cross-encoder reads all retrieved chunks and scores true relevance. Keeps top 5.
   - **Layer 3 — Compression:** `ContextualCompressionRetriever` combines Multi-Query + Re-ranking into a single retriever interface.
3. **QA Chain:** `create_stuff_documents_chain` injects the top chunks into a carefully crafted system prompt that enforces academic tone, detailed explanations, and source citations.

---

### `api/` — REST API Layer (FastAPI)
A thin HTTP wrapper around the `src/` modules. No business logic lives here — it only handles HTTP concerns (request parsing, file uploads, error responses).

| File | Endpoint | Method | Description |
|---|---|---|---|
| `routes/health.py` | `/health` | `GET` | Pings Qdrant, returns connection status |
| `routes/collections.py` | `/collections` | `GET` | Lists all knowledge base collections |
| `routes/ingest.py` | `/ingest` | `POST` | Accepts PDF upload via `multipart/form-data`, runs full ingestion pipeline |
| `routes/query.py` | `/query` | `POST` | Accepts JSON `{collection, question}`, returns `{answer, sources}` |
| `models.py` | — | — | Pydantic schemas for request/response validation |
| `server.py` | — | — | FastAPI app instance, registers all routers |

---

## 🚀 Setup Guide

### Prerequisites
- Python 3.11+
- [UV](https://docs.astral.sh/uv/) package manager
- Docker & Docker Compose
- Apple Silicon Mac (for MPS acceleration) or CUDA GPU

### 1. Clone & Setup Environment
```bash
git clone <repo-url>
cd RAG_Pipeline
uv venv --python 3.11
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
uv add -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_URL=http://localhost:6333
```

### 4. Start Qdrant Database
```bash
docker-compose up -d
```
Verify at: http://localhost:6333/dashboard

### 5. Add Data
```bash
mkdir -p data
# Place your PDF textbooks into the ./data folder
```

---

## 📖 Usage

### Ingest a PDF
```bash
python main.py ingest --file ./data/[PDF_FILE_NAME].pdf --collection learning_knowledge_base
```

**What happens:**
```
PDF File
  │
  ▼
[Docling] Converts PDF → Markdown (preserves structure, tables, figures)
  │
  ▼
[Front-Matter Cleaning] Strips ToC, Copyright, Index
  │
  ▼
[MarkdownHeaderTextSplitter] Splits by # / ## / ### headers
  │
  ▼
[RecursiveCharacterTextSplitter] Splits into 1000-char chunks (100 overlap)
  │
  ▼
[Source Tagging] Adds filename to each chunk's metadata
  │
  ▼
[MD5 Hashing] Generates deterministic UUIDs per chunk
  │
  ▼
[Old Version Cleanup] Deletes existing chunks for this file from Qdrant
  │
  ▼
[Qwen3-VL Embedding] Converts chunks → 2048-dim vectors (on Apple MPS)
  │
  ▼
[Batch Upload] Stores in Qdrant in batches of 20 with progress bar
  │
  ▼
✅ Done! Chunks are searchable in Qdrant.
```

### Query the Knowledge Base
```bash
python main.py query --collection learning_knowledge_base --question "Explain how backpropagation works"
```

**What happens:**
```
User Question: "Explain how backpropagation works"
  │
  ▼
[Multi-Query Expansion] Gemini rewrites into 3 query variations:
  1. "How does the backpropagation algorithm compute gradients?"
  2. "What is the chain rule in the context of neural network training?"
  3. "Explain the forward and backward pass in deep learning"
  │
  ▼
[Qdrant Vector Search] Runs ALL 3 queries → retrieves ~30 unique chunks
  │
  ▼
[BGE Re-ranker] Cross-encoder READS each chunk alongside the question
  Scores relevance 0.0 → 1.0 → keeps only the top 5 chunks
  │
  ▼
[Prompt Construction] Top 5 chunks injected into StudyAI system prompt
  │
  ▼
[Gemini 2.5-Flash] Generates detailed, textbook-lecture-style answer
  (If Flash fails → automatic fallback to Gemini 1.5-Pro)
  │
  ▼
[Citation Extraction] Extracts chapter/section names from chunk metadata
  │
  ▼
✅ Answer returned with sources
```

---

## 🌐 API Usage

### Start the API Server
```bash
python main.py serve
```
Swagger UI available at: **http://localhost:8000/docs**

### Health Check
```bash
curl http://localhost:8000/health
```
```json
{"status": "healthy", "qdrant": "connected"}
```

### List Collections
```bash
curl http://localhost:8000/collections
```
```json
{"collections": [{"name": "learning_knowledge_base"}]}
```

### Ingest a PDF via API
```bash
curl -X POST http://localhost:8000/ingest \
  -F "collection=learning_knowledge_base" \
  -F "file=@./data/[PDF_FILE_NAME].pdf"
```
```json
{"status": "success", "filename": "[PDF_FILE_NAME].pdf", "collection": "learning_knowledge_base", "chunks_stored": 758}
```

### Query via API
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"collection": "learning_knowledge_base", "question": "What is backpropagation?"}'
```
```json
{"answer": "Backpropagation is...", "sources": ["Training Feed-Forward Neural Networks", "Implementing Backpropagation"]}
```

---

## 🔀 Retrieval Modes

| Mode | Strategy | Speed | Accuracy | Use Case |
|---|---|---|---|---|
| `mmr` | Max Marginal Relevance (k=10 from top 30) | ⚡ Fast | ✅ Good | General questions, broad topic exploration |
| `advanced` | Multi-Query + BGE Re-ranking (30 → top 5) | 🐢 +1-2s | 🎯 Best | Specific technical questions, exact term lookup |

Configure the default in `src/config.py`:
```python
RETRIEVAL_MODE = "mmr"  # or "advanced"
```

Or set per-query in `main.py`:
```python
retriever = qdrant_manager.get_retriever(mode="advanced")
```

---

## 🛡️ Production Features

### Graceful LLM Degradation
If Gemini 2.5-Flash returns a server error (500, timeout, rate limit), the system **automatically and silently** retries with Gemini 1.5-Pro. The user never sees a crash.

### Deterministic Vector IDs
Each chunk gets a stable UUID derived from `MD5(filename + content)`. Re-ingesting the same file replaces old vectors instead of creating duplicates.

### Idempotent Ingestion
Before uploading new chunks, the pipeline deletes all existing chunks matching the source filename. This means you can re-run ingestion safely without manual cleanup.

### Payload Indexing
Qdrant payload indexes on `metadata.Header_1` and `metadata.source` enable fast filtered searches and efficient deletion by source file.

---

## 🗺️ Roadmap

- [x] **FastAPI Server** — REST API endpoints for ingest, query, and collection management
- [ ] **Streamlit UI** — Web interface for document upload and interactive Q&A
- [ ] **Multi-Modal Ingestion** — Support `.docx`, `.xlsx`, and web URL scraping via Docling
- [ ] **Hybrid Search** — Dense + BM25 sparse vectors for exact keyword matching
- [ ] **Streaming Responses** — Stream LLM output token-by-token via SSE
- [ ] **Authentication** — API key middleware for the FastAPI server

---

## 📄 License
MIT
