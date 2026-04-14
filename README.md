# RAG_Pipeline
Retrival Augmented Generation Pipeline

# Modular RAG Pipeline

An end-to-end, production-ready Retrieval-Augmented Generation (RAG) pipeline built for deep learning textbooks. This project uses local document parsing, local vector embeddings, and Google Gemini for highly accurate, cited answers.

## 🏗️ Architecture Stack
* **Data Parsing:** [Docling](https://github.com/DS4SD/docling) (IBM's visual document parser)
* **Chunking:** LangChain `MarkdownHeaderTextSplitter`
* **Embeddings:** `Qwen3-VL-Embedding-2B` (Running locally via HuggingFace)
* **Vector Database:** Qdrant (Local disk mode)
* **LLM:** Google Gemini 1.5 Flash

## 🚀 Initial Setup

### 1. Environment Variables
Create a file named `.env` in the root directory and add your Google Gemini API key:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 2. Initialize & Create Virtual Environment
```bash
uv init
uv venv --python 3.11
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
uv add -r requirements.txt
```

### 4. make data directory
mkdir data
# Move FundamentalsofDeepLearning.pdf into the /data folder
