import os
from dotenv import load_dotenv

# load env from os
load_dotenv()

class Config:
  # API Keys
  GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

  # Path
  DATA_DIR="./data"
  
  # Qdrant Configuration
  QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
  QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

  # Embedding Model
  EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-2B"

  # LLM Models
  LLM_MODEL = "gemini-2.5-flash"
  FALLBACK_MODEL = "gemini-1.5-pro"

  # Chunk Size
  CHUNK_SIZE = 1000
  CHUNK_OVERLAP = 100

  # Retrieval Settings
  RETRIEVAL_MODE = "mmr" # default mode
  RERANK_MODEL = "BAAI/bge-reranker-base"

# Intantiating the config
cfg = Config()