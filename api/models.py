from pydantic import BaseModel

# ============================================================
# REQUEST MODELS — What the client sends TO the API
# ============================================================

class QueryRequest(BaseModel):
    """The JSON body for POST /query"""
    collection: str = "learning_knowledge_base"
    question: str


# ============================================================
# RESPONSE MODELS — What the API sends BACK to the client
# ============================================================

class IngestResponse(BaseModel):
    """Returned after a successful PDF ingestion"""
    status: str
    filename: str
    collection: str
    chunks_stored: int

class QueryResponse(BaseModel):
    """Returned after a successful RAG query"""
    answer: str
    sources: list[str]

class HealthResponse(BaseModel):
    """Returned by GET /health"""
    status: str
    qdrant: str