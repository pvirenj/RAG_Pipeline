from fastapi import APIRouter, HTTPException
from qdrant_client import QdrantClient
from src.config import cfg
from api.models import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API server and Qdrant database are reachable."""
    try:
        client = QdrantClient(url=cfg.QDRANT_URL, api_key=cfg.QDRANT_API_KEY)
        client.get_collections()  # This will throw if Qdrant is down
        return HealthResponse(status="healthy", qdrant="connected")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant unreachable: {e}")