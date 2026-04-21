from fastapi import APIRouter, HTTPException
from qdrant_client import QdrantClient
from src.config import cfg

router = APIRouter()

@router.get("/collections")
def list_collections():
    """List all Qdrant collections (knowledge bases)."""
    try:
        client = QdrantClient(url=cfg.QDRANT_URL, api_key=cfg.QDRANT_API_KEY)
        collections = client.get_collections().collections
        return {
            "collections": [
                {"name": c.name} for c in collections
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {e}")