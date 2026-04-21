from fastapi import APIRouter, HTTPException
from src.config import cfg
from src.vector_embeddings import QdrantManager
from src.generation import RAGGeneration
from api.models import QueryRequest, QueryResponse

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
def query_knowledge_base(request: QueryRequest):
    """Ask a question to the RAG system and get a cited answer."""
    try:
        # 1. Connect to the Qdrant collection
        qdrant_manager = QdrantManager(request.collection)
        retriever = qdrant_manager.get_retriever(mode=cfg.RETRIEVAL_MODE)

        # 2. Build the RAG chain (LLM + Advanced Retriever)
        rag = RAGGeneration(retriever)

        # 3. Run the chain — this triggers Multi-Query → Qdrant → Re-ranking → Gemini
        response = rag.qa_chain.invoke({"input": request.question})
        answer = response["answer"]

        # 4. Extract sources from retrieved context
        sources = []
        if "context" in response:
            for doc in response["context"]:
                source_name = doc.metadata.get("Header_1") or doc.metadata.get("source", "Unknown")
                if source_name not in sources:
                    sources.append(source_name)

        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")