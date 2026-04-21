from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pathlib import Path
from src.config import cfg
from src.data_ingestion import DataIngestion
from src.chunking import Chunking
from src.vector_embeddings import QdrantManager
from api.models import IngestResponse

router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    collection: str = Form(default="learning_knowledge_base"),
    file: UploadFile = File(...)
):
    """Upload a PDF file and ingest it into the vector database."""
    
    # 1. Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # 2. Save the uploaded file to ./data/
    save_path = Path(cfg.DATA_DIR) / file.filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        # 3. Parse PDF → Markdown
        ingestion = DataIngestion()
        markdown_text = ingestion.load_pdf_to_markdown(str(save_path))

        # 4. Chunk the Markdown
        splitter = Chunking()
        chunks = splitter.split_markdown(markdown_text, file.filename)

        # 5. Embed and Store in Qdrant
        qdrant_manager = QdrantManager(collection)
        qdrant_manager.store_embeddings(chunks)

        return IngestResponse(
            status="success",
            filename=file.filename,
            collection=collection,
            chunks_stored=len(chunks)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")