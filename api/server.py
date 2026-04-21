from fastapi import FastAPI
from api.routes import health, ingest, query, collections

# Create the FastAPI application
app = FastAPI(
    title="StudyAI RAG API",
    description="REST API for ingesting documents and querying the RAG knowledge base",
    version="1.0.0"
)

# Register route files
# Each router handles a specific resource (health, ingest, query, etc.)
app.include_router(health.router, tags=["System"])
app.include_router(collections.router, tags=["Collections"])
app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(query.router, tags=["Query"])