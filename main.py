import argparse
from pathlib import Path
from src.data_ingestion import DataIngestion
from src.chunking import Chunking
from src.vector_embeddings import QdrantManager
from src.generation import RAGGeneration

def ingest_data(pdf_path: str, collection_name: str):
  """Handles the heavy lifting: Parsing, Chunking and Storing."""
  print(f"\n🚀 Staring ingesting pipeline for: {pdf_path}")

  # Step 1: Ingest
  ingestion = DataIngestion()
  markdown_text = ingestion.load_pdf_to_markdown(pdf_path)
  
  # Step 2: Chunk
  splitter = Chunking()
  chunks = splitter.split_markdown(markdown_text, Path(pdf_path).name)
  
  # Step 3 & 4: Embed and Store in Qdrant
  qdrant_manager = QdrantManager(collection_name)
  qdrant_manager.store_embeddings(chunks)

  print("\n✅ Ingestion Complete! The data in now in Qdrant.")

def query_system(collection_name: str, question: str):
  """Handles fast retrieval and LLM generation."""
  print(f"\n🔎 Connecting to Qdrant collection: '{collection_name}'...")
  
  # We ONLY initialize what we need for querying
  qdrant_manager = QdrantManager(collection_name)
  retriever = qdrant_manager.get_retriever()
  rag_generation = RAGGeneration(retriever)
  
  # Generate answer
  answer = rag_generation.answer_question(question)
  print(f"\n Study AI:\n {answer}")
  
if __name__ == "__main__":
    #Setup the Argument Parser
    parser = argparse.ArgumentParser(description="StudyAI RAG Pipeline Manager")

    # Create 'subparsers' to handle entirely different commands (ingest vs query)
    subparsers = parser.add_subparsers(dest="command", required=True, help="Action to perform")

    # 1. Setup the 'ingest' command
    ingest_parser = subparsers.add_parser("ingest", help="Process a PDF and store it in Qdrant")
    ingest_parser.add_argument("--file", type=str, required=True, help="Path to the PDF file (e.g., ./data/book.pdf)")
    ingest_parser.add_argument("--collection", type=str, default="learning_knowledge_base", required=True, help="Name of the Qdrant collection")

    # 2. Setup the 'query' command
    query_parser = subparsers.add_parser("query", help="Ask a question to the RAG system")
    query_parser.add_argument("--collection", type=str, default="learning_knowledge_base", required=True, help="Name of the Qdrant collection")
    query_parser.add_argument("--question", type=str, required=True, help="Your question in quotes")

    # Parse the arguments
    args = parser.parse_args()

    # Route to the correct function
    if args.command == "ingest":
      ingest_data(args.file, args.collection)
    elif args.command == "query":
      query_system(args.collection, args.question)