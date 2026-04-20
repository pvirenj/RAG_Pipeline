import hashlib
import uuid
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType, Filter, FieldCondition, MatchValue
from src.config import cfg

class QdrantManager:
  def __init__(self, collection_name: str):
    self.collection_name = collection_name
    
    print(f"🚀 Loading Embedding Model: {cfg.EMBEDDING_MODEL}")
    self.embeddings = HuggingFaceEmbeddings(
      model_name=cfg.EMBEDDING_MODEL,
      model_kwargs={'device':'mps'}
    )
    print("✅ Embedding model loaded successfully")

    print(f"🚀 Connecting to Qdrant at {cfg.QDRANT_URL}")
    # Initialize Local Dockerize Qdrant
    self.client = QdrantClient(
      url=cfg.QDRANT_URL,
      api_key=cfg.QDRANT_API_KEY
    )
    self._ensure_collection_exists()
    
  def _ensure_collection_exists(self):
    """Check if collection exists, if not create it."""
    if not self.client.collection_exists(collection_name=self.collection_name):
      print(f"🚀 Creating collection: {self.collection_name}")
      self.client.create_collection(
        collection_name=self.collection_name,
        vectors_config=VectorParams(
          size=len(self.embeddings.embed_query("test")),
          distance=Distance.COSINE
        )
      )
      
      print("⚡ Building Payload Indexes for metadata...")
      
      # 2. Index the Markdown Chapter Headers
      self.client.create_payload_index(
        collection_name=self.collection_name,
        field_name="metadata.Header_1",
        field_type=PayloadSchemaType.KEYWORD
      )

      # 3. Index the Source File Name
      self.client.create_payload_index(
        collection_name=self.collection_name,
        field_name="metadata.source",
        field_type=PayloadSchemaType.KEYWORD
      )

      print(f"✅ Collection {self.collection_name} created successfully")
  
  def store_embeddings(self, documents):
    """Embeds and upserts chunks to Qdrant using deterministic IDs. Wipes old file data, then embeds and upserts new chunks to Qdrant."""
    print(f"🚀 Storing embeddings in collection: {self.collection_name}")
    print(f"Documents to store: {len(documents)}")

    if not documents:
      print("⚠️ No documents to store.")
      return

    # 1. Identify the source file we are updating
    # We assume all chunks in this batch come from the same file    
    
    source_file = documents[0].metadata.get("source", "unknown")
    print(f"🧹 Checking for old versions of '{source_file}' in Qdrant...")

    # 2. Delete ALL existing chunks that match this exact source file
    # Because we indexed metadata.source, this is blazing fast!
    self.client.delete(
      collection_name=self.collection_name,
      points_selector=Filter(
        must=[FieldCondition(key="metadata.source", match=MatchValue(value=source_file))]
      ),
    )
    print(f"✅ Deleted old version of '{source_file}'")

    # 3. Generate stable UUIDs for the new chunks
    print(f"💾 Upserting {len(documents)} fresh chunks...")
    ids = []
    for doc in documents:
      content = doc.page_content
      unique_string = f"{source_file}-{content}"
      chunk_hash = hashlib.md5(unique_string.encode("utf-8")).hexdigest()
      stable_id = str(uuid.UUID(chunk_hash))
      ids.append(stable_id)
      
    # 4. Upload to Qdrant
    #QdrantVectorStore.from_documents(
    #  documents=documents,
    #  embedding=self.embeddings,
    #  url=cfg.QDRANT_URL,
    #  api_key=cfg.QDRANT_API_KEY,
    #  collection_name=self.collection_name,
    #  ids=ids
    #)
    vector_store = QdrantVectorStore(
      client=self.client, 
      collection_name=self.collection_name,
      embedding=self.embeddings
    )

    batch_size = 20
    for i in tqdm(range(0, len(documents), batch_size), desc="📥 Uploading chunks to Qdrant"):
      batch_docs = documents[i : i + batch_size]
      batch_ids = ids[i : i + batch_size]
      vector_store.add_documents(documents=batch_docs, ids=batch_ids)

    print(f"✅ Embeddings stored successfully in collection: {self.collection_name}")
    
  def get_retriever(self, mode=cfg.RETRIEVAL_MODE):
    """Returns a retriever object so the LLM can search the DB. We have to use MMR Algorithms for searching better and get rid of issue: Retrival Diversity
    
    MMR: Max Marginal Relevance (MMR) is an algorithm used to select a subset of items from a larger set, such as documents from a collection, that are both relevant to a query and diverse from each other.
    """
    #qdrant_retriever = QdrantVectorStore(
    #  client=self.client,
    #  collection_name=self.collection_name,
    #  embedding=self.embeddings
    #)
    #return qdrant_retriever.as_retriever(search_kwargs={"k": 3})
    qdrant_retriever = QdrantVectorStore(
      client=self.client,
      collection_name=self.collection_name,
      embedding=self.embeddings
    )
    
    if mode == "advanced":
      # Cast a wide net — the re-ranker will filter down to top 5
      return qdrant_retriever.as_retriever(search_kwargs={"k": 30})

    # search_type="mmr" for diversity
    # fetch_k=30 means "grab the top 30 chunks first"
    # k=10 means "pick the 10 most diverse chunks out of that 30"
    return qdrant_retriever.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 30})


    