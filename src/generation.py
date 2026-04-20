from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from src.config import cfg
# for Advance RAG
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

class RAGGeneration:
  def __init__(self, retriever):
    self.retriever = retriever
    
    # Initialize Primary LLM
    print(f"🚀 Loading Primary LLM: {cfg.LLM_MODEL}")
    primary_llm = ChatGoogleGenerativeAI(
      model=cfg.LLM_MODEL,
      google_api_key=cfg.GEMINI_API_KEY,
      temperature=0.0 # Standardized for consistency
    )

    # Initialize Fallback LLM
    print(f"🚀 Initializing Fallback LLM: {cfg.FALLBACK_MODEL}")
    fallback_llm = ChatGoogleGenerativeAI(
        model=cfg.FALLBACK_MODEL,
        google_api_key=cfg.GEMINI_API_KEY,
        temperature=0.0 # Stable fallback
    )

    # Combine with Fallbacks
    self.llm = primary_llm.with_fallbacks([fallback_llm])
    print("✅ LLM with fallback mechanism ready")

    # 1. Build the Advanced Retrieval Pipeline
    self.advanced_retriever = self._build_advanced_retriever(self.retriever)

    self.qa_chain = self._build_chain()

  def _build_advanced_retriever(self, base_retriever):
    """Constructs the Multi-Query + Re-ranking Pipeline."""
    print("⚙️ Building Advanced Retrieval Pipeline...")
    # 1. Multi-Query Expansion
    # Uses Gemini to generate 3 variations of the user's prompt to maximize recall
    multi_query_retriever = MultiQueryRetriever.from_llm(
      retriever=base_retriever,
      llm=self.llm
    )

    # LAYER 2: RE-RANKING (Cross-Encoder)
    # Uses a local BAAI model to read all retrieved chunks and score them accurately
    print("🧠 Loading Re-ranker Model: BAAI/bge-reranker-base...")
    cross_encoder_model = HuggingFaceCrossEncoder(model_name=cfg.RERANK_MODEL)
    
    # Keep only the top 5 most relevant chunks after re-ranking
    reranker_compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=5)
    
    # LAYER 3: COMBINE
    # Combines the Multi-Query expansion with the Re-ranking compression
    advanced_retriever = ContextualCompressionRetriever(
        base_retriever=multi_query_retriever,
        base_compressor=reranker_compressor
    )
    
    return advanced_retriever
    
  def _build_chain(self):
    """Construct the LangChain RAG pipeline."""
    # Prompt Template
    system_prompt = (
      "You are 'StudyAI', an expert professor and academic tutor in deep learning. "
      "Your goal is to provide highly detailed, comprehensive, and elaborated answers based ONLY on the provided context.\n\n"
      "--- CONTEXT RULES ---\n"
      "1. Answer based ONLY on the provided context. Do not use outside knowledge. "
      "If the information is missing, explicitly state what is missing.\n\n"
      "--- FORMATTING & DEPTH RULES ---\n"
      "1. Elaborate extensively. Do not just give a brief summary. Explain the 'how' and 'why' in detail using the retrieved text.\n"
      "2. Break down complex concepts into digestible, logical steps. If the text provides examples or analogies, include them.\n"
      "3. Structure your response like a textbook lecture using clear Markdown headings, bullet points, and bold text for readability.\n"
      "4. Your goal is to explain the concept so thoroughly that the student does not need to open the book themselves.\n\n"
      "--- CITATION RULES ---\n"
      "1. At the end of your answer, list the sources you used.\n\n"
      "Context: {context}"
        )
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", system_prompt),
      ("human", "{input}")
    ])

    # This chains stuffs the retrieved documents into the prompt
    question_answer_chain = create_stuff_documents_chain(self.llm, prompt)

    # This chain combines the retriever with the question_answer_chain
    rag_chain = create_retrieval_chain(self.advanced_retriever, question_answer_chain)
    return rag_chain

  def answer_question(self, question: str ) -> str:
    """Takes a user question, searches Qdrant, and generates an answer."""
    print(f"\n❓ Question: {question}")
    response = self.qa_chain.invoke({"input": question})
    answer = response["answer"]
    
    # Extract sources from the 'context' (retrieved docs)
    sources = set()
    if "context" in response:
        for doc in response["context"]:
            # Check for header metadata from MarkdownHeaderTextSplitter
            source_name = doc.metadata.get("Header 1") or doc.metadata.get("source", "Unknown Section")
            sources.add(source_name)
    
    print(f"\n✅ Answer generated. Sources found: {sources}")
    # You can choose to append the sources to the answer manually or let the LLM do it
    return f"{answer}\n\n**Sources:** {', '.join(sources)}"
        
    

    

  