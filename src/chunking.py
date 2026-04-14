from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from src.config import cfg

class Chunking:
    def __init__(self):
      # Pass 1: Split logically by chapters and sections
      headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
      ]
      self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

      # Pass 2: Split physically to fit embedding model limits
      self.char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP
      )
    
    def split_markdown(self, markdown_content: str, source_file: str):
      """Takes a raw markdown string and returns LangChain Document chunks."""
      print("✂️ Chunking markdown text...")

      # First split by headers
      md_header_split = self.markdown_splitter.split_text(markdown_content)

      # Second split remaining text into smaller chunks
      final_chunks = self.char_splitter.split_documents(md_header_split)
      
      # Add source file to metadata
      for chunk in final_chunks:
        chunk.metadata["source"] = source_file
      print(f"✅ Split into {len(final_chunks)} chunks")
      return final_chunks

      
        
