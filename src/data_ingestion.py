from pathlib import Path
from docling.document_converter import DocumentConverter
from src.config import cfg

class DataIngestion:
    def __init__(self):
      # We initialize the converter once when the class is created
      self.converter = DocumentConverter()

    def load_pdf_to_markdown(self, pdf_path: str) -> str :
      """
      Load a PDF file and return its text content in markdown format.
      """
      path = Path(pdf_path)
      if not path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")
      if not path.suffix == ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")
      try:

        print(f"📄 Ingesting and Parsing PDF: {path.name}...")
        result = self.converter.convert(path)
        markdown_content = result.document.export_to_markdown()
        print(f"✅ Successfully converted {path.name} to markdown")
        
        # Cleaning The Table of Contents, the Copyright page, the Preface, and the Index. They have no instructional value for the LLM
        # Find where the actual book starts (e.g., Chapter 1's title)
        # and slice off everything before it (Title page, ToC, Dedication)

        start_keyword="## Building Intelligent Machines"
        if start_keyword in markdown_content:
          markdown_content = markdown_content[markdown_content.index(start_keyword):]
          print(f"✅ Cleaned up front matter")
        else:
          print(f"⚠️ Could not find start keyword '{start_keyword}'")

        return markdown_content
      except Exception as e:
        print(f"❌ Error while ingesting and parsing PDF: {e}")
        return ""
      