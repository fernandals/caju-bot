from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.models import RawDocument, DocumentType, DocumentChunk

def spliting_documents(docs: list[RawDocument]) -> list[list[DocumentChunk]]:
  print("Fazendo split dos documentos...")

  text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True  
  )

  docs_chunks = []

  for doc in docs:
    all_chunks = []

    if doc.metadata["doc_type"] == DocumentType.SLIDES.value:

      for slide in doc.metadata.get("slides", []):
        content = f"[SLIDE TITLE]\n{slide['title']}\n\n[CONTENT]\n{slide['content']}"
        
        metadata = {
          "source": doc.metadata["source"],
          "file_path": doc.metadata["file_path"],
          "page": slide["page"],
          "doc_type": doc.metadata["doc_type"]
        }
        
        chunks = text_splitter.create_documents([content], metadatas=[metadata])
        all_chunks.extend(chunks)
    else:
      sections = doc.metadata.get("sections", [])

      base_metadata = {
        k: v for k, v in doc.metadata.items()
        if k not in ["sections"]
      }
      
      for section in sections:
        lc_doc = Document(
          page_content=section["content"],
          metadata={
            **base_metadata,
            "section_id": section["id"],
            "section_title": section["title"],
            "section_start": section["start"],
            "section_end": section["end"]
          }
        )

        chunks = text_splitter.split_documents([lc_doc])
        all_chunks.extend(chunks)

    docs_chunks.append(all_chunks)

  return docs_chunks