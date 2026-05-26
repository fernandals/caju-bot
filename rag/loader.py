import os
import re
import fitz

from rag.models import RawDocument, DocumentType

from utils.helpers import detect_pdf_type, extract_slide_structure

def load_documents(folder_pth: str) -> list[RawDocument]:
  print(f"Carregando PDFs da pasta '{folder_pth}' ...")

  docs = []

  for file_pth in folder_pth.glob("**/*.pdf"): # type: ignore
    print(f"Carregando {file_pth} ...")
    
    doc = fitz.open(file_pth)
    
    num_pages = doc.page_count
    doc_type = detect_pdf_type(doc[0])

    full_text = ""
    page_offsets = []

    for page_number, page in enumerate(doc): # type: ignore
      text = page.get_text("text")
      page_offsets.append({
        "page": page_number + 1,
        "start": len(full_text),
        "text": text
      })
      full_text += text + "\n"

    metadata = {
      "source": str(file_pth),
      "file_path": str(os.path.basename(file_pth)),
      "num_pages": num_pages,
      "doc_type": doc_type.value
    }
      
    docs.append(RawDocument(content=full_text, metadata=metadata, pages=page_offsets))
  
  return docs

def parse_documents(docs: list[RawDocument]) -> list[RawDocument]:
  parsed_docs = []

  for doc in docs:
    if doc.metadata["doc_type"] == DocumentType.SLIDES.value:
      slides = []

      for page in doc.pages:
        title, body = extract_slide_structure(page["text"])
        slides.append({
          "page": page["page"],
          "title": title,
          "content": body
        })

      doc.metadata["slides"] = slides
      parsed_docs.append(doc)

    else:
      section_regex = re.compile(
        r"(?m)^(\d+(?:\.\d+)*)\s{2,}([A-Z][A-Za-z\- ]{3,60})"
      )

      content = doc.content
      sections = []

      matches = list(section_regex.finditer(content))

      for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)

        sections.append({
          "id": m.group(1),
          "title": m.group(2).strip(),
          "start": start,
          "end": end,
          "content": content[start:end]
        })

      doc.metadata["sections"] = sections
      parsed_docs.append(doc)
  
  return parsed_docs