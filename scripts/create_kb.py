from pathlib import Path
from rag.knowledge_base import create_and_save_knowledge_base, list_knowledge_bases

if __name__ == "__main__":
  folder = Path("/home/fernanda/home/langchain-rag/pdfs")
  discipline = "Software Architecture"

  create_and_save_knowledge_base(folder, discipline)

  kbs = list_knowledge_bases()
  for kb in kbs:
    print(f"Knowledge Base: {kb['name']} (ID: {kb['id']})")