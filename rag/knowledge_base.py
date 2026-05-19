from pathlib import Path
from rag.loader import load_documents, parse_documents
from rag.splitter import spliting_documents
from rag.vectorstore import build_and_persist_vectorstore
from rag.models import KnowledgeBase
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from utils.helpers import generate_kb_id
from datetime import datetime
import json

def create_and_save_knowledge_base(folder_path: str, discipline_name: str):
  kb_id = generate_kb_id()
  base_path = Path("data/knowledge_bases") / kb_id
  persist_dir = base_path / "chroma"

  base_path.mkdir(parents=True, exist_ok=True)

  print("[LOAD] Carregando documentos da disciplina ...")
  raw_docs = load_documents(folder_path)

  print("[PARSE] Processando documentos ...")
  parsed_docs = parse_documents(raw_docs)

  print("[SPLIT] Dividindo documentos em chunks ...")
  chunks = spliting_documents(parsed_docs)

  print("[VECTORSTORE] Construindo base de conhecimento ...")
  vectorstore = build_and_persist_vectorstore(chunks, str(persist_dir))

  metadata = {
    "id": kb_id,
    "name": discipline_name,
    "created_at": datetime.now().isoformat(),
    "embedding_model": os.getenv("EMBED_MODEL", "text-embedding-3-large"),
    "num_chunks": sum(len(c) for c in chunks)
  }

  with open(base_path / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

  print(f"Base salva em: {persist_dir}")

  return KnowledgeBase(
    name=discipline_name,
    retriever=vectorstore.as_retriever()
  )

def load_knowledge_base(discipline_name: str):
    base_path = Path("data/knowledge_bases")
    
    # find the kb folder that matches the discipline name
    for kb_dir in base_path.iterdir():
        metadata_path = kb_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        if metadata["name"] == discipline_name:
            persist_dir = kb_dir / "chroma"
            
            embeddings = OpenAIEmbeddings(
                model=metadata.get("embedding_model", os.getenv("EMBED_MODEL", "text-embedding-3-large"))
            )
            
            vectorstore = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=embeddings
            )
            
            return vectorstore.as_retriever(
              search_type="mmr",
              search_kwargs={
                "k": 5,
                "fetch_k": 20,
                "lambda_mult": 0.7
              }
            )
    
    raise ValueError(f"No knowledge base found for discipline: {discipline_name!r}")

def list_knowledge_bases() -> list[dict]:
  base_dir = Path("data/knowledge_bases")
  kbs = []

  for kb_folder in base_dir.iterdir():
    metadata_file = kb_folder / "metadata.json"
    if metadata_file.exists():
      with open(metadata_file) as f:
        kbs.append(json.load(f))

  return kbs