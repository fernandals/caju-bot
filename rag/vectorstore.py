import os

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def build_and_persist_vectorstore(chunks, persist_dir: str):
  embeddings_model = os.getenv("EMBED_MODEL", "text-embedding-3-large")
  embeddings = OpenAIEmbeddings(model=embeddings_model)

  vectorstore = Chroma.from_documents(
    documents=[chunk for chunk_list in chunks for chunk in chunk_list],
    embedding=embeddings,
    persist_directory=persist_dir
  )

  return vectorstore