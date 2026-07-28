import os

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def build_and_persist_vectorstore(
    documents: list[Document],
    persist_dir: str,
) -> Chroma:

    embedding_model = os.getenv(
        "EMBED_MODEL",
        "text-embedding-3-large",
    )

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
    )

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
    )