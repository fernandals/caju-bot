import os

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

EMBED_BATCH_SIZE = 200


def build_and_persist_vectorstore(
    documents: list[Document],
    persist_dir: str,
) -> Chroma:

    embedding_model = os.getenv(
        "EMBED_MODEL",
        "text-embedding-3-large",
    )

    # The API caps one embeddings request at 300k tokens; LangChain's
    # default batch of 1000 chunks broke that on a course with a whole
    # textbook (~660 tokens per chunk). 200 chunks stay well under it.
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        chunk_size=EMBED_BATCH_SIZE,
    )

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
    )