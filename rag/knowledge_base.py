import json
import os
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from rag.loader import load_documents
from rag.models import KnowledgeBase
from rag.parser import parse_documents
from rag.splitter import (
    split_documents,
    to_langchain_documents,
)
from rag.vectorstore import build_and_persist_vectorstore
from utils.helpers import generate_kb_id

# Below this average of extracted characters per page, a PDF is very
# likely scanned/image-based rather than text - fitz would silently
# return near-empty text for it rather than erroring, so this is the only
# signal we have without adding OCR.
LOW_TEXT_CHARS_PER_PAGE = 20

# ==========================================================
# Create
# ==========================================================

def detect_low_text_files(raw_documents) -> list[str]:
    return [
        doc.metadata.file_path
        for doc in raw_documents
        if doc.metadata.num_pages > 0
        and len(doc.content) / doc.metadata.num_pages < LOW_TEXT_CHARS_PER_PAGE
    ]

def create_and_save_knowledge_base(
    folder_path: Path,
    discipline_name: str,
) -> KnowledgeBase:

    kb_id = generate_kb_id()

    base_path = Path("data/knowledge_bases") / kb_id
    persist_dir = base_path / "chroma"

    base_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("[LOAD] Loading documents...")
    raw_documents = load_documents(folder_path)

    low_text_files = detect_low_text_files(raw_documents)

    if low_text_files:
        print(
            "[LOAD] WARNING: little to no extractable text in: "
            f"{', '.join(low_text_files)} (likely scanned/image-based PDFs)"
        )

    print("[PARSE] Parsing document structure...")
    parsed_documents = parse_documents(raw_documents)

    # --------------------------------------------------
    # Future step
    #
    # parsed_documents = enrich_documents(parsed_documents)
    # --------------------------------------------------

    print("[SPLIT] Creating semantic chunks...")
    chunks = split_documents(parsed_documents)

    print("[CONVERT] Converting to LangChain documents...")
    lc_documents = to_langchain_documents(chunks)

    print("[INDEX] Building vector store...")
    vectorstore = build_and_persist_vectorstore(
        documents=lc_documents,
        persist_dir=str(persist_dir),
    )

    metadata = {
        "id": kb_id,
        "name": discipline_name,
        "created_at": datetime.now().isoformat(),  # noqa: DTZ005
        "embedding_model": os.getenv(
            "EMBED_MODEL",
            "text-embedding-3-large",
        ),
        "documents": len(raw_documents),
        "sections": sum(
            len(doc.sections)
            for doc in parsed_documents
        ),
        "chunks": len(lc_documents),
        "parser_version": "1.0",
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "low_text_files": low_text_files,
    }

    with open(
        base_path / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Knowledge base saved to {persist_dir}")

    return KnowledgeBase(
        name=discipline_name,
        retriever=vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
                "lambda_mult": 0.7,
            },
        ),
        stats={
            "documents": metadata["documents"],
            "sections": metadata["sections"],
            "chunks": metadata["chunks"],
            "low_text_files": low_text_files,
        },
    )


# ==========================================================
# Load
# ==========================================================

def load_knowledge_base(
    discipline_name: str,
):

    base_path = Path("data/knowledge_bases")

    for kb_dir in base_path.iterdir():

        metadata_path = kb_dir / "metadata.json"

        if not metadata_path.exists():
            continue

        with open(
            metadata_path,
            encoding="utf-8",
        ) as f:

            metadata = json.load(f)

        if metadata["name"] != discipline_name:
            continue

        persist_dir = kb_dir / "chroma"

        embeddings = OpenAIEmbeddings(
            model=metadata.get(
                "embedding_model",
                os.getenv(
                    "EMBED_MODEL",
                    "text-embedding-3-large",
                ),
            )
        )

        vectorstore = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )

        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
                "lambda_mult": 0.7,
            },
        )

    raise ValueError(
        f"No knowledge base found for discipline {discipline_name!r}"
    )


# ==========================================================
# List
# ==========================================================

def list_knowledge_bases():

    base_dir = Path("data/knowledge_bases")

    knowledge_bases = []

    for kb_folder in base_dir.iterdir():

        metadata_file = kb_folder / "metadata.json"

        if metadata_file.exists():

            with open(
                metadata_file,
                encoding="utf-8",
            ) as f:

                knowledge_bases.append(
                    json.load(f)
                )

    return knowledge_bases