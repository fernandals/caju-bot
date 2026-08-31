import json
import os
import shutil
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

    # Keep the original PDFs next to the index so the student app can link
    # each citation straight to its source page. The whole
    # data/knowledge_bases/<id>/ tree (this included) is baked into the
    # course image, so nothing else needs to ship the files.
    sources_dir = base_path / "sources"
    sources_dir.mkdir(exist_ok=True)

    for pdf in folder_path.glob("**/*.pdf"):
        shutil.copy2(pdf, sources_dir / pdf.name)

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


# ==========================================================
# Locate
# ==========================================================

def knowledge_base_dir(discipline_name: str) -> Path | None:
    """Path to the KB folder for a discipline, or None if not found."""
    base_path = Path("data/knowledge_bases")

    if not base_path.is_dir():
        return None

    for kb_dir in base_path.iterdir():

        metadata_path = kb_dir / "metadata.json"

        if not metadata_path.exists():
            continue

        with open(metadata_path, encoding="utf-8") as f:
            if json.load(f).get("name") == discipline_name:
                return kb_dir

    return None


def resolve_source_pdf(discipline_name: str, file_name: str) -> Path | None:
    """
    Absolute path to a citation's source PDF, if it was saved alongside
    the KB (see create_and_save_knowledge_base). Returns None for KBs
    built before sources were persisted.
    """
    kb_dir = knowledge_base_dir(discipline_name)

    if kb_dir is None:
        return None

    candidate = kb_dir / "sources" / Path(file_name).name

    return candidate if candidate.is_file() else None


# ==========================================================
# Describe (overviews from the persisted chunk metadata)
# ==========================================================

def _kb_chunk_metadatas(discipline_name: str) -> list[dict]:
    """
    Every chunk's metadata dict from a course's persisted vector store -
    no embeddings or model calls. Shared by the overview helpers below.
    """
    import chromadb

    kb_dir = knowledge_base_dir(discipline_name)

    if kb_dir is None:
        return []

    client = chromadb.PersistentClient(path=str(kb_dir / "chroma"))
    collections = client.list_collections()

    if not collections:
        return []

    records = client.get_collection(collections[0].name).get(
        include=["metadatas"]
    )

    return records.get("metadatas") or []


def describe_course_materials(discipline_name: str) -> list[dict]:
    """
    Best-effort list of the source materials indexed for a course, read
    straight from the persisted vector store's chunk metadata - nothing
    hardcoded about the course.

    Returns one entry per source file: {"file", "chapter", "title"},
    sorted by chapter. "chapter"/"title" are None when the ingestion
    didn't detect them. Used to populate the student-facing readme.
    """
    by_file: dict[str, dict] = {}

    for meta in _kb_chunk_metadatas(discipline_name):

        file_path = meta.get("file_path")

        if not file_path:
            continue

        entry = by_file.setdefault(
            file_path,
            {"file": file_path, "chapter": None, "_titles": set()},
        )

        chapter = meta.get("chapter_number")

        if chapter is not None:
            try:
                entry["chapter"] = int(chapter)
            except (TypeError, ValueError):
                pass

        title = (meta.get("chapter_title") or "").strip()

        if title:
            entry["_titles"].add(title)

    materials = []

    for entry in by_file.values():
        titles = entry.pop("_titles")
        entry["title"] = titles.pop() if len(titles) == 1 else None
        materials.append(entry)

    materials.sort(
        key=lambda m: (m["chapter"] is None, m["chapter"] or 0, m["file"])
    )

    return materials


def list_material_sections(discipline_name: str) -> list[dict]:
    """
    Every distinct (file, section) present in the KB, so the metrics
    panel can spot which parts of the material student questions never
    reach. Returns {"file", "section_id", "section_title", "chunks"}.
    """
    by_section: dict[tuple, dict] = {}

    for meta in _kb_chunk_metadatas(discipline_name):

        file_path = meta.get("file_path")

        if not file_path:
            continue

        section_id = meta.get("section_id")
        section_title = meta.get("section_title")
        key = (file_path, section_id, section_title)

        entry = by_section.setdefault(
            key,
            {
                "file": file_path,
                "section_id": section_id,
                "section_title": section_title,
                "chunks": 0,
            },
        )
        entry["chunks"] += 1

    return sorted(
        by_section.values(),
        key=lambda s: (s["file"], str(s["section_id"] or ""), s["section_title"] or ""),
    )