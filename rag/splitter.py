from __future__ import annotations

import uuid

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.models import (
    ChunkMetadata,
    DocumentChunk,
    ParsedDocument,
    SemanticBlock,
)

# ==========================================================
# Configuration
# ==========================================================
#
# Both in TOKENS (not characters). Chunk packing below (split_section) and
# the oversized-block fallback splitter must agree on the same unit, or
# "1000" means two very different chunk sizes depending on which code path
# a given paragraph happens to hit.

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

_ENCODING_NAME = "cl100k_base"  # matches OpenAI embedding/chat tokenization
_encoding = tiktoken.get_encoding(_ENCODING_NAME)

_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name=_ENCODING_NAME,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)


def _token_length(text: str) -> int:
    return len(_encoding.encode(text))


# ==========================================================
# Public API
# ==========================================================

def split_documents(
    docs: list[ParsedDocument],
) -> list[list[DocumentChunk]]:
    """
    Converts ParsedDocuments into semantic chunks.

    Strategy

        ParsedDocument
            ↓
        Section
            ↓
        SemanticBlocks
            ↓
        Semantic Chunks
    """

    print("Creating semantic chunks...")

    all_documents: list[list[DocumentChunk]] = []

    for doc in docs:
        all_documents.append(split_document(doc))

    return all_documents


# ==========================================================
# Document
# ==========================================================

def split_document(
    doc: ParsedDocument,
) -> list[DocumentChunk]:

    chunks: list[DocumentChunk] = []

    for section in doc.sections:
        chunks.extend(
            split_section(
                document=doc,
                section=section,
            )
        )

    assign_chunk_indexes(chunks)

    return chunks


# ==========================================================
# Section
# ==========================================================

def split_section(
    document: ParsedDocument,
    section,
) -> list[DocumentChunk]:

    chunks: list[DocumentChunk] = []

    current_blocks: list[SemanticBlock] = []
    current_size = 0

    for block in section.blocks:

        block_size = _token_length(block.content)

        # Very large paragraph
        if block_size > CHUNK_SIZE:

            if current_blocks:
                chunks.append(
                    build_chunk(
                        document,
                        section,
                        current_blocks,
                    )
                )

                current_blocks = []
                current_size = 0

            chunks.extend(
                split_large_block(
                    document,
                    section,
                    block,
                )
            )

            continue

        # Chunk full
        if current_size + block_size > CHUNK_SIZE:

            chunks.append(
                build_chunk(
                    document,
                    section,
                    current_blocks,
                )
            )

            current_blocks = []
            current_size = 0

        current_blocks.append(block)
        current_size += block_size

    if current_blocks:
        chunks.append(
            build_chunk(
                document,
                section,
                current_blocks,
            )
        )

    return chunks


# ==========================================================
# Chunk builders
# ==========================================================

def build_chunk(
    document: ParsedDocument,
    section,
    blocks: list[SemanticBlock],
) -> DocumentChunk:

    page_start = blocks[0].page_start
    page_end = blocks[-1].page_end

    start_offset = blocks[0].start_offset
    end_offset = blocks[-1].end_offset

    body = "\n\n".join(
        block.content
        for block in blocks
    )

    header = build_chunk_header(
        document=document,
        section=section,
        page_start=page_start,
        page_end=page_end,
    )

    content = f"{header}\n\n{body}"

    metadata = ChunkMetadata(
        source=document.metadata.source,
        file_path=document.metadata.file_path,
        doc_type=document.metadata.doc_type.value,
        page_start=page_start,
        page_end=page_end,
        section_id=section.id,
        section_title=section.title,
        chapter_number=section.chapter_number,
        chapter_title=section.chapter_title,
        chunk_index=0,
        total_chunks=0,
        start_offset=start_offset,
        end_offset=end_offset,
    )

    return DocumentChunk(
        id=str(uuid.uuid4()),
        content=content,
        metadata=metadata,
    )


def build_chunk_header(
    document: ParsedDocument,
    section,
    page_start: int,
    page_end: int,
) -> str:

    title = document.title or document.metadata.file_path

    lines = [f"Document: {title}"]

    if section.chapter_number:
        chapter_label = section.chapter_number

        if section.chapter_title:
            chapter_label = f"{chapter_label} — {section.chapter_title}"

        lines.append(f"Chapter: {chapter_label}")

    lines.append(f"Section: {section.title}")
    lines.append(f"Pages: {page_start}-{page_end}")

    return "\n".join(lines)


# ==========================================================
# Large blocks
# ==========================================================

def split_large_block(
    document: ParsedDocument,
    section,
    block: SemanticBlock,
) -> list[DocumentChunk]:

    header = build_chunk_header(
        document=document,
        section=section,
        page_start=block.page_start,
        page_end=block.page_end,
    )

    lc_doc = Document(
        page_content=block.content,
    )

    pieces = _splitter.split_documents([lc_doc])

    chunks: list[DocumentChunk] = []

    for piece in pieces:

        metadata = ChunkMetadata(
            source=document.metadata.source,
            file_path=document.metadata.file_path,
            doc_type=document.metadata.doc_type.value,
            page_start=block.page_start,
            page_end=block.page_end,
            section_id=section.id,
            section_title=section.title,
            chapter_number=section.chapter_number,
            chapter_title=section.chapter_title,
            chunk_index=0,
            total_chunks=0,
            start_offset=block.start_offset,
            end_offset=block.end_offset,
        )

        chunks.append(
            DocumentChunk(
                id=str(uuid.uuid4()),
                content=f"{header}\n\n{piece.page_content}",
                metadata=metadata,
            )
        )

    return chunks


# ==========================================================
# Helpers
# ==========================================================

def assign_chunk_indexes(
    chunks: list[DocumentChunk],
):

    total = len(chunks)

    for index, chunk in enumerate(chunks, start=1):

        chunk.metadata.chunk_index = index
        chunk.metadata.total_chunks = total


# ==========================================================
# LangChain conversion
# ==========================================================

def to_langchain_documents(
    chunks: list[list[DocumentChunk]],
) -> list[Document]:
    """
    Converts semantic chunks into LangChain Documents.
    """

    documents: list[Document] = []

    for doc_chunks in chunks:

        for chunk in doc_chunks:

            documents.append(
                Document(
                    page_content=chunk.content,
                    metadata=chunk.metadata.model_dump(),
                )
            )

    return documents