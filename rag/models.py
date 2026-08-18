from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(Enum):
    SLIDES = "slides"
    PDF = "pdf"
    UNKNOWN = "unknown"

# ==== LOADER ====

class Page(BaseModel):
    number: int
    text: str
    start_offset: int

class RawDocumentMetadata(BaseModel):
    source: str
    file_path: str
    num_pages: int
    doc_type: DocumentType

class RawDocument(BaseModel):
    content: str
    pages: list[Page]
    metadata: RawDocumentMetadata

# ==== PARSER ====

class SemanticBlockType(str, Enum):
    PARAGRAPH = "paragraph"
    DEFINITION = "definition"
    EXAMPLE = "example"
    EXERCISE = "exercise"
    NOTE = "note"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    UNKNOWN = "unknown"

class SemanticBlock(BaseModel):
    type: SemanticBlockType = SemanticBlockType.PARAGRAPH

    content: str

    page_start: int
    page_end: int

    start_offset: int
    end_offset: int

class Section(BaseModel):
    id: str | None = None

    title: str

    level: int = 1

    page_start: int
    page_end: int

    start_offset: int
    end_offset: int

    content: str

    blocks: list[SemanticBlock] = Field(default_factory=list)

    chapter_number: str | None = None
    chapter_title: str | None = None

class ParsedDocument(BaseModel):
    metadata: RawDocumentMetadata

    title: str | None = None

    sections: list[Section] = Field(default_factory=list)

# ==== SPLITTER ====

class ChunkMetadata(BaseModel):
    source: str
    file_path: str

    doc_type: str

    page_start: int
    page_end: int

    section_id: str | None = None
    section_title: str | None = None

    chapter_number: str | None = None
    chapter_title: str | None = None

    chunk_index: int
    total_chunks: int

    start_offset: int
    end_offset: int

class DocumentChunk(BaseModel):
    id: str

    content: str

    metadata: ChunkMetadata

# ==== RUNTIME ====

class KnowledgeBase(BaseModel):
    name: str
    retriever: Any
    stats: dict[str, int] | None = None

class KnowledgeBaseMetadata(BaseModel):
    id: str
    name: str
    created_at: datetime
    embedding_model: str
    parser_version: str
    chunk_size: int
    chunk_overlap: int
    documents: int
    sections: int
    chunks: int