import re

from rag.models import (
    DocumentType,
    ParsedDocument,
    RawDocument,
    Section,
    SemanticBlock,
    SemanticBlockType,
)

from utils.helpers import (
    extract_slide_structure,
    offset_to_page
  )

def parse_documents(
    docs: list[RawDocument],
) -> list[ParsedDocument]:

    parsed = []

    for doc in docs:
        parsed.append(parse_document(doc))

    return parsed

def parse_document(
    doc: RawDocument,
) -> ParsedDocument:

    if doc.metadata.doc_type == DocumentType.SLIDES:
        return parse_slides(doc)

    return parse_pdf(doc)

# ==== PDFs ====

SECTION_REGEX = re.compile(
    r"(?m)^(\d+(?:\.\d+)*)\s{2,}([A-Z][A-Za-z\- ]{3,60})"
)

def parse_pdf(
    doc: RawDocument,
) -> ParsedDocument:

    title = extract_document_title(doc)

    sections = extract_sections(doc)

    return ParsedDocument(
        metadata=doc.metadata,
        title=title,
        sections=sections,
    )

def extract_document_title(
    doc: RawDocument,
) -> str | None:
    """
    Very simple heuristic.
    """

    if not doc.pages:
        return None

    lines = [
        line.strip()
        for line in doc.pages[0].text.splitlines()
        if line.strip()
    ]

    if not lines:
        return None

    return lines[0]

def extract_sections(
    doc: RawDocument,
) -> list[Section]:

    content = doc.content

    matches = list(SECTION_REGEX.finditer(content))

    if not matches:
        return [
            Section(
                id=None,
                title="Document",
                level=1,
                page_start=1,
                page_end=doc.metadata.num_pages,
                start_offset=0,
                end_offset=len(content),
                content=content,
                blocks=create_semantic_blocks(
                    content,
                    doc,
                    0,
                ),
            )
        ]

    sections = []

    for i, match in enumerate(matches):

        start = match.start()

        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(content)
        )

        page_start = offset_to_page(
            start,
            doc,
        )

        page_end = offset_to_page(
            end,
            doc,
        )

        section_content = content[start:end]

        sections.append(
            Section(
                id=match.group(1),
                title=match.group(2).strip(),
                level=match.group(1).count(".") + 1,
                page_start=page_start,
                page_end=page_end,
                start_offset=start,
                end_offset=end,
                content=section_content,
                blocks=create_semantic_blocks(
                    section_content,
                    doc,
                    start,
                ),
            )
        )

    return sections

# ==== SLIDES ====

def parse_slides(
    doc: RawDocument,
) -> ParsedDocument:

    sections = []

    for page in doc.pages:

        title, body = extract_slide_structure(page.text)

        blocks = create_semantic_blocks(
            body,
            doc,
            page.start_offset,
        )

        sections.append(
            Section(
                id=str(page.number),
                title=title,
                level=1,
                page_start=page.number,
                page_end=page.number,
                start_offset=page.start_offset,
                end_offset=page.start_offset + len(page.text),
                content=body,
                blocks=blocks,
            )
        )

    return ParsedDocument(
        metadata=doc.metadata,
        title=extract_document_title(doc),
        sections=sections,
    )

# ==== SEMANTIC BLOCKS ====

def create_semantic_blocks(
    text: str,
    doc: RawDocument,
    base_offset: int,
) -> list[SemanticBlock]:
    """
    First version.

    One semantic block per paragraph.
    """

    blocks = []

    cursor = base_offset

    for paragraph in text.split("\n\n"):

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        start = cursor
        end = start + len(paragraph)

        blocks.append(
            SemanticBlock(
                type=SemanticBlockType.PARAGRAPH,
                content=paragraph,
                page_start=offset_to_page(start, doc),
                page_end=offset_to_page(end, doc),
                start_offset=start,
                end_offset=end,
            )
        )

        cursor = end + 2

    return blocks