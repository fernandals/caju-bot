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
    detect_chapter_from_filename,
    detect_chapter_header,
    extract_slide_structure,
    offset_to_page,
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
    # [ \t]{2,} (not \s{2,}) is deliberate: \s also matches newlines, which
    # let a bare page-number footer line bleed into the next line's
    # capitalized text and register as a false section/chapter heading
    # (e.g. a footer "2" immediately followed by "Note that..." on the next
    # line was being read as section "2  Note that...").
    r"(?m)^(\d+(?:\.\d+)*)[ \t]{2,}([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ0-9\-,:() ]{3,60})"
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
        sections = [
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

        apply_chapter_fallback(sections, doc)

        return sections

    sections = []

    current_chapter_number = None
    current_chapter_title = None

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

        level = match.group(1).count(".") + 1

        # A dot-less numbered heading (e.g. "12  Introduction") marks the
        # start of a new chapter; deeper headings inherit it. We do not scan
        # arbitrary body text for "Chapter N" mentions, since chapters are
        # often referenced in passing elsewhere ("see Chapter 9 for...").
        if level == 1:
            current_chapter_number = match.group(1)

            _, header_title = detect_chapter_header(
                page_text_for(doc, page_start)
            )
            current_chapter_title = header_title

        section_content = content[start:end]

        sections.append(
            Section(
                id=match.group(1),
                title=match.group(2).strip(),
                level=level,
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
                chapter_number=current_chapter_number,
                chapter_title=current_chapter_title,
            )
        )

    apply_chapter_fallback(sections, doc)

    return sections


def page_text_for(
    doc: RawDocument,
    page_number: int,
) -> str:
    index = page_number - 1

    if 0 <= index < len(doc.pages):
        return doc.pages[index].text

    return ""


def apply_chapter_fallback(
    sections: list[Section],
    doc: RawDocument,
) -> None:
    """
    When no chapter number could be detected from the document's own
    structure, fall back to the filename convention (e.g. 'Chapter12.pdf').
    """

    if any(section.chapter_number for section in sections):
        return

    fallback_number = detect_chapter_from_filename(doc.metadata.file_path)

    if not fallback_number:
        return

    for section in sections:
        section.chapter_number = fallback_number

# ==== SLIDES ====

def parse_slides(
    doc: RawDocument,
) -> ParsedDocument:

    sections = []

    current_chapter_number = None
    current_chapter_title = None

    for page in doc.pages:

        title, body = extract_slide_structure(page.text)

        # Slide decks typically repeat a "Chapter N. Title" running header
        # on every content page. Carry the last detected chapter forward
        # for pages where it's absent (e.g. the deck's cover page).
        header_number, header_title = detect_chapter_header(page.text)

        if header_number:
            current_chapter_number = header_number
            current_chapter_title = header_title

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
                chapter_number=current_chapter_number,
                chapter_title=current_chapter_title,
            )
        )

    apply_chapter_fallback(sections, doc)

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