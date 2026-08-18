from rag.models import ChunkMetadata, DocumentType


def format_citation(metadata: ChunkMetadata) -> str:
    """
    Builds a precise, deterministic citation string from chunk metadata:
    file, chapter (when known), and section/slide + pages.

    This intentionally does not involve an LLM: citations are factual
    pointers back into the professor's material and must be exact.
    """

    parts = [metadata.file_path]

    if metadata.chapter_number:
        chapter_label = f"Chapter {metadata.chapter_number}"

        if metadata.chapter_title:
            chapter_label = f"{chapter_label} – {metadata.chapter_title}"

        parts.append(chapter_label)

    if metadata.doc_type == DocumentType.SLIDES.value:
        location_label = _slide_location(metadata)
    else:
        location_label = _section_location(metadata)

    if location_label:
        parts.append(location_label)

    return f"[{', '.join(parts)}]"


def _section_location(metadata: ChunkMetadata) -> str | None:
    pages = _page_range(metadata)

    if metadata.section_title:
        label = f"Section {metadata.section_id}" if metadata.section_id else "Section"
        label = f"{label} – {metadata.section_title}"

        return f"{label}, Pages {pages}" if pages else label

    return f"Pages {pages}" if pages else None


def _slide_location(metadata: ChunkMetadata) -> str | None:
    label = (
        f"Slide {metadata.page_start}"
        if metadata.page_start == metadata.page_end
        else f"Slides {metadata.page_start}-{metadata.page_end}"
    )

    if metadata.section_title:
        return f"{label} – {metadata.section_title}"

    return label


def _page_range(metadata: ChunkMetadata) -> str:
    if metadata.page_start == metadata.page_end:
        return str(metadata.page_start)

    return f"{metadata.page_start}-{metadata.page_end}"
