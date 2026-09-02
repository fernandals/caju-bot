from pathlib import Path

import fitz

from rag.models import (
    Page,
    RawDocument,
    RawDocumentMetadata,
)
from utils.helpers import detect_pdf_type


def load_documents(
    folder_path: Path,
) -> tuple[list[RawDocument], list[dict]]:
    """
    Loads all PDF files from a folder into RawDocument objects.

    This stage performs no semantic parsing. It only extracts raw text,
    page boundaries and metadata.

    Returns the documents that loaded plus a list of {"file", "error"}
    entries for PDFs that could not be read (encrypted, corrupt, empty):
    one unreadable file shouldn't abort the whole knowledge base build.
    """

    print(f"Loading PDFs from '{folder_path}'...")

    documents: list[RawDocument] = []
    failed: list[dict] = []

    for file_path in sorted(folder_path.glob("**/*.pdf")):

        print(f"Loading {file_path.name}...")

        try:
            documents.append(load_pdf(file_path))
        except Exception as exc:  # noqa: BLE001 - report the file, don't abort
            print(f"  [SKIP] {file_path.name}: {exc}")
            failed.append({"file": file_path.name, "error": str(exc)})

    return documents, failed


def load_pdf(file_path: Path) -> RawDocument:

    pdf = fitz.open(file_path)

    if pdf.page_count == 0:
        raise ValueError("the PDF has no pages")

    full_text = ""
    pages: list[Page] = []

    for page_number, page in enumerate(pdf, start=1): # type: ignore

        text = page.get_text("text")

        pages.append(
            Page(
                number=page_number,
                text=text,
                start_offset=len(full_text),
            )
        )

        full_text += text + "\n"

    metadata = RawDocumentMetadata(
        source=str(file_path),
        file_path=file_path.name,
        num_pages=pdf.page_count,
        doc_type=detect_pdf_type(pdf),
    )

    return RawDocument(
        content=full_text,
        pages=pages,
        metadata=metadata,
    )
