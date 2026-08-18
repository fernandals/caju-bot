from pathlib import Path

import fitz

from rag.models import (
    Page,
    RawDocument,
    RawDocumentMetadata,
)
from utils.helpers import detect_pdf_type


def load_documents(folder_path: Path) -> list[RawDocument]:
    """
    Loads all PDF files from a folder into RawDocument objects.

    This stage performs no semantic parsing.
    It only extracts raw text, page boundaries and metadata.
    """

    print(f"Loading PDFs from '{folder_path}'...")

    documents: list[RawDocument] = []

    for file_path in folder_path.glob("**/*.pdf"):

        print(f"Loading {file_path.name}...")

        pdf = fitz.open(file_path)

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
            doc_type=detect_pdf_type(pdf[0]),
        )

        documents.append(
            RawDocument(
                content=full_text,
                pages=pages,
                metadata=metadata,
            )
        )

    return documents
