from langchain_core.documents import Document

from rag.models import ChunkMetadata


SEPARATOR = "=" * 60


def build_generation_context(
    retrieved_docs: list[Document],
) -> str:
    """
    Converts retrieved LangChain documents into a structured context
    optimized for answer generation.
    """

    if not retrieved_docs:
        return ""

    blocks = []

    for index, doc in enumerate(retrieved_docs, start=1):

        metadata = ChunkMetadata.model_validate(doc.metadata)

        block = f"""
{SEPARATOR}

REFERENCE: DOC_{index}

Section:
{metadata.section_title or "N/A"}

Pages:
{metadata.page_start}-{metadata.page_end}

Content:

{doc.page_content}
""".strip()

        blocks.append(block)

    return "\n\n".join(blocks)