from pydantic import BaseModel, Field


class GradeDocument(BaseModel):
    """
    Pedagogical relevance assessment for a retrieved document chunk.
    """

    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How useful this chunk is for answering the student's question "
            "and supporting learning. "
            "0.0 = irrelevant, 1.0 = highly relevant."
        )
    )

    reason: str = Field(
        description=(
            "Brief explanation of why the chunk received this score."
        )
    )