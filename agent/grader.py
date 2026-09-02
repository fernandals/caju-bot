from pydantic import BaseModel, Field


class ChunkAssessment(BaseModel):
    """
    Combined relevance grade + evidence extraction for one retrieved
    chunk, produced in a single LLM pass (see
    agent/nodes.py::assess_documents). Replaces the old two-step
    GradeDocument-then-extract flow.
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

    evidence: list[str] = Field(
        default_factory=list,
        description=(
            "Atomic factual statements directly supported by this chunk, "
            "for a later generation step. Empty when the chunk is "
            "irrelevant to the student's question."
        )
    )
