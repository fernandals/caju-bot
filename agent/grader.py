from pydantic import BaseModel, Field

class GradeDocument(BaseModel):
    """Binary relevance score for a retrieved document chunk."""
    relevant: bool = Field(description="True if the chunk is relevant to the question, False otherwise.")
    reason: str = Field(description="Brief justification for the score.")