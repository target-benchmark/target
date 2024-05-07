from pydantic import BaseModel, Field
from typing import List


class RetrievalResultDataModel(BaseModel):
    dataset_name: str
    query_id: int
    retrieval_results: List[str]
    retrieved_tables: List[str] = Field(
        default=[],
        description="a list of the string representation of the tables retrieved",
    )
