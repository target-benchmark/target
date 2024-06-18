from pydantic import BaseModel, Field
from typing import List, Tuple


class RetrievalResultDataModel(BaseModel):
    dataset_name: str
    query_id: int
    retrieval_results: List[Tuple] = Field(
        default=[], description="retrieved table, a tuple of (database id, table id)"
    )
    retrieved_tables: List[str] = Field(
        default=[],
        description="a list of the string representation of the tables retrieved",
    )
