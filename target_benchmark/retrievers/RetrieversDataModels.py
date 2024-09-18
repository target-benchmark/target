from typing import List, Tuple, Union

from pydantic import BaseModel, Field


class RetrievalResultDataModel(BaseModel):
    dataset_name: str
    query_id: Union[int, str]
    retrieval_results: List[Tuple] = Field(
        default=[], description="retrieved table, a tuple of (database id, table id)"
    )
