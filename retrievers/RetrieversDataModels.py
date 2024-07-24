from pydantic import BaseModel, Field
from typing import List, Tuple, Union


class RetrievalResultDataModel(BaseModel):
    dataset_name: str
    query_id: Union[int, str]
    retrieval_results: List[Tuple] = Field(
        default=[], description="retrieved table, a tuple of (database id, table id)"
    )
    retrieved_tables: List[str] = Field(
        default=[],
        description="a list of the string representation of the tables retrieved",
    )


class EmbeddingStatisticsDataModel(BaseModel):
    embedding_creation_time: float
    avg_embedding_creation_time: float
    embedding_storage_usage: float
    avg_embedding_storage_usage: float


class RetrievalStatisticsDataModel(BaseModel):
    retrieval_time: float
    avg_retrieval_time: float
