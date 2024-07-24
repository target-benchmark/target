from pydantic import BaseModel, Field, field_validator
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
    
    @field_validator("embedding_creation_time", "avg_embedding_creation_time", "embedding_storage_usage", "avg_embedding_storage_usage")
    @classmethod
    def round_float(cls, value: float):
        return round(value, 2)


class RetrievalStatisticsDataModel(BaseModel):
    retrieval_time: float
    avg_retrieval_time: float
    @field_validator("retrieval_time", "avg_retrieval_time")
    @classmethod
    def round_float(cls, value: float):
        return round(value, 2)