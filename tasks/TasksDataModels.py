from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict
from typing import Dict, Optional


class EmbeddingStatisticsDataModel(BaseModel):
    embedding_creation_time: float = Field(
        ..., description="Total amount of time taken for the embedding to be created"
    )
    avg_embedding_creation_time: float = Field(
        ..., description="Average time taken to create embeddings"
    )
    embedding_size: float = Field(
        ...,
        description="Totla size of embeddings (all tables in dataset) in bytes, INACCURATE for custom retrievers.",
    )
    avg_embedding_size: float = Field(
        ...,
        description="Average size of an embedding in bytes, INACCURATE for custom retrievers.",
    )


class RetrievalPerformanceDataModel(BaseModel):
    k: int = Field(default=5, description="k value for top k metrics.")
    accuracy: float = Field(description="the accuracy of the retrieval")
    precision: float = Field(default=None, description="the precision of the retrieval")
    recall: float = Field(default=None, description="the recall of the retrieval")

    retrieval_time: float = Field(
        ...,
        description="total time took to complete all retrievals in seconds.",
    )
    avg_retrieval_time: float = Field(
        ..., description="avg time too for each retrieval in seconds."
    )


class DownstreamTaskPerformanceDataModel(BaseModel):
    task_name: str = Field(default=None, description="name of the downstream task")
    scores: Optional[Dict] = Field(
        default=None, description="all metrics with the metric name prepended."
    )


class FactVerificationTaskPerformanceDataModel(DownstreamTaskPerformanceDataModel):
    task_name: str = Field(
        default="Fact Verification Task",
        description="name of the downstream task",
    )


class TableQATaskPerformanceDataModel(DownstreamTaskPerformanceDataModel):
    task_name: str = Field(
        default="Table Question Answering Task",
        description="name of the downstream task",
    )


class Text2SQLTaskPerformanceDataModel(DownstreamTaskPerformanceDataModel):
    task_name: str = Field(
        default="Text to SQL Task",
        description="name of the downstream task",
    )


class TaskResultsDataModel(BaseModel):
    retrieval_performance: RetrievalPerformanceDataModel
    downstream_task_performance: DownstreamTaskPerformanceDataModel
    embedding_statistics: Optional[EmbeddingStatisticsDataModel] = Field(
        default=None, description="Stats on latency and embedding size."
    )
