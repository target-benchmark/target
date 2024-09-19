from typing import Dict, Optional

from pydantic import BaseModel, Field


class EmbeddingStatisticsDataModel(BaseModel):
    embedding_creation_duration_process: float = Field(
        ...,
        description="Total amount of time taken for the embedding to be created, measured by process time.",
    )
    avg_embedding_creation_duration_process: float = Field(
        ...,
        description="Average time taken to create embeddings, measured by process time.",
    )
    embedding_creation_duration_wall_clock: float = Field(
        ...,
        description="Total amount of time taken for the embedding to be created, measured by wall clock time.",
    )
    avg_embedding_creation_duration_wall_clock: float = Field(
        ...,
        description="Average time taken to create embeddings, measured by wall clock time.",
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

    retrieval_duration_process: float = Field(
        ...,
        description="total time took to complete all retrievals in seconds, measured by process time.",
    )
    avg_retrieval_duration_process: float = Field(
        ...,
        description="avg time too for each retrieval in seconds, measured by process time.",
    )
    retrieval_duration_wall_clock: float = Field(
        ...,
        description="total time took to complete all retrievals in seconds, measured by wall clock time.",
    )
    avg_retrieval_duration_wall_clock: float = Field(
        ...,
        description="avg time too for each retrieval in seconds, measured by wall clock time.",
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
