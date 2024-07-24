from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict
from typing import Dict, Optional
from retrievers.RetrieversDataModels import EmbeddingStatisticsDataModel


class RetrievalPerformanceDataModel(BaseModel):
    k: int = Field(default=5, description="k value for top k metrics.")
    accuracy: float = Field(description="the accuracy of the retrieval")
    precision: float = Field(default=None, description="the precision of the retrieval")
    recall: float = Field(default=None, description="the recall of the retrieval")

    retrieval_time: Optional[float] = Field(
        default=-1.0,
        description="total time took to complete all retrievals in seconds.",
    )
    avg_retrieval_time: Optional[float] = Field(
        default=-1.0, description="avg time too for each retrieval in seconds."
    )

    @field_validator("retrieval_time", "avg_retrieval_time", mode="before")
    @classmethod
    def round_float(cls, value: float):
        return round(value, 2)


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
    embedding_statistics: Optional[EmbeddingStatisticsDataModel]
