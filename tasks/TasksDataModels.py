from pydantic import BaseModel, Field
from typing import Dict


class RetrievalPerformanceDataModel(BaseModel):
    k: int = Field(default=5, description="k value for top k metrics.")
    accuracy: float = Field(description="the accuracy of the retrieval")
    precision: float = Field(default=None, description="the precision of the retrieval")
    recall: float = Field(default=None, description="the recall of the retrieval")


class DownstreamTaskPerformanceDataModel(BaseModel):
    task_name: str = Field(default=None, description="name of the downstream task")


class TableQATaskPerformanceDataModel(DownstreamTaskPerformanceDataModel):
    task_name: str = Field(
        default="Table Question Answering Task",
        description="name of the downstream task",
    )
    bert_score: Dict = Field(
        default=None,
        description="Calculated bert score, dictionary with fields precision, recall, f1, and hashcode of the model used for scoring.",
    )
    bleurt_score: Dict = Field(
        default=None,
        description="Calculated bleurt score. dictionary with one field, a list of scores",
    )
    sacrebleu_score: Dict = Field(
        default=None,
        description="Calculate sacreBleu score. dictionary with fields score, counts, totals, precisions, bp, sys_len, ref_len",
    )
    rouge_score: Dict = Field(
        default=None,
        description="Calculated rouge score. Contains keys rouge1, rouge2, rougeL, rougeLsum.",
    )
    meteor_score: Dict = Field(
        default=None, description="Calculated meteor score. Contains key meteor."
    )


class TaskResultsDataModel(BaseModel):
    retrieval_performance: RetrievalPerformanceDataModel
    downstream_task_performance: DownstreamTaskPerformanceDataModel
