from dataset_loaders.LoadersDataModels import (
    DatasetConfigDataModel,
    QueryForTasksDataModel,
)
from dataset_loaders.TargetDatasetConfig import *

from generators.AbsGenerator import AbsGenerator
from generators.GeneratorsDataModels import DownstreamGeneratedResultDataModel

from retrievers.RetrieversDataModels import RetrievalResultDataModel

from tasks.AbsTask import AbsTask
from tasks.TasksDataModels import DownstreamTaskPerformanceDataModel, TableQATaskPerformanceDataModel

import evaluate
from typing import List, Dict, Union



class QuestionAnsweringTask(AbsTask):

    AVAILABLE_METRICS = set("bertscore", "bleurt", "sacrebleu", "rouge", "meteor")
    def __init__(
        self,
        datasets_config: Dict[str, Dict[str, str]] = None,
        overwrite_default_datasets: bool = False,
        task_generator: AbsGenerator = None,
        lang: str = "en",
        metrics: Union[str, List[str]] = "bertscore",
        **kwargs,
    ):
        super().__init__(
            task_name=self.get_default_task_name(),
            datasets_config=datasets_config,
            overwrite_default_datasets=overwrite_default_datasets,
            task_generator=task_generator,
            **kwargs,
        )
        # set up the evaluator objects
        if isinstance(metrics, str):
            metrics = [metrics]
        for metric in metrics:
            if metric not in QuestionAnsweringTask.AVAILABLE_METRICS:
                raise ValueError(f"the metric {metric} is not one of the available metrics!")

        self.bert_score = evaluate.load("bertscore") if "bertscore" in metrics else None
        self.bleurt = evaluate.load("bleurt", module_type="metric") if "bleurt" in metrics else None
        self.sacre_bleu = evaluate.load("sacrebleu") if "sacrebleu" in metrics else None
        self.rouge = evaluate.load("rouge") if "rouge" in metrics else None
        self.meteor = evaluate.load("meteor") if "meteor" in metrics else None
        self.language = lang
        self.pred_answers = []
        self.ref_answers = []

    @classmethod
    def get_default_task_name(cls) -> str:
        return "Table Question Answering Task"
    @classmethod
    def get_available_metrics(cls) -> str:
        return str(cls.AVAILABLE_METRICS)


    def _get_default_dataset_config(self) -> Dict[str, DatasetConfigDataModel]:
        """
        Returns the default dataset config for the class. MUST be implemented by any inherited task class.
        """
        # TODO: add more things here. this is for testing. carl note 4/24
        return {
            # this is for testing!!
            DEFAULT_DUMMY_DATASET_CONFIG.dataset_name: DEFAULT_DUMMY_DATASET_CONFIG,
        }

    def _get_downstream_task_results(
        self,
        query_batch: List[QueryForTasksDataModel],
        retrieval_results: List[RetrievalResultDataModel],
        dataset_name: str,
    ) -> List[DownstreamGeneratedResultDataModel]:
        """
        currently just markdown reps of table strings
        All downstreams tasks should fill out this method. ideally uses the retrieval results to generate the downstream answer, and return the performance of the downstream generation.
        """
        return [
            DownstreamGeneratedResultDataModel(
                dataset_name=dataset_name,
                query_id=query.query_id,
                generated_results=self.task_generator.generate(
                    table_str="\n".join(
                        table_str for table_str in result.retrieved_tables
                    ),
                    query=query.query,
                ),
            )
            for query, result in zip(query_batch, retrieval_results)
        ]

    def _update_downstream_task_results(
        self,
        query_batch: List[QueryForTasksDataModel],
        downstream_answers: List[DownstreamGeneratedResultDataModel],
    ) -> None:
        """
        Update any values you keep track of for the downstream tasks.
        """
        self.pred_answers.extend([downstream_answer.generated_results for downstream_answer in downstream_answers])
        self.ref_answers.extend([query.answer for query in query_batch])

    def _calculate_downstream_task_metrics(
        self, **kwargs
    ) -> TableQATaskPerformanceDataModel:
        """
        Calculate downstream task metrics for the question answering task.
        """
        result = TableQATaskPerformanceDataModel(
            bert_score=self.bert_score.compute(predictions=self.pred_answers, references=self.ref_answers, lang="en") if self.bert_score else None,
            bleurt_score=self.bleurt.compute(predictions=self.pred_answers, references=self.ref_answers) if self.bleurt else None,
            sacrebleu_score=self.sacre_bleu.compute(predictions=self.pred_answers, references=self.ref_answers) if self.sacre_bleu else None,
            rouge_score=self.rouge.compute(predictions=self.pred_answers, references=self.ref_answers) if self.rouge else None,
            meteor_score=self.meteor.compute(predictions=self.pred_answers, references=self.ref_answers) if self.meteor else None,
        )
        
        self.pred_answers = []
        self.ref_answers = []
        return result
