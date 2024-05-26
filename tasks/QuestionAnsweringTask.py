from dataset_loaders.LoadersDataModels import (
    DatasetConfigDataModel,
    QueryForTasksDataModel,
)
from dataset_loaders.TargetDatasetConfig import *

from generators.AbsGenerator import AbsGenerator
from generators.GeneratorsDataModels import DownstreamGeneratedResultDataModel

from retrievers.RetrieversDataModels import RetrievalResultDataModel

from tasks.AbsTask import AbsTask
from tasks.TasksDataModels import (
    TableQATaskPerformanceDataModel,
)

import evaluate
from typing import List, Dict, Union


class QuestionAnsweringTask(AbsTask):

    AVAILABLE_METRICS = set(
        ["bertscore", "bleu", "bleurt", "sacrebleu", "rouge", "meteor"]
    )
    DEFAULT_METRICS = set(["bleu", "sacrebleu", "rouge"])

    def __init__(
        self,
        datasets_config: Dict[str, Dict[str, str]] = None,
        overwrite_default_datasets: bool = False,
        task_generator: AbsGenerator = None,
        lang: str = "en",
        metrics: Union[str, List[str]] = list(DEFAULT_METRICS),
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

        self.evals = {}
        for metric in metrics:
            if metric not in QuestionAnsweringTask.AVAILABLE_METRICS:
                raise ValueError(
                    f"the metric {metric} is not one of the available metrics!"
                )
            self.evals[metric] = evaluate.load(metric)

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

    def _update_downstream_task_metrics(
        self,
        query_batch: List[QueryForTasksDataModel],
        downstream_results: List[DownstreamGeneratedResultDataModel],
    ) -> None:
        """
        Update any values you keep track of for the downstream tasks.
        """
        self.pred_answers.extend(
            [
                downstream_answer.generated_results
                for downstream_answer in downstream_results
            ]
        )
        self.ref_answers.extend([query.answer for query in query_batch])

    def _calculate_downstream_task_performance(
        self, **kwargs
    ) -> TableQATaskPerformanceDataModel:
        """
        Calculate downstream task metrics for the question answering task.
        """
        scores = {}
        for metric_name, evaluator in self.evals.items():
            calculated_result = None
            if metric_name == "bertscore":
                calculated_result = evaluator.compute(
                    predictions=self.pred_answers,
                    references=self.ref_answers,
                    lang="en",
                )
            else:
                calculated_result = evaluator.compute(
                    predictions=self.pred_answers, references=self.ref_answers
                )
            scores[metric_name] = calculated_result

        result = TableQATaskPerformanceDataModel(scores=scores)

        self.pred_answers = []
        self.ref_answers = []
        return result
