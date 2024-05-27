from dataset_loaders.LoadersDataModels import (
    DatasetConfigDataModel,
    QueryForTasksDataModel,
)
from dataset_loaders.TargetDatasetConfig import *

from generators import DefaultGenerator
from generators.AbsGenerator import AbsGenerator
from generators.GeneratorsDataModels import DownstreamGeneratedResultDataModel
from generators.GeneratorPrompts import DEFAULT_TEXT2SQL_USER_PROMPT

from retrievers.RetrieversDataModels import RetrievalResultDataModel

from tasks.AbsTask import AbsTask
from tasks.TasksDataModels import (
    Text2SQLTaskPerformanceDataModel,
)

import evaluate
from typing import List, Dict, Union


class Text2SQLTask(AbsTask):

    AVAILABLE_METRICS = set(["execution_accuracy", "query_match"])
    DEFAULT_METRICS = set(["execution_accuracy", "query_match"])

    def __init__(
        self,
        datasets_config: Dict[str, Dict[str, str]] = None,
        overwrite_default_datasets: bool = False,
        task_generator: AbsGenerator = None,
        metrics: Union[str, List[str]] = list(DEFAULT_METRICS),
        **kwargs,
    ):
        if task_generator == None:
            task_generator = DefaultGenerator(
                user_message=DEFAULT_TEXT2SQL_USER_PROMPT
            )        
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

        self.evals = ""
        for metric in metrics:
            if metric not in Text2SQLTask.AVAILABLE_METRICS:
                raise ValueError(
                    f"the metric {metric} is not one of the available metrics!"
                )
        if "execution_accuracy" in metrics and "query_match" in metrics:
            self.evals = "all"
        elif "execution_accuracy" in metrics:
            self.evals = "exec"
        elif "query_match" in metrics:
            self.evals = "match"

        self.pred_answers = []
        self.ref_answers = []

    @classmethod
    def get_default_task_name(cls) -> str:
        return "Text to SQL Task"

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
                        table_str for table_str in result.retrieved_tables # TODO: what to pass to the generator
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
    ) -> Text2SQLTaskPerformanceDataModel:
        """
        Calculate downstream task metrics for the question answering task.
        """
        scores = {}
        

        result = Text2SQLTaskPerformanceDataModel(scores=scores)

        self.pred_answers = []
        self.ref_answers = []
        return result
