from dataset_loaders.LoadersDataModels import (
    DatasetConfigDataModel,
    QueryForTasksDataModel,
)
from dataset_loaders.TargetDatasetConfig import *

from generators.AbsGenerator import AbsGenerator
from generators.GeneratorsDataModels import DownstreamGeneratedResultDataModel

from retrievers.RetrieversDataModels import RetrievalResultDataModel

from tasks.AbsTask import AbsTask
from tasks.TasksDataModels import DownstreamTaskPerformanceDataModel

from typing import List, Dict


class TableRetrievalTask(AbsTask):
    def __init__(
        self,
        datasets_config: Dict[str, Dict[str, str]] = None,
        overwrite_default_datasets: bool = False,
        task_generator: AbsGenerator = None,
        **kwargs,
    ):
        super().__init__(
            task_name=self.get_default_task_name(),
            datasets_config=datasets_config,
            overwrite_default_datasets=overwrite_default_datasets,
            task_generator=task_generator,
            **kwargs,
        )

    @classmethod
    def get_default_task_name(cls) -> str:
        return "Table Retrieval Task"

    def _get_default_dataset_config(self) -> Dict[str, DatasetConfigDataModel]:
        """
        Returns the default dataset config for the class. MUST be implemented by any inherited task class.
        """
        # TODO: add more things here. this is for testing. carl note 4/10
        return {
            # DEFAULT_FETAQA_DATASET_CONFIG.dataset_name: DEFAULT_FETAQA_DATASET_CONFIG,
            DEFAULT_DUMMY_DATASET_CONFIG.dataset_name: DEFAULT_DUMMY_DATASET_CONFIG
        }

    def _get_downstream_task_results(
        self,
        query_batch: List[QueryForTasksDataModel],
        retrieval_results: List[RetrievalResultDataModel],
        dataset_name: str,
    ) -> List[DownstreamGeneratedResultDataModel]:
        """
        TODO: how to pass through the tables? nested arrays, etc
        All downstreams tasks should fill out this method. ideally uses the retrieval results to generate the downstream answer, and return the performance of the downstream generation.
        """
        return []

    def _update_downstream_task_metrics(
        self,
        query_batch: List[QueryForTasksDataModel],
        downstream_results: List[DownstreamGeneratedResultDataModel],
    ) -> None:
        """
        Update any values you keep track of for the downstream tasks.
        """
        pass

    def _calculate_downstream_task_performance(
        self, **kwargs
    ) -> DownstreamTaskPerformanceDataModel:
        """
        All downstreams tasks should fill out this method. uses whatever values that's been tracked & updated through the query eval, and calculate the metrics.
        """
        return DownstreamTaskPerformanceDataModel()
