from typing import Dict, List, Tuple, Union

from target_benchmark.dataset_loaders.LoadersDataModels import DatasetConfigDataModel
from target_benchmark.dataset_loaders.TargetDatasetConfig import (
    FACT_VER_DATASETS,
    QUESTION_ANSWERING_DATASETS,
    TEXT_2_SQL_DATASETS,
)
from target_benchmark.generators.AbsGenerator import AbsGenerator
from target_benchmark.generators.GeneratorsDataModels import (
    DownstreamGeneratedResultDataModel,
)
from target_benchmark.retrievers.RetrieversDataModels import RetrievalResultDataModel
from target_benchmark.tasks.AbsTask import AbsTask
from target_benchmark.tasks.TasksDataModels import DownstreamTaskPerformanceDataModel


class TableRetrievalTask(AbsTask):
    AVAILABLE_METRICS = set(["precision"])
    DEFAULT_METRICS = set(["precision"])

    def __init__(
        self,
        datasets_config: Union[
            Dict[str, Union[Dict[str, str], DatasetConfigDataModel]], None
        ] = None,
        task_generator: AbsGenerator = None,
        **kwargs,
    ):
        super().__init__(
            task_name=self.get_default_task_name(),
            datasets_config=datasets_config,
            task_generator=task_generator,
            **kwargs,
        )

    @classmethod
    def get_default_task_name(cls) -> str:
        return "Table Retrieval Task"

    @classmethod
    def get_available_metrics(cls) -> str:
        return str(cls.AVAILABLE_METRICS)

    @classmethod
    def _get_default_dataset_config(cls) -> Dict[str, DatasetConfigDataModel]:
        """
        Returns the default dataset config for the class. MUST be implemented by any inherited task class.
        """
        config = dict(QUESTION_ANSWERING_DATASETS)
        config.update(dict(FACT_VER_DATASETS))
        config.update(dict(TEXT_2_SQL_DATASETS))
        return config

    def _get_downstream_task_results(
        self,
        query_batch: Dict[str, List],
        retrieval_results: List[RetrievalResultDataModel],
        dataset_name: str,
        table_id_to_table: Dict[Tuple[str, str], List[List]],
    ) -> List[DownstreamGeneratedResultDataModel]:
        return []

    def _update_downstream_task_metrics(
        self,
        query_batch: Dict[str, List],
        downstream_results: List[DownstreamGeneratedResultDataModel],
    ) -> None:
        pass

    def _calculate_downstream_task_performance(
        self, **kwargs
    ) -> DownstreamTaskPerformanceDataModel:
        return DownstreamTaskPerformanceDataModel()
