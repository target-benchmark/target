from tasks.AbsTargetTask import AbsTargetTask
from tasks.TasksDataModels import DownstreamTaskPerformanceDataModel
from dataset_loaders.LoadersDataModels import DatasetConfigDataModel
from dataset_loaders.TargetDatasetConfig import *
from generators.AbsTargetGenerator import AbsTargetGenerator
from generators.DefaultTargetGenerator import DefaultTargetGenerator


class TableRetrievalTask(AbsTargetTask):
    def __init__(
        self,
        task_name: str = "Table Retrieval Task",
        datasets_config: dict[str, dict[str, str]] = None,
        overwrite_default_datasets: bool = False,
        task_generator: AbsTargetGenerator = DefaultTargetGenerator,
        **kwargs,
    ):
        super().__init__(
            task_name=task_name,
            datasets_config=datasets_config,
            overwrite_default_datasets=overwrite_default_datasets,
            task_generator=task_generator,
            **kwargs,
        )

    def _get_default_dataset_config(self) -> dict[str, DatasetConfigDataModel]:
        """
        Returns the default dataset config for the class. MUST be implemented by any inherited task class.
        """
        # TODO: add more things here. this is for testing. carl note 4/10
        return {
            # DEFAULT_FETAQA_DATASET_CONFIG.dataset_name: DEFAULT_FETAQA_DATASET_CONFIG,
            "test_dataset": DEFAULT_FETAQA_DATASET_CONFIG  # this is for testing!!
        }

    def _get_downstream_task_results(
        self,
        id_to_query: dict[str, str],
        id_to_table_id: dict[str, str],
        retrieval_results: dict[str, list[str]],
        dataset_name: str,
    ) -> dict[str, str]:
        """
        no downstream task results to obtain here, return empty dictionary
        """
        return {}

    def _update_downstream_task_results(
        self,
        id_to_answer: dict[str, str],
        downstream_answers: dict[str, str],
    ) -> None:
        """
        No downstream tasks values to update for basic table retrieval task.
        """
        return

    def _calculate_downstream_task_metrics(
        self, **kwargs
    ) -> DownstreamTaskPerformanceDataModel:
        """
        No downstream metrics to calculate. return a default data model
        """
        return DownstreamTaskPerformanceDataModel()
