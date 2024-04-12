from abc import ABC, abstractmethod

from generators.AbsTargetGenerator import AbsTargetGenerator
from generators.DefaultTargetGenerator import DefaultTargetGenerator

from retrievers.AbsTargetRetrieverBase import AbsTargetRetrieverBase
from retrievers.AbsTargetCustomEmbeddingRetriver import (
    AbsTargetCustomEmbeddingRetriver as CustomEmbRetr,
)
from retrievers.AbsTargetStandardizedEmbeddingRetriever import (
    AbsTargetStandardizedEmbeddingRetriever as StandardizedEmbRetr,
)

from dataset_loaders.AbsTargetDatasetLoader import AbsTargetDatasetLoader
from dataset_loaders.LoadersDataModels import (
    QueryForTasksDataModel,
    DatasetConfigDataModel,
    HFDatasetConfigDataModel,
    GenericDatasetConfigDataModel,
)

from tasks.TasksDataModels import (
    RetrievalPerformanceDataModel,
    DownstreamTaskPerformanceDataModel,
    TaskResultsDataModel,
)

from logging import Logger
from dictionary_keys import *


class AbsTargetTask(ABC):

    def __init__(
        self,
        task_name: str,
        datasets_config: dict[str, dict[str, str]] = None,
        overwrite_default_datasets: bool = False,
        task_generator: AbsTargetGenerator = DefaultTargetGenerator,
        **kwargs,
    ):
        """
        Construct a task to run. The task should have an unique name, a generator for the downstream task completion (if needed). The user can optionally pass in a dictionary of dataset configs for the dataloader to load any custom datasets for the class.
        Parameters:
            task_name (str): name of the task. should be an unique identifier.

            datasets_config (dict[str, dict[str, str]], optional): if the user wants to add any custom datasets to the task, they can do so by passing in a dictionary to specify the dataset configuration. for the outer dictionary, the key is the name of the dataset, and the value is another dictionary. for the inner dictionary, either paths to hf corpus & queries datasets or a local path to a generic dataset should be included.
            example for a huggingface dataset:
                {
                    'hf_corpus_path': 'target-benchmark/fetaqa-corpus',
                    'hf_queries_path': 'target-benchmark/fetaqa-queries'
                }
            example for a local generic dataset:
                {
                    'dataset_path': 'local/path/to/dataset/foler/'
                }

            overwrite_default_datasets (bool, optional): each task have a set of default datasets that will be tested on. if the user chooses to input some dataset config that has a dataset under the same name as one of the default sets, this boolean dictates whether to overwrite the default datasets or not. defaults to False, as no overwrites.

            task_generator (AbsTargetGenerator, optional): each task as one corresponding generator for the downstream task. defaults to a default generator, just sends some openai api requests.
        """
        self.task_name: str = task_name
        self.dataset_config: dict[str, DatasetConfigDataModel] = (
            self._construct_dataset_config(datasets_config, overwrite_default_datasets)
        )
        self.task_generator: AbsTargetGenerator = task_generator
        self.tp = 0
        self.total_queries_processed = 0

    def _construct_dataset_config(
        self,
        datasets_config: dict[str, dict[str, str]],
        overwrite_default_datasets: bool,
    ) -> dict[str, DatasetConfigDataModel]:
        """
        builds the dataset config according to the user inputted dataset config (if any) and the default for the class.

        Parameters:
            datasets_config (dict[str, dict[str, str]]): user inputted datasets config dictionary.
            overwrite_default_datasets (bool): whether to overwrite the default datasets or not if the same name dataset is provided.
        """
        constructed_config: dict[str, DatasetConfigDataModel] = (
            self._get_default_dataset_config()
        )
        if datasets_config is not None:
            for key, value in datasets_config.items():
                assert (
                    HF_DATASET_CONFIG_CORPUS_FIELD in value
                    and HF_DATASET_CONFIG_QUERIES_FIELD in value
                ) or GENERIC_DATASET_CONFIG_FIELD in value, f"user inputted data config for {key} is missing fields! (current config: {value})"
                if key not in constructed_config or overwrite_default_datasets:
                    if key not in value:
                        value[DATASET_NAME] = key
                    if HF_DATASET_CONFIG_CORPUS_FIELD in value:
                        constructed_config[key] = HFDatasetConfigDataModel(**value)
                    else:
                        constructed_config[key] = GenericDatasetConfigDataModel(**value)

        return constructed_config

    def get_dataset_config(self) -> dict[str, DatasetConfigDataModel]:
        """
        Returns the dataset config of the task.

        Returns:
            a dictionary mapping dataset names to dataset config data models.
        """
        return self.dataset_config

    @abstractmethod
    def _get_default_dataset_config(self) -> dict[str, DatasetConfigDataModel]:
        """
        Returns the default dataset config for the class. MUST be implemented by any inherited task class.
        """
        pass

    def task_run(
        self,
        retriever: AbsTargetRetrieverBase,
        dataset_loaders: dict[str, AbsTargetDatasetLoader],
        logger: Logger,
        batch_size: int = 64,
        splits: str | list[str] = "test",
        top_k: int = 5,
        **kwargs,
    ) -> dict:
        assert (
            self.dataset_config.keys() <= dataset_loaders.keys()
        ), f"task's dataset config is not a subset of the dataset loaders passed in! \ntask dataset config: {self.dataset_config.keys()}\ndataset loaders passed in: {dataset_loaders.keys()}"

        assert isinstance(retriever, CustomEmbRetr) or isinstance(
            retriever, StandardizedEmbRetr
        ), f"the passed in retriever doesn't correctly inherit from the standardized or custom retriever classes!"

        task_results = {}

        logger.info(f"start task {self.task_name}")

        for dataset_name, dataset_loader in dataset_loaders.items():
            logger.info(f"running task on dataset {dataset_name}")

            for query_batch in dataset_loader.get_queries_for_task(splits, batch_size):
                id_to_query = self._query_info_to_queries(query_batch)
                id_to_answer = self._query_info_to_answers(query_batch)
                id_to_table_id = self._query_info_to_table_ids(query_batch)
                retrieved_tables = self._get_retrieval_results(
                    retriever, id_to_query, dataset_name, top_k
                )
                self._update_retrieval_results(id_to_table_id, retrieved_tables)

                downstream_task_results = self._get_downstream_task_results(
                    id_to_query, id_to_table_id, retrieved_tables, dataset_name
                )
                self._update_downstream_task_results(
                    id_to_answer, downstream_task_results
                )

                logger.info(
                    f"number of queries processed: {self.total_queries_processed}"
                )
            retrieval_performance = self._calculate_table_retrieval_metrics(top_k)
            downstream_task_performance = self._calculate_downstream_task_metrics(
                **kwargs
            )

            task_results[dataset_name] = TaskResultsDataModel(
                retrieval_performance=retrieval_performance,
                downstream_task_performance=downstream_task_performance,
            )
            logger.info(f"finished running task {self.task_name}")
        return task_results

    def _query_info_to_queries(
        self, batch: list[QueryForTasksDataModel]
    ) -> dict[int, str]:
        id_to_query = {}
        for query_info in batch:
            id_to_query[query_info.query_id] = query_info.query
        return id_to_query

    def _query_info_to_answers(
        self, batch: list[QueryForTasksDataModel]
    ) -> dict[int, str]:
        id_to_answer = {}
        for query_info_dict in batch:
            id_to_answer[query_info_dict.query_id] = query_info_dict.answer
        return id_to_answer

    def _query_info_to_table_ids(
        self, batch: list[QueryForTasksDataModel]
    ) -> dict[int, str]:
        id_to_table_id = {}
        for query_info_dict in batch:
            id_to_table_id[query_info_dict.query_id] = query_info_dict.table_id
        return id_to_table_id

    def _get_retrieval_results(
        self,
        retriever: AbsTargetRetrieverBase,
        id_to_query: dict[int, str],
        dataset_name: str,
        top_k: int,
    ) -> dict[int, list[str]]:
        is_standard = True
        if isinstance(retriever, CustomEmbRetr):
            is_standard = False
        if is_standard:
            # TODO: figure out what to do with embedding here
            # retreival_results = retriever.retrieve_batch(corpus_embedding=)
            retrieval_results = {}
        else:
            retrieval_results = retriever.retrieve_batch(
                queries=id_to_query, dataset_name=dataset_name, top_k=top_k
            )

        return retrieval_results

    def _update_retrieval_results(
        self,
        id_to_table_id: dict[int, str],
        new_retrieved_tables: dict[int, list[str]],
    ) -> None:
        for query_id, retrieved_tables in new_retrieved_tables.items():
            if id_to_table_id[query_id] in retrieved_tables:
                self.tp += 1
            self.total_queries_processed += 1

    def _calculate_table_retrieval_metrics(
        self, top_k: int
    ) -> RetrievalPerformanceDataModel:
        """
        Calculate the retrieval metrics after the table retrieval has been completed.
        """
        performace = RetrievalPerformanceDataModel(
            k=top_k, accuracy=self.tp / self.total_queries_processed
        )

        self.tp = 0
        self.total_queries_processed = 0
        return performace

    @abstractmethod
    def _get_downstream_task_results(
        self,
        id_to_query: dict[int, str],
        id_to_table_id: dict[int, str],
        retrieval_results: dict[int, list[str]],
        dataset_name: str,
    ) -> dict[int, str]:
        """
        TODO: how to pass through the tables? nested arrays, etc
        All downstreams tasks should fill out this method. ideally uses the retrieval results to generate the downstream answer, and return the performance of the downstream generation.
        """
        pass

    @abstractmethod
    def _update_downstream_task_results(
        self,
        id_to_answer: dict[int, str],
        downstream_answers: dict[int, str],
    ) -> None:
        """
        Update any values you keep track of for the downstream tasks.
        """
        pass

    @abstractmethod
    def _calculate_downstream_task_metrics(
        self, **kwargs
    ) -> DownstreamTaskPerformanceDataModel:
        """
        All downstreams tasks should fill out this method. uses whatever values that's been tracked & updated through the query eval, and calculate the metrics.
        """
        pass
