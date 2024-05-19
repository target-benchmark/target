from dataset_loaders.AbsDatasetLoader import AbsDatasetLoader
from dataset_loaders.LoadersDataModels import (
    QueryForTasksDataModel,
    DatasetConfigDataModel,
    HFDatasetConfigDataModel,
    GenericDatasetConfigDataModel,
)
from dataset_loaders.utils import markdown_table_with_headers

from dictionary_keys import *

from generators.AbsGenerator import AbsGenerator
from generators.DefaultGenerator import DefaultGenerator
from generators.GeneratorsDataModels import DownstreamGeneratedResultDataModel

from retrievers.AbsRetrieverBase import AbsRetrieverBase
from retrievers.AbsCustomEmbeddingRetriever import (
    AbsCustomEmbeddingRetriever as CustomEmbRetr,
)
from retrievers.AbsStandardizedEmbeddingRetriever import (
    AbsStandardizedEmbeddingRetriever as StandardizedEmbRetr,
)
from retrievers.RetrieversDataModels import RetrievalResultDataModel


from tasks.TasksDataModels import (
    RetrievalPerformanceDataModel,
    DownstreamTaskPerformanceDataModel,
    TaskResultsDataModel,
)


from abc import ABC, abstractmethod
from logging import Logger
from typing import Union, List, Dict


class AbsTask(ABC):

    def __init__(
        self,
        task_name: str = None,
        datasets_config: Dict[str, Dict[str, str]] = None,
        overwrite_default_datasets: bool = False,
        task_generator: AbsGenerator = None,
        **kwargs,
    ):
        """
        Construct a task to run. The task should have an unique name, a generator for the downstream task completion (if needed). The user can optionally pass in a dictionary of dataset configs for the dataloader to load any custom datasets for the class.
        Parameters:
            task_name (str): name of the task. should be an unique identifier.

            datasets_config (Dict[str, Dict[str, str]], optional): if the user wants to add any custom datasets to the task, they can do so by passing in a dictionary to specify the dataset configuration. for the outer dictionary, the key is the name of the dataset, and the value is another dictionary. for the inner dictionary, either paths to hf corpus & queries datasets or a local path to a generic dataset should be included.
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

            task_generator (AbsGenerator, optional): each task as one corresponding generator for the downstream task. defaults to a default generator, just sends some openai api requests.
        """
        if task_name is None:
            self.task_name = self.get_default_task_name()
        else:
            self.task_name: str = task_name
        self.dataset_config: Dict[str, DatasetConfigDataModel] = (
            self._construct_dataset_config(datasets_config, overwrite_default_datasets)
        )
        if task_generator is None:
            self.task_generator = DefaultGenerator()
        else:
            self.task_generator = task_generator
        self.true_positive = 0
        self.total_queries_processed = 0

    @classmethod
    @abstractmethod
    def get_default_task_name(cls):
        """
        Returns the default name of the task.
        """
        pass

    def get_task_name(self):
        """
        Returns the name of the task. NOTE: not the same as `get_default_task_name`. this name can be customized upon creation of the task.
        """
        return self.task_name

    def _construct_dataset_config(
        self,
        datasets_config: Dict[str, Dict[str, str]],
        overwrite_default_datasets: bool,
    ) -> Dict[str, DatasetConfigDataModel]:
        """
        builds the dataset config according to the user inputted dataset config (if any) and the default for the class.

        Parameters:
            datasets_config (Dict[str, Dict[str, str]]): user inputted datasets config dictionary.
            overwrite_default_datasets (bool): whether to overwrite the default datasets or not if the same name dataset is provided.

        Returns:
            a dictionary mapping the names of the dataset to the corresponding dataset configuration data model objects.
        """
        constructed_config: Dict[str, DatasetConfigDataModel] = (
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

    def get_dataset_config(self) -> Dict[str, DatasetConfigDataModel]:
        """
        Returns the dataset config of the task.

        Returns:
            a dictionary mapping dataset names to dataset config data models.
        """
        return self.dataset_config

    @abstractmethod
    def _get_default_dataset_config(self) -> Dict[str, DatasetConfigDataModel]:
        """
        Returns the default dataset config for the task. MUST be implemented by any inherited task class. For example, text-2-sql task would probably have SPIDER and BIRD as default datasets, thus the configs for these datasets should be included in this function when implementing the text-2-sql task class.
        """
        pass

    def task_run(
        self,
        retriever: AbsRetrieverBase,
        dataset_loaders: Dict[str, AbsDatasetLoader],
        logger: Logger,
        batch_size: int = 64,
        splits: Union[str, List[str]] = "test",
        top_k: int = 5,
        **kwargs,
    ) -> Dict[str, TaskResultsDataModel]:
        """
        Executes a retrieval task using the specified retriever and dataset loaders.

        Parameters:
            retriever (AbsRetrieverBase): The retriever instance to use for the task.
            dataset_loaders (Dict[str, AbsDatasetLoader]): Dictionary of dataset loaders keyed by dataset names.
            logger (Logger): Logger instance to log the task execution details.
            batch_size (int): The number of items to process in a single batch. Default is 64.
            splits (Union[str, List[str]], optional): Dataset split(s) to run the task on. Default is "test".
            top_k (int, optional): The top k tables to retrieve. Default is 5.
            **kwargs: Additional keyword arguments for fine-tuning the task execution.

        Returns:
            A dictionary with the results of the retrieval task. Maps dataset name to a task result data model object. The task result data model object records both the retrieval performance and the downstream generation results.
        """
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
            table_id_to_table = dataset_loader.get_table_id_to_table(splits=splits)
            for query_batch in dataset_loader.get_queries_for_task(splits, batch_size):
                retrieved_tables = self._get_retrieval_results(
                    retriever, query_batch, dataset_name, top_k
                )
                self._update_retrieval_metrics(query_batch, retrieved_tables)
                self._fill_retrieval_results_with_table_strs(
                    retrieved_tables, table_id_to_table
                )
                downstream_task_results = self._get_downstream_task_results(
                    query_batch, retrieved_tables, dataset_name
                )
                logger.info(
                    f"generated results {downstream_task_results}"
                )  # TODO: comment this out, this is for testing
                self._update_downstream_task_metrics(
                    query_batch, downstream_task_results
                )

                logger.info(
                    f"number of queries processed: {self.total_queries_processed}"
                )
            retrieval_performance = self._calculate_table_retrieval_performance(top_k)
            downstream_task_performance = self._calculate_downstream_task_performance(
                **kwargs
            )

            task_results[dataset_name] = TaskResultsDataModel(
                retrieval_performance=retrieval_performance,
                downstream_task_performance=downstream_task_performance,
            )
            logger.info(f"finished running task {self.task_name}")
        return task_results

    def _fill_retrieval_results_with_table_strs(
        self,
        retrieval_results: List[RetrievalResultDataModel],
        table_id_to_tables: Dict[str, List[List]],
    ) -> None:
        """
        Fills the retrieval result data model objects with Markdown table strings based on the table IDs stored in each retrieval result.

        Parameters:
            retrieval_results (List[RetrievalResultDataModel]): List of retrieval result data models to be filled with table strings.
            table_id_to_tables (Dict[str, List[List]]): Dictionary mapping table IDs to their corresponding table data in nested list format.

        Returns:
            None
        """
        for result in retrieval_results:
            result.retrieved_tables = [
                markdown_table_with_headers(table_id_to_tables[table_id])
                for table_id in result.retrieval_results
            ]

    def _get_retrieval_results(
        self,
        retriever: AbsRetrieverBase,
        query_batch: List[QueryForTasksDataModel],
        dataset_name: str,
        top_k: int,
    ) -> List[RetrievalResultDataModel]:
        """
        Retrieves the top k results for each query in the batch using the specified retriever from a dataset.

        Parameters:
            retriever (AbsRetrieverBase): The retriever for fetching the results.
            query_batch (List[QueryForTasksDataModel]): A list of queries for which results are to be retrieved.
            dataset_name (str): The name of the dataset to retrieve results from.
            top_k (int): The number of top results to retrieve for each query.

        Returns:
            A list of retrieval result data models, each containing the top k results for a query.
        """
        if isinstance(retriever, StandardizedEmbRetr):
            # TODO: figure out what to do with embedding here
            # retreival_results = retriever.retrieve_batch(corpus_embedding=)
            retrieval_results = {}
        elif isinstance(retriever, CustomEmbRetr):
            retrieval_results = retriever.retrieve_batch(
                queries=query_batch, dataset_name=dataset_name, top_k=top_k
            )
        else:
            raise ValueError(
                f"retriever passed in doesn't inherit from the base retriever classes! (is of type {type(retriever)})"
            )

        return retrieval_results

    def _update_retrieval_metrics(
        self,
        query_batch: List[QueryForTasksDataModel],
        new_retrieved_tables: List[RetrievalResultDataModel],
    ) -> None:
        """
        Updates the tracked retrieval metrics with the new retrieval results.

        Parameters:
            query_batch (List[QueryForTasksDataModel]): queries & the corresponding gold table and gold answer.
            new_retrieved_tables (List[RetrievalResultDataModel]): New retrieval result data models that contains the retrieval results.

        Returns:
            None
        """
        for query, retrieval_result in zip(query_batch, new_retrieved_tables):
            if query.table_id in retrieval_result.retrieval_results:
                self.true_positive += 1
            self.total_queries_processed += 1

    def _calculate_table_retrieval_performance(
        self, top_k: int
    ) -> RetrievalPerformanceDataModel:
        """
        Calculate the retrieval performance after the table retrieval has been completed.

        Parameters:
            top_k (int): The top k tables to retrieved.

        Returns:
            a retrieval performance data model that contains the accuracy of the retrieval for a dataset on this task.
        """
        performace = RetrievalPerformanceDataModel(
            k=top_k, accuracy=self.true_positive / self.total_queries_processed
        )

        self.true_positive = 0
        self.total_queries_processed = 0
        return performace

    @abstractmethod
    def _get_downstream_task_results(
        self,
        query_batch: List[QueryForTasksDataModel],
        retrieval_results: List[RetrievalResultDataModel],
        dataset_name: str,
    ) -> List[DownstreamGeneratedResultDataModel]:
        """
        Given the query and the retrieval results, generate downstream task results. Uses the tasks's generator to generate the downstream task result.

        Parameters:
            query_batch (List[QueryForTasksDataModel]): Datamodel objects, contains queries to generate answers for.
            retrieval_results (List[RetrievalResultDataModel]): retrieved tables.
            dataset_name (str): Name of the dataset.

        Returns:
            a list of downstream generated result data model objects, contains query id to generate answer.
        """
        pass

    @abstractmethod
    def _update_downstream_task_metrics(
        self,
        query_batch: List[QueryForTasksDataModel],
        downstream_answers: List[DownstreamGeneratedResultDataModel],
    ) -> None:
        """
        Update any values needed for the calculation of metrics for the downstream tasks. For example, if the task is table fact verification, update the tp, fp, tn, fn in order to caluclate f1, accuracy, etc.

        Parameters:
            query_batch (List[QueryForTasksDataModel]): Data model objects, contains gold tables and gold answer for the query.
            downstream_answers (List[DownstreamGeneratedResultDataModel]): generated downstream answers.
        """
        pass

    @abstractmethod
    def _calculate_downstream_task_performance(
        self, **kwargs
    ) -> DownstreamTaskPerformanceDataModel:
        """
        All downstreams tasks should fill out this method. uses whatever values that's been tracked & updated through the query eval, and calculate the metrics.
        """
        pass
