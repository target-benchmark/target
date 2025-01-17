import time
from abc import ABC, abstractmethod
from logging import Logger
from pathlib import Path
from typing import Dict, Generator, List, Tuple, Union

from tqdm import tqdm

from target_benchmark.dataset_loaders.AbsDatasetLoader import AbsDatasetLoader
from target_benchmark.dataset_loaders.DatasetLoaderEnums import QueryType
from target_benchmark.dataset_loaders.LoadersDataModels import (
    DatasetConfigDataModel,
    GenericDatasetConfigDataModel,
    HFDatasetConfigDataModel,
    NeedleInHaystackDatasetConfigDataModel,
    Text2SQLDatasetConfigDataModel,
)
from target_benchmark.dataset_loaders.TargetDatasetConfig import (
    NEEDLE_IN_HAYSTACK_DATASETS,
)
from target_benchmark.dictionary_keys import (
    CLIENT_KEY_NAME,
    DATABASE_ID_COL_NAME,
    DATASET_NAME,
    HF_DATASET_CONFIG_CORPUS_FIELD,
    QUERY_TYPE,
    TABLE_ID_COL_NAME,
)
from target_benchmark.generators.AbsGenerator import AbsGenerator
from target_benchmark.generators.DefaultGenerator import DefaultGenerator
from target_benchmark.generators.GeneratorsDataModels import (
    DownstreamGeneratedResultDataModel,
)
from target_benchmark.retrievers.AbsCustomEmbeddingRetriever import (
    AbsCustomEmbeddingRetriever as CustomEmbRetr,
)
from target_benchmark.retrievers.AbsRetrieverBase import AbsRetrieverBase
from target_benchmark.retrievers.AbsStandardEmbeddingRetriever import (
    AbsStandardEmbeddingRetriever as StandardizedEmbRetr,
)
from target_benchmark.retrievers.RetrieversDataModels import RetrievalResultDataModel
from target_benchmark.tasks.TasksDataModels import (
    DownstreamTaskPerformanceDataModel,
    RetrievalPerformanceDataModel,
    TaskResultsDataModel,
)
from target_benchmark.tasks.utils import (
    append_results,
    construct_persistence_path,
    find_resume_indices,
    generate_batches_from_file,
    load_data_model_from_persistence_file,
    update_query_batch,
    validate_dataset_configs,
)


class AbsTask(ABC):
    def __init__(
        self,
        task_name: str = None,
        datasets_config: Union[Dict[str, Union[Dict[str, str], DatasetConfigDataModel]], None] = None,
        task_generator: AbsGenerator = None,
        **kwargs,
    ):
        """
        Construct a task to run. The task should have an unique name, a generator for the downstream task completion (if needed). The user can optionally pass in a dictionary of dataset configs for the dataloader to load any custom datasets for the class.
        Parameters:
            task_name (str): name of the task. should be an unique identifier.

            datasets_config (Dict[str, Union[Dict[str, str], DatasetConfigDataModel]], optional): if the user wants to add any custom datasets to the task, they can do so by passing in a dictionary to specify the dataset configuration. for the outer dictionary, the key is the name of the dataset, and the value is another dictionary. for the inner dictionary, either paths to hf corpus & queries datasets or a local path to a generic dataset should be included.
            example for a huggingface dataset:
                {
                    'hf_corpus_path': 'target-benchmark/fetaqa-corpus',
                    'hf_queries_path': 'target-benchmark/fetaqa-queries'
                }
            example for a local generic dataset:
                {
                    'dataset_path': 'local/path/to/dataset/foler/'
                }

            task_generator (AbsGenerator, optional): each task as one corresponding generator for the downstream task. defaults to a default generator, just sends some openai api requests.
        """
        if task_name is None:
            self.task_name = self.get_default_task_name()
        else:
            self.task_name: str = task_name
        self.dataset_config: Dict[str, DatasetConfigDataModel] = self._construct_dataset_config(datasets_config)

        self.task_generator = task_generator
        if task_generator is None and task_name != "Table Retrieval Task":
            self.task_generator = DefaultGenerator()
        self.total_queries_processed = 0
        self.num_overlap = 0
        self.total_tables = 0
        self.total_tables_capped = 0

    @classmethod
    @abstractmethod
    def get_default_task_name(cls) -> str:
        """
        Returns the default name of the task.
        """
        pass

    @classmethod
    @abstractmethod
    def get_available_metrics(cls) -> str:
        """
        Returns the metrics available for a class.
        """
        pass

    @classmethod
    def get_available_datasets(cls) -> Dict[str, DatasetConfigDataModel]:
        return cls.append_nih_datasets(cls._get_default_dataset_config())

    def _validate_dataset_loaders(self, dataset_loaders: Dict[str, AbsDatasetLoader]):
        for dataset_loader in dataset_loaders.values():
            if dataset_loader.dataset_name not in self.dataset_config.keys():
                raise ValueError(
                    f"Invalid dataset loader: '{dataset_loader.dataset_name}' is not configured for this task. "
                    f"Ensure that the names match one of the configured datasets: {list(self.  dataset_config.keys())}."
                )

    def get_task_name(self):
        """
        Returns the name of the task. NOTE: not the same as `get_default_task_name`. this name can be customized upon creation of the task.
        """
        return self.task_name

    def _construct_dataset_config(
        self,
        datasets_config: Union[Dict[str, Union[Dict[str, str], DatasetConfigDataModel]], None] = None,
    ) -> Dict[str, DatasetConfigDataModel]:
        """
        builds the dataset config according to the user inputted dataset config (if any) and the default for the class.

        Parameters:
            datasets_config (Dict[str, Dict[str, str]]): user inputted datasets config dictionary.

        Returns:
            a dictionary mapping the names of the dataset to the corresponding dataset configuration data model objects.
        """
        if datasets_config is None:
            return self._get_default_dataset_config()
        constructed_config = {}
        for key, value in datasets_config.items():
            assert key not in constructed_config, f"duplicate dataset name {key}!"
            if isinstance(value, Dict):
                # TODO: Needle in haystack config creation
                assert QUERY_TYPE in value, f"need to specify a query type in the dictionary config with key {QUERY_TYPE}"
                if key not in value:
                    value[DATASET_NAME] = key
                if value[QUERY_TYPE] == QueryType.TEXT_2_SQL.value:
                    constructed_config[key] = Text2SQLDatasetConfigDataModel(**value)
                elif value[QUERY_TYPE] == QueryType.NIH.value:
                    constructed_config[key] = NeedleInHaystackDatasetConfigDataModel(**value)
                elif HF_DATASET_CONFIG_CORPUS_FIELD in value:
                    constructed_config[key] = HFDatasetConfigDataModel(**value)
                else:
                    constructed_config[key] = GenericDatasetConfigDataModel(**value)
            elif isinstance(value, DatasetConfigDataModel):
                constructed_config[key] = value
            else:
                wrong_type = type(value)
                raise ValueError(
                    f"passed in config {value} is of type {wrong_type}, not one of type dictionary or `DatasetConfigDataModel`."
                )
        validate_dataset_configs(constructed_config)
        return constructed_config

    def get_dataset_config(self) -> Dict[str, DatasetConfigDataModel]:
        """
        Returns the dataset config of the task.

        Returns:
            a dictionary mapping dataset names to dataset config data models.
        """
        return self.dataset_config

    @classmethod
    @abstractmethod
    def _get_default_dataset_config(cls) -> Dict[str, DatasetConfigDataModel]:
        """
        Returns the default dataset config for the task. MUST be implemented by any inherited task class. For example, text-2-sql task would probably have SPIDER and BIRD as default datasets, thus the configs for these datasets should be included in this function when implementing the text-2-sql task class.
        """
        pass

    @classmethod
    def append_nih_datasets(cls, configs: Dict[str, DatasetConfigDataModel]) -> Dict[str, DatasetConfigDataModel]:
        """
        Appends needle in haystack dataset configs to the passed in configs.
        Needle in Haystack includes the following datasets:
            gittables
            TODO: more to come
        """
        updated_configs = dict(NEEDLE_IN_HAYSTACK_DATASETS)
        updated_configs.update(dict(configs))
        return updated_configs

    def _run_retrieval_batch(
        self,
        retriever: AbsRetrieverBase,
        dataset_name: str,
        query_batch: Dict[str, List],
        top_k: int,
        prev_retrieval_results_gen: Generator[List[RetrievalResultDataModel], None, None],
        path_to_retrieval_results: Union[Path, None],
        **kwargs,
    ) -> tuple[list[RetrievalResultDataModel], float, float, int]:
        """
        Run the retrieval on a new batch of queries.
        """
        retrieval_results = []
        process_duration, wall_clock_duration = 0, 0
        num_retrieved = 0
        try:
            # getting some more results from the previous results generator
            retrieval_results = next(prev_retrieval_results_gen)
        except StopIteration:
            pass
        # count how many loaded queries
        num_prev_res = len(retrieval_results)
        # count how many entries in the batch
        batch_size = len(query_batch[DATABASE_ID_COL_NAME])
        if num_prev_res < batch_size:
            num_retrieved = batch_size - num_prev_res
            # if the last batch from the generator is contains fewer entries than batch_size
            # or the generator is empty, need to do more retrieval with retriever

            # truncate batch as necessary
            updated_batch = update_query_batch(query_batch, num_prev_res)
            # call retriever to get new results
            retrieval_results_new, process_duration, wall_clock_duration = self._get_retrieval_results(
                retriever,
                updated_batch,
                dataset_name,
                top_k,
                **kwargs,
            )

            # write results
            append_results(retrieval_results_new, path_to_retrieval_results)

            # update full list
            retrieval_results.extend(retrieval_results_new)

        # update the metrics
        self._update_retrieval_metrics(query_batch, retrieval_results)
        return retrieval_results, process_duration, wall_clock_duration, num_retrieved

    def _run_downstream_batch(
        self,
        retrieval_results: List[RetrievalResultDataModel],
        dataset_name: str,
        query_batch: Dict[str, List],
        table_id_to_table: Dict[Tuple[str, str], List[List]],
        prev_downstream_results_gen: Generator[List[DownstreamGeneratedResultDataModel], None, None],
        path_to_downstream_results: Union[Path, None],
    ):
        downstream_results = []
        try:
            # try to get more previously persisted results for the generator
            downstream_results = next(prev_downstream_results_gen)
        except StopIteration:
            pass
        if len(downstream_results) < len(query_batch[DATABASE_ID_COL_NAME]):
            # if the last batch from the generator is contains fewer entries than batch_size
            # or the generator is empty, need to generate more downstream.

            updated_batch = update_query_batch(query_batch, len(downstream_results))
            downstream_results_new = self._get_downstream_task_results(
                query_batch=updated_batch,
                retrieval_results=retrieval_results,
                dataset_name=dataset_name,
                table_id_to_table=table_id_to_table,
            )
            append_results(downstream_results_new, path_to_downstream_results)
            downstream_results.extend(downstream_results_new)
        self._update_downstream_task_metrics(query_batch, downstream_results)

    def task_run(
        self,
        retriever: AbsRetrieverBase,
        dataset_loaders: Dict[str, AbsDatasetLoader],
        logger: Logger,
        batch_size: int = 64,
        top_k: int = 5,
        path_to_retrieval_results_dir: Union[Path, None] = None,
        path_to_downstream_results_dir: Union[Path, None] = None,
        **kwargs,
    ) -> Dict[str, TaskResultsDataModel]:
        """
        Executes a retrieval task using the specified retriever and dataset loaders.

        Parameters:
            retriever (AbsRetrieverBase): The retriever instance to use for the task.
            dataset_loaders (Dict[str, AbsDatasetLoader]): Dictionary of dataset loaders keyed by dataset names.
            logger (Logger): Logger instance to log the task execution details.
            batch_size (int): The number of items to process in a single batch. Default is 64.
            top_k (int, optional): The top k tables to retrieve. Default is 5.
            **kwargs: Additional keyword arguments for fine-tuning the task execution.

        Returns:
            A dictionary with the results of the retrieval task. Maps dataset name to a task result data model object. The task result data model object records both the retrieval performance and the downstream generation results.
        """
        self._validate_dataset_loaders(dataset_loaders)

        assert isinstance(retriever, CustomEmbRetr) or isinstance(
            retriever, StandardizedEmbRetr
        ), "the passed in retriever doesn't correctly inherit from the standardized or custom retriever classes!"

        task_results = {}

        logger.info(f"start task {self.task_name}")

        for dataset_name, dataset_loader in dataset_loaders.items():
            # construct the path to persistence files
            path_to_retrieval_results = construct_persistence_path(path_to_retrieval_results_dir, dataset_name, top_k)
            path_to_downstream_results = construct_persistence_path(path_to_downstream_results_dir, dataset_name, top_k)

            # construct generators
            prev_retrieval_res_gen = generate_batches_from_file(
                path_to_retrieval_results,
                batch_size,
                RetrievalResultDataModel,
            )
            prev_downstream_res_gen = generate_batches_from_file(
                path_to_downstream_results,
                batch_size,
                DownstreamGeneratedResultDataModel,
            )

            logger.info(f"running task on dataset {dataset_name}")

            table_id_to_table = dataset_loader.get_table_id_to_table()
            # some retrieval metrics to track
            total_process_duration = 0
            total_wall_clock_duration = 0
            total_num_retrieved = 0

            # set up progress bar
            total_num_queries = dataset_loader.get_queries_size()
            progress_bar = tqdm(total=total_num_queries, desc=f"Retrieving Tables for {dataset_name}...")
            for query_batch in dataset_loader.get_queries_for_task(batch_size=batch_size):
                # run retrieval on batch
                retrieval_results, process_duration, wall_clock_duration, num_retrieved = self._run_retrieval_batch(
                    retriever=retriever,
                    dataset_name=dataset_name,
                    query_batch=query_batch,
                    top_k=top_k,
                    prev_retrieval_results_gen=prev_retrieval_res_gen,
                    path_to_retrieval_results=path_to_retrieval_results,
                    **kwargs,
                )

                # update time spent
                total_process_duration += process_duration
                total_wall_clock_duration += wall_clock_duration
                total_num_retrieved += num_retrieved

                # run downstream on batch
                self._run_downstream_batch(
                    retrieval_results,
                    dataset_name,
                    query_batch,
                    table_id_to_table,
                    prev_downstream_res_gen,
                    path_to_downstream_results,
                )

                if self.total_queries_processed % 200 == 0:
                    logger.info(f"number of queries processed: {self.total_queries_processed}")
                progress_bar.update(batch_size)
            progress_bar.update(total_num_queries - progress_bar.n)
            progress_bar.close()

            # retrieval performance, precision, recall, f1, etc.
            retrieval_performance = self._calculate_table_retrieval_performance(
                top_k,
                total_process_duration,
                total_wall_clock_duration,
                total_num_retrieved,
            )
            # downstream performance, depends on what task is being run.
            downstream_task_performance = self._calculate_downstream_task_performance(**kwargs)

            task_results[dataset_name] = TaskResultsDataModel(
                retrieval_performance=retrieval_performance,
                downstream_task_performance=downstream_task_performance,
            )
            logger.info(f"finished running task {self.task_name}")
        return task_results

    def evaluate_downstream(
        self,
        logger: Logger,
        dataset_loaders: Dict[str, AbsDatasetLoader],
        path_to_retrieval_results: Path,
        path_to_downstream_results: Union[Path, None] = None,
        **kwargs,
    ) -> Dict[str, DownstreamTaskPerformanceDataModel]:
        # TODO: Deprecate. Legacy function. resume functionalities now fully encapsulated within the `task_run` funciton.
        task_results = {}

        # get the persisted retrieval results
        retrieval_results = load_data_model_from_persistence_file(path_to_retrieval_results, RetrievalResultDataModel)
        if not retrieval_results:
            raise ValueError("File empty or could not parse any RetrievalResultDataModel objects!")
        # if previously partial downstream results are obtained, find start indices
        resume_indices = find_resume_indices(dataset_loaders, path_to_downstream_results)
        # load all downstream results in file
        all_prev_downstream_results = load_data_model_from_persistence_file(
            path_to_downstream_results, DownstreamGeneratedResultDataModel
        )
        prev_downstream_results = {}
        # put them into dictionary by dataset name
        for result in all_prev_downstream_results:
            current_dataset_name = result.dataset_name
            if current_dataset_name not in prev_downstream_results:
                prev_downstream_results[current_dataset_name] = [result]
            else:
                prev_downstream_results[current_dataset_name].append(result)
        # TODO: support batching
        batch_size = 1
        idx = 0
        for dataset_name, dataset_loader in dataset_loaders.items():
            table_id_to_table = dataset_loader.get_table_id_to_table()
            resume_index = resume_indices[dataset_name]
            for current_index, query_batch in tqdm(
                enumerate(dataset_loader.get_queries_for_task(batch_size)),
                total=dataset_loader.get_queries_size(),
                desc="Getting downstream task results...",
            ):
                if current_index < resume_index:
                    # if the resume index is greater,
                    # just update metrics using the existing index
                    self._update_downstream_task_metrics(
                        query_batch,
                        prev_downstream_results[dataset_name][current_index : current_index + batch_size],
                    )
                    idx += batch_size
                    continue
                retrieved_table = retrieval_results[idx : idx + batch_size]
                downstream_results = self._get_downstream_task_results(
                    query_batch, retrieved_table, dataset_name, table_id_to_table
                )
                append_results(downstream_results, path_to_downstream_results)
                self._update_downstream_task_metrics(query_batch, downstream_results)
                idx += 1
            performance = self._calculate_downstream_task_performance(dataset_name=dataset_name, **kwargs)
            task_results[dataset_name] = performance
            logger.info(f"finished running downstream eval on {dataset_name}")

        return task_results

    def _get_retrieval_results(
        self,
        retriever: AbsRetrieverBase,
        query_batch: Dict[str, List],
        dataset_name: str,
        top_k: int,
        **kwargs,
    ) -> Tuple[List[RetrievalResultDataModel], float, float]:
        """
        Retrieves the top k results for each query in the batch using the specified retriever from a dataset.

        Parameters:
            retriever (AbsRetrieverBase): The retriever for fetching the results.
            query_batch (Dict[str, List]): A dictionary of list of queries for which results are to be retrieved.
            dataset_name (str): The name of the dataset to retrieve results from.
            top_k (int): The number of top results to retrieve for each query.

        Returns:
            A list of retrieval result data models, each containing the top k results for a query.
        """
        start_process_time = time.process_time()
        start_wall_clock_time = time.time()
        if isinstance(retriever, StandardizedEmbRetr):
            if CLIENT_KEY_NAME not in kwargs:
                raise KeyError(f"missing kwarg {CLIENT_KEY_NAME}, required for standardized retriever")
            retrieval_results = retriever.retrieve_batch(
                queries=query_batch,
                dataset_name=dataset_name,
                top_k=top_k,
                client=kwargs.get(CLIENT_KEY_NAME),
            )
        elif isinstance(retriever, CustomEmbRetr):
            retrieval_results = retriever.retrieve_batch(queries=query_batch, dataset_name=dataset_name, top_k=top_k)
        else:
            raise ValueError(
                f"retriever passed in doesn't inherit from the base retriever classes! (is of type {type(retriever)})"
            )
        end_process_time = time.process_time()
        end_wall_clock_time = time.time()
        process_duration = end_process_time - start_process_time
        wall_clock_duration = end_wall_clock_time - start_wall_clock_time
        return retrieval_results, process_duration, wall_clock_duration

    def _update_retrieval_metrics(
        self,
        query_batch: Dict[str, List],
        new_retrieval_results: List[RetrievalResultDataModel],
    ) -> None:
        """
        Updates the tracked retrieval metrics with the new retrieval results.

        Parameters:
            query_batch (Dict[str, List]): queries & the corresponding gold table and gold answer.
            new_retrieval_results (List[RetrievalResultDataModel]): New retrieval result data models that contains the retrieval results.

        Returns:
            None
        """
        for idx in range(len(new_retrieval_results)):
            gold_db_id: str = query_batch[DATABASE_ID_COL_NAME][idx]
            gold_table_id: Union[str, List[str]] = query_batch[TABLE_ID_COL_NAME][idx]
            retrieval_result = new_retrieval_results[idx]
            if not isinstance(gold_table_id, list):
                # Treat all datasets (even single-table settings) as a list
                gold_table_id = [gold_table_id]
            normalized_gold_tables = set(
                (gold_db_id, t) for t in gold_table_id
            )  # E.g. {('soccer_3', 'club'), ('soccer_3', 'players')}
            self.num_overlap += len(normalized_gold_tables.intersection(set(retrieval_result.retrieval_results)))
            self.total_tables += len(normalized_gold_tables)
            # Cap denominator at len(retrieval_result.retrieval_results) (aka `k`)
            self.total_tables_capped += min(len(normalized_gold_tables), len(retrieval_result.retrieval_results))
            self.total_queries_processed += 1

    def _calculate_table_retrieval_performance(
        self,
        top_k: int,
        total_retrieval_duration_process: float,
        total_retrieval_duration_wall_clock: float,
        num_queries_retrieved: int,
    ) -> RetrievalPerformanceDataModel:
        """
        Calculate the retrieval performance after the table retrieval has been completed.

        Parameters:
            top_k (int): The top k tables to retrieved.

        Returns:
            a retrieval performance data model that contains the accuracy of the retrieval for a dataset on this task.
        """
        retrieval_duration_process = round(total_retrieval_duration_process, 5)
        retrieval_duration_wall_clock = round(total_retrieval_duration_wall_clock, 5)
        avg_dur_process = 0
        avg_dur_wall_clock = 0
        if num_queries_retrieved == 0:
            retrieval_duration_process = 0
            retrieval_duration_wall_clock = 0
        else:
            avg_dur_process = round(retrieval_duration_process / num_queries_retrieved, 5)
            avg_dur_wall_clock = round(retrieval_duration_wall_clock / num_queries_retrieved, 5)

        if self.total_queries_processed != 0:
            # TODO: update recall calculation once text 2 sql in db retrieval is done
            performace = RetrievalPerformanceDataModel(
                k=top_k,
                # TODO: what is meant to be captured by accuracy?
                accuracy=self.num_overlap / self.total_tables,
                recall=self.num_overlap / self.total_tables,
                capped_recall=self.num_overlap / self.total_tables_capped,
                retrieval_duration_process=retrieval_duration_process,
                avg_retrieval_duration_process=avg_dur_process,
                retrieval_duration_wall_clock=retrieval_duration_wall_clock,
                avg_retrieval_duration_wall_clock=avg_dur_wall_clock,
            )
        else:
            raise ValueError("haven't processed any queries!")

        self.total_queries_processed = 0
        self.num_overlap = 0
        self.total_tables = 0
        self.total_tables_capped = 0
        return performace

    @abstractmethod
    def _get_downstream_task_results(
        self,
        query_batch: Dict[str, List],
        retrieval_results: List[RetrievalResultDataModel],
        dataset_name: str,
        table_id_to_table: Dict[Tuple[str, str], List[List]],
    ) -> List[DownstreamGeneratedResultDataModel]:
        """
        Given the query and the retrieval results, generate downstream task results. Uses the tasks's generator to generate the downstream task result.

        Parameters:
            query_batch (Dict[str, List]): dictionaries, contains queries to generate answers for.
            retrieval_results (List[RetrievalResultDataModel]): retrieved tables.
            dataset_name (str): Name of the dataset.

        Returns:
            a list of downstream generated result data model objects, contains query id to generate answer.
        """
        pass

    @abstractmethod
    def _update_downstream_task_metrics(
        self,
        query_batch: Dict[str, List],
        downstream_results: List[DownstreamGeneratedResultDataModel],
    ) -> None:
        """
        Update any values needed for the calculation of metrics for the downstream tasks. For example, if the task is table fact verification, update the tp, fp, tn, fn in order to caluclate f1, accuracy, etc.

        Parameters:
            query_batch (Dict[str, List]): dictionaries, contains gold tables and gold answer for the query.
            downstream_results (List[DownstreamGeneratedResultDataModel]): generated downstream answers.
        """
        pass

    @abstractmethod
    def _calculate_downstream_task_performance(self, **kwargs) -> DownstreamTaskPerformanceDataModel:
        """
        All downstreams tasks should fill out this method.
        Uses whatever values that's been tracked & updated for the downstream task and calculate the metrics.
        Reset any values necessary (ie instance vars, class vars, etc.) for new eval on the next dataset.

        Parameters:
            whatever needed.
        """
        pass
