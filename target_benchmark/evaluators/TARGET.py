import logging
import os
import shutil
import time
from datetime import datetime
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
from qdrant_client import QdrantClient, models

from target_benchmark.dataset_loaders import HFDatasetLoader, Text2SQLDatasetLoader
from target_benchmark.dataset_loaders.AbsDatasetLoader import AbsDatasetLoader
from target_benchmark.dataset_loaders.LoadersDataModels import (
    DatasetConfigDataModel,
    GenericDatasetConfigDataModel,
    HFDatasetConfigDataModel,
    Text2SQLDatasetConfigDataModel,
)
from target_benchmark.dataset_loaders.utils import get_dummy_table_of_format
from target_benchmark.dictionary_keys import (
    CONTEXT_COL_NAME,
    DATABASE_ID_COL_NAME,
    METADATA_DB_ID_KEY_NAME,
    METADATA_TABLE_ID_KEY_NAME,
    TABLE_COL_NAME,
    TABLE_ID_COL_NAME,
)
from target_benchmark.evaluators.utils import find_tasks
from target_benchmark.retrievers import (
    AbsCustomEmbeddingRetriever,
    AbsRetrieverBase,
    AbsStandardEmbeddingRetriever,
)
from target_benchmark.tasks import TableRetrievalTask, Text2SQLTask
from target_benchmark.tasks.AbsTask import AbsTask
from target_benchmark.tasks.TasksDataModels import (
    EmbeddingStatisticsDataModel,
    TaskResultsDataModel,
)


class TARGET:
    def __init__(
        self,
        downstream_tasks: Union[
            str,
            Tuple[str, Union[str, List[str]]],
            AbsTask,
            List[Union[str, Tuple[str, Union[str, List[str]]], AbsTask]],
        ] = None,
        persist_log: bool = True,
        log_file_path: str = None,
    ):
        """
        Pass in a list of task names for the evaluator to run. If no tasks are passed in, default table retrieval task will be created for running.

        Parameters:
            downstream_tasks (Union[str, Tuple[str, Union[str, List[str]]], AbsTask, List[Union[str, Tuple[str, Union[str, List[str]]], AbsTask]]], optional): tasks to perform. you can pass in either a single:
                str: name of the single task to run.
                Tuple[str, Union[str, List[str]]]: if you'd like more granular control, you can specify the task name mapped to a dictionary containing the dataset name mapped to the split to run evaluation on. we only support one split for each dataset. example inputs:
                ("Table Question Answering Task", ["fetaqa", "ottqa"])
                ("Text to SQL Task", "spider)
                AbsTask: a custom task. if a you want to run some task with a custom dataset that is not one of target's default datasets, you can first create the task object with the specified dataset configs, then simply pass the task object in here.
            OR you can pass in a list of containing multiple of these items. Be sure that any dataset is only mentioned once in your input.
            persist_log (bool, optional): whether to persist the log to a file or not.
            log_file_path (string, optional): the path to persis the log to. if none is provided, default to target_run_log_<current time>.txt
        """

        # set up a logger for target
        self.logger = self.setup_logger(
            persist_log=persist_log, log_file_path=log_file_path
        )
        self.logger.info("Logger for TARGET is set up!")

        self.logger.info("Starting to load the specified tasks...")
        self.tasks: Dict[str, AbsTask] = self.load_tasks(downstream_tasks)
        self.logger.info(
            f"Finished loading tasks! Tasks loaded: {list(self.tasks.keys())}"
        )

        self.logger.info("Started creating dataset information...")
        self.dataset_info: Dict[str, DatasetConfigDataModel] = self.create_dataset_info(
            self.tasks
        )
        self.logger.info(
            "Finished creating dataset config information. Finished setting up."
        )
        self.dataloaders = {}

    def load_tasks(
        self,
        downstream_tasks: Union[
            str,
            Tuple[str, Union[str, List[str]]],
            AbsTask,
            List[Union[str, Tuple[str, Union[str, List[str]]], AbsTask]],
        ] = None,
    ) -> Dict[str, AbsTask]:
        """
        Returns the task objects specified in the list of downstream tasks. If no tasks are specified, load the table retrieval task.

        Parameters:
            downstream_tasks (Union[str, Tuple[str, Union[str, List[str]]], AbsTask, List[Union[str, Tuple[str, Union[str, List[str]]], AbsTask]]], optional): tasks to perform. can be either a single:
                str: name of the single task to run.
                Tuple[str, Union[str, List[str]]]: if you'd like more granular control, you can specify the task name mapped to a dictionary containing the dataset name mapped to the split to run evaluation on. we only support one split for each dataset. example inputs:
                ("Table Question Answering Task", ["fetaqa", "ottqa"])
                ("Text to SQL Task", "spider-test")
                AbsTask: a custom task. if a you want to run some task with a custom dataset that is not one of target's default datasets, you can first create the task object with the specified dataset configs, then simply pass the task object in here.
            OR a list of containing multiple of these items. Be sure that any dataset is only mentioned once in your input.

        Returns:
            a dictionary mapping task names to task objects.
        """
        if not downstream_tasks:
            return {TableRetrievalTask.get_default_task_name(): TableRetrievalTask()}
        if not isinstance(downstream_tasks, List):
            downstream_tasks = [downstream_tasks]
        loaded_tasks = {}
        tasks_dict = find_tasks()
        for task in downstream_tasks:  # iterate through each of the passed in tasks
            if isinstance(task, str):  # if passed in task name
                if (
                    task in tasks_dict
                ):  # check if task name exists in the available target tasks
                    task_class = tasks_dict[task]
                    task_default_name = task_class.get_default_task_name()
                    if (
                        task_default_name in loaded_tasks
                    ):  # warning for overwriting due to duplicate task names
                        self.logger.error(
                            f"task by name {task_default_name} already loaded. this action will overwrite the previously loaded task. be careful as this may not be intended behavior!"
                        )
                    loaded_tasks[
                        task_default_name
                    ] = task_class()  # create a default instance of that task class
                else:
                    self.logger.warning(
                        f"task named {task} doesn't exist. please double check your input values. skipping this task..."
                    )
            elif isinstance(
                task, Tuple
            ):  # otherwise if it's an instance of tuple, user specified task and datasets to run
                task_name, task_dataset_names = task
                # validating the dataset specified has the correct type
                if not isinstance(task_dataset_names, (str, List)):
                    wrong_type = type(task_dataset_names)
                    error_msg = f"task dataset info passed in for task {task_name} is not a string or a list of strings, but instead {wrong_type}. Please double check inputs!"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                if isinstance(task_dataset_names, str):
                    task_dataset_names = [task_dataset_names]
                # check that the task is one of the target default tasks
                if task_name in tasks_dict:
                    task_class = tasks_dict[task_name]
                    default_datasets = task_class._get_default_dataset_config()
                    needed_datasets = {}
                    # validating dataset names provided are a part of the default datasets for that class
                    for task_dataset_name in task_dataset_names:
                        if task_dataset_name not in default_datasets:
                            task_class_name = task_class.__name__
                            error_msg = f"provided dataset {task_dataset_name} is not one of the datasets available for task {task_name}. pls use `{task_class_name}._get_default_dataset_config()` to check what default datasets are available"
                            self.logger.error(error_msg)
                            raise ValueError(error_msg)
                        else:
                            needed_datasets[task_dataset_name] = default_datasets[
                                task_dataset_name
                            ]

                    task_default_name = task_class.get_default_task_name()
                    if task_default_name in loaded_tasks:
                        self.logger.error(
                            f"task by name {task_default_name} already loaded. this action will overwrite the previously loaded task. be careful as this may not be intended behavior!"
                        )
                    # create the task with the specified dataset configs
                    loaded_tasks[task_default_name] = task_class(
                        datasets_config=needed_datasets, overwrite_default_datasets=True
                    )
                else:
                    self.logger.warning(
                        f"task named {task_name} doesn't exist. please double check your input values. skipping this task..."
                    )
            elif isinstance(task, AbsTask):  # if it's an instance of the tasks classes
                task_name = task.get_task_name()
                if (
                    task_name in loaded_tasks
                ):  # warning for overwriting due to duplicate task names
                    self.logger.warning(
                        f"task by name {task_name} already loaded. this action will overwrite the previously loaded task. be careful as this may not be intended behavior!"
                    )
                # assign task to loaded tasks dictionary with the corresponding name
                loaded_tasks[task_name] = task
            else:
                wrong_type = type(task)
                error_msg = f"passed in an object {task} of type {wrong_type}. pls see documentation for the accepted types for the downstream tasks."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        return loaded_tasks

    def get_loaded_tasks(self) -> List[str]:
        """
        Getter function for all the loaded tasks.

        Returns:
            a list of task names of the loaded tasks. if no tasks are loaded, return an empty list.
        """
        if self.tasks is not None:
            return list(self.tasks.keys())
        else:
            return []

    def create_dataset_info(
        self, tasks: Dict[str, AbsTask]
    ) -> Dict[str, DatasetConfigDataModel]:
        """
        After loading in the tasks, create the dataset information dictionary
        Parameters:
            tasks (Dict[str, AbsTask]): a dictionary mapping task names to tasks.

        Returns:
            a dictionary mapping dataset names to dataset configs.
        """
        eval_dataset_config = {}
        for _, task_object in tasks.items():
            dataset_config = task_object.get_dataset_config()
            for dataset_name, config in dataset_config.items():
                if dataset_name not in eval_dataset_config:
                    eval_dataset_config[dataset_name] = config
        return eval_dataset_config

    def create_dataloaders(
        self,
        dataset_config: Dict[str, DatasetConfigDataModel],
        split: Literal["test", "train", "validation"] = "test",
    ) -> Dict[str, AbsDatasetLoader]:
        """
        Create the dataloaders according to the dataset config. Doesn't load the data until the tasks are actually being run.

        Parameters:
            dataset_config (Dict[str, DatasetConfigDataModel]): A dictionary mapping dataset names to the config data models.

        Returns:
            a dictionary of dataloaders mapping dataset names to dataloader objects.
        """
        eval_dataloaders = {}
        for dataset_name, config in dataset_config.items():
            # if the dataset with the same name and the same split already exists, no need to do anything
            if (
                dataset_name in self.dataloaders
                and config.split == self.dataloaders[dataset_name].split
            ):
                continue
            config.split = split
            if isinstance(config, Text2SQLDatasetConfigDataModel):
                eval_dataloaders[dataset_name] = Text2SQLDatasetLoader(
                    **config.model_dump()
                )
            elif isinstance(config, HFDatasetConfigDataModel):
                eval_dataloaders[dataset_name] = HFDatasetLoader(**config.model_dump())
            elif isinstance(config, GenericDatasetConfigDataModel):
                eval_dataloaders[dataset_name] = GenericDatasetConfigDataModel(
                    **config.model_dump()
                )
            else:
                self.logger.warning(
                    f"The dataset config passed in for {dataset_name} is not a valid dataset config data model. Skipping..."
                )
        return eval_dataloaders

    def load_datasets_for_task(
        self,
        dataset_names: List[str],
    ) -> Dict[str, AbsDatasetLoader]:
        """
        Load the datasets through the dataloaders for a task.

        Parameters:
            dataset_names (List[str]): a list of names for the datasets to load.

        Return:
            a dictionary mapping dataset name to the corresponding dataloader object.
        """
        dataloaders_for_task = {}
        for dataset_name in dataset_names:
            if dataset_name not in self.dataloaders:
                self.logger.warning(
                    f"Dataset {dataset_name} was not included at task creation. Please double check if you've inputted the dataset config correctly!"
                )
            else:
                dataloader = self.dataloaders[dataset_name]
                dataloader.load()
                dataloaders_for_task[dataset_name] = dataloader
        return dataloaders_for_task

    def setup_logger(
        self, persist_log: bool = True, log_file_path: str = None
    ) -> logging.Logger:
        """
        set up a logger for logging all evaluator actions.
        Parameters:
            persist_log (bool, optional): whether to persist the logs or not.
            log_file_path (string, optional): the path to persis the log to. if none is provided, default to logs/target_run_log_<current time>.txt

        Returns:
            a logger with the correct file handling set up.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        if persist_log:
            if not log_file_path:
                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file_path = os.path.join("logs", f"./target_run_log_{time_str}.txt")
            log_dir = os.path.dirname(log_file_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            # Create file handler which logs even debug messages
            fh = logging.FileHandler(log_file_path)
            fh.setLevel(logging.DEBUG)

            # Create formatter and add it to the handler
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            fh.setFormatter(formatter)

            # Add the handler to the logger
            logger.addHandler(fh)
        return logger

    def embed_with_standardized_embeddings(
        self,
        retriever: AbsStandardEmbeddingRetriever,
        dataset_name: str,
        client: QdrantClient,
    ) -> Tuple[float, float]:
        """
        Create embeddings with retriever inheriting from `AbsStandardizedEmbeddingRetriever`. Includes an in-memory vector database for storage support. Should only be used after the dataloaders have been correctly loaded.

        Parameters:
            retriever (AbsStandardizedEmbeddingRetriever): the retriever object
            dataset_name (str): name of the dataset to embed
            client (QdrantClient): an in memory qdrant vector db
        """
        vec_size = len(
            retriever.embed_corpus(
                dataset_name,
                {
                    DATABASE_ID_COL_NAME: 1,
                    TABLE_ID_COL_NAME: "",
                    TABLE_COL_NAME: get_dummy_table_of_format(
                        retriever.get_expected_corpus_format()
                    ),
                    CONTEXT_COL_NAME: {},
                },
            )
        )
        client.delete_collection(collection_name=dataset_name)
        client.create_collection(
            collection_name=dataset_name,
            vectors_config=models.VectorParams(
                size=vec_size, distance=models.Distance.COSINE
            ),
        )
        cur_dataloader = self.dataloaders[dataset_name]
        vectors = []
        metadata = []
        start_time = time.process_time()
        for entry in cur_dataloader.convert_corpus_table_to(
            retriever.get_expected_corpus_format()
        ):  # TODO: support batching
            entry = {key: value[0] for key, value in entry.items()}
            table_embedding = retriever.embed_corpus(dataset_name, entry)
            vectors.append(table_embedding)
            metadata.append(
                {
                    METADATA_TABLE_ID_KEY_NAME: entry[TABLE_ID_COL_NAME],
                    METADATA_DB_ID_KEY_NAME: entry[DATABASE_ID_COL_NAME],
                }
            )
        end_time = time.process_time()
        duration = end_time - start_time
        vectors = np.array(vectors)
        embedding_size = vectors.nbytes

        client.upload_collection(
            collection_name=dataset_name,
            vectors=vectors,
            payload=metadata,
        )
        return duration, embedding_size

    def embed_with_custom_embeddings(
        self,
        retriever: AbsCustomEmbeddingRetriever,
        dataset_name: str,
        batch_size: int,
    ) -> Tuple[float, float]:
        start_disk_usage = shutil.disk_usage("/").used
        start_time = time.process_time()

        retriever.embed_corpus(
            dataset_name,
            self.dataloaders[dataset_name].convert_corpus_table_to(
                retriever.get_expected_corpus_format(), batch_size
            ),
        )
        end_time = time.process_time()
        duration = end_time - start_time
        end_disk_usage = shutil.disk_usage("/").used
        embedding_size = (end_disk_usage - start_disk_usage) * 1.0 / 1_000_000
        return duration, embedding_size

    def run(
        self,
        retriever: AbsRetrieverBase,
        split: Literal["test", "train", "validation"] = "test",
        batch_size: int = 1,
        top_k: int = 5,
        **kwargs,
    ) -> Dict[str, TaskResultsDataModel]:
        """
        Call this function to run the tasks! Woohoo!!!

        Parameters:
            retriever (AbsRetrieverBase): a retriever that either inherits from AbsStandardEmbeddingRetriever or AbsCustomEmbeddingRetriver.
            split (Literal["test", "train", "validation"], optional): split of data to run the tasks on.
            batch_size (int, optional): number of queries / number of tables to pass to the retriever at once.
            top_k (int, optional): top k tables to retrieve.
        """
        self.logger.info("Started creating data loader objects...")
        self.dataloaders: Dict[
            str, AbsDatasetLoader
        ] = self.dataloaders | self.create_dataloaders(self.dataset_info, split)

        all_results = {}
        loaded_datasets = set()
        embedding_stats = {}
        if isinstance(retriever, AbsStandardEmbeddingRetriever):
            standardized = True
            client = QdrantClient(":memory:")
        elif isinstance(retriever, AbsCustomEmbeddingRetriever):
            standardized = False
            client = None
        else:
            self.logger.warning(
                "the retriever passed in is in the wrong format! it doens't inherit from any target retriever classes. "
            )

        for task_name, task in self.tasks.items():
            self.logger.info(f"Start running {task_name}...")
            self.logger.info("Start checking for new corpus to embed...")
            # load the datasets needed
            dataset_names = task.get_dataset_config().keys()
            dataloaders_for_task = self.load_datasets_for_task(
                dataset_names=dataset_names
            )
            if isinstance(task, Text2SQLTask):
                for name, loader in dataloaders_for_task.items():
                    assert isinstance(
                        loader, Text2SQLDatasetLoader
                    ), f"data loader for dataset {name} is not a text to sql dataset."
                task.setup_database_dirs(dataloaders_for_task)

            # call embed corpus on the retriever to embed/preprocess the tables

            for dataset_name in dataset_names:
                if dataset_name not in loaded_datasets:
                    size_of_corpus = self.dataloaders[dataset_name].get_corpus_size()
                    duration, embedding_size = -1.0, -1.0
                    if standardized:
                        (
                            duration,
                            embedding_size,
                        ) = self.embed_with_standardized_embeddings(
                            retriever, dataset_name, client
                        )
                    else:
                        duration, embedding_size = self.embed_with_custom_embeddings(
                            retriever, dataset_name, batch_size
                        )
                    loaded_datasets.add(dataset_name)

                    # create embedding statistics data object to record latency & size of embedding
                    embedding_stats[dataset_name] = EmbeddingStatisticsDataModel(
                        embedding_creation_duration=round(duration, 5),
                        avg_embedding_creation_duration=round(duration / size_of_corpus, 5),
                        embedding_size=round(embedding_size, 5),
                        avg_embedding_size=round(embedding_size / size_of_corpus, 5),
                    )

            self.logger.info("Finished embedding all new corpus!")

            # run the task!
            task_result = task.task_run(
                retriever=retriever,
                dataset_loaders=dataloaders_for_task,
                logger=self.logger,
                batch_size=batch_size,
                top_k=top_k,
                client=client,
                **kwargs,
            )

            # add the embedding duration & sizes statistics to the results
            for dataset_name, results in task_result.items():
                results.embedding_statistics = embedding_stats[dataset_name]

            all_results[task_name] = task_result
        self.logger.info("Finished running all tasks!")
        return all_results
