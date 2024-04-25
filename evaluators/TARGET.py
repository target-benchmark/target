from dataset_loaders.AbsDatasetLoader import AbsDatasetLoader
from dataset_loaders.HFDatasetLoader import HFDatasetLoader
from dataset_loaders.GenericDatasetLoader import GenericDatasetLoader
from dataset_loaders.LoadersDataModels import (
    DatasetConfigDataModel,
    GenericDatasetConfigDataModel,
    HFDatasetConfigDataModel,
)
from retrievers import (
    AbsRetrieverBase,
    AbsCustomEmbeddingRetriever,
    AbsStandardizedEmbeddingRetriever,
)
from tasks.AbsTask import AbsTask
from tasks.TableRetrievalTask import TableRetrievalTask
from tasks.TasksDataModels import TaskResultsDataModel
import os
from evaluators.utils import find_tasks
from datetime import datetime

import logging


class TARGET:
    def __init__(
        self,
        downstream_task_names: str | list[str] = [],
        downstream_task_objects: AbsTask | list[AbsTask] = [],
        persist_log: bool = True,
        log_file_path: str = None,
    ):
        """
        Pass in a list of task names for the evaluator to run. If no tasks are passed in, default table retrieval task will be created for running.

        Parameters:
            downstream_task_names (str | list[str], optional): name of the tasks.
            downstream_task_objects (AbsTask | list[AbsTask], optional): a list of custom tasks. for example, if a user wants to run some task with a custom dataset, they can first create the task object with the specified dataset configs, then simply pass the task object in here.
            persist_log (bool, optional): whether to persist the log to a file or not.
            log_file_path (string, optional): the path to persis the log to. if none is provided, default to target_run_log_<current time>.txt
        """

        # set up a logger for target
        self.logger = self.setup_logger(
            persist_log=persist_log, log_file_path=log_file_path
        )
        self.logger.info("Logger for TARGET is set up!")

        self.logger.info("Starting to load the specified tasks...")
        self.tasks: dict[str, AbsTask] = self.load_tasks(
            downstream_task_names, downstream_task_objects
        )
        self.logger.info(f"Finished loading tasks! Tasks loaded: {self.tasks.keys()}")

        self.logger.info("Started creating dataset information...")
        self.dataset_info: dict[str, DatasetConfigDataModel] = self.create_dataset_info(self.tasks)
        self.logger.info("Finished creating dataset config information.")

        self.logger.info("Started creating data loader objects...")
        self.dataloaders: dict[str, AbsDatasetLoader] = self.create_dataloaders(self.dataset_info)
        self.logger.info("Finished creating dataset loaders. Finished setting up.")

    def load_tasks(
        self,
        downstream_task_names: str | list[str],
        downstream_task_objects: AbsTask | list[AbsTask],
    ) -> dict[str, AbsTask]:
        """
        Returns the task objects specified in the list of downstream tasks. If no tasks are specified, load the table retrieval task.

        Parameters:
            downstream_task_names (str | list[str]): list of default tasks names for loading default tasks.
            downstream_task_objects: list of created task objects.

        Returns:
            a dictionary mapping task names to task objects.
        """
        if not isinstance(downstream_task_names, list):
            downstream_task_names = [downstream_task_names]
        if not isinstance(downstream_task_objects, list):
            downstream_task_objects = [downstream_task_objects]
        if len(downstream_task_names) + len(downstream_task_objects) == 0:
            return {TableRetrievalTask.get_default_task_name(): TableRetrievalTask()}
        loaded_tasks = {}
        tasks_dict = find_tasks()
        for task_name in downstream_task_names:
            if task_name in tasks_dict:
                task_class = tasks_dict[task_name]
                task_default_name = task_class.get_default_task_name()
                if task_default_name in loaded_tasks:
                    self.logger.warning(
                        f"task by name {task_default_name} already loaded. this action will overwrite the previously loaded task. be careful as this may not be intended behavior!"
                    )
                loaded_tasks[task_default_name] = task_class()
            else:
                self.logger.warning(
                    f"task named {task_name} doesn't exist. please double check your input values. skipping this task..."
                )
        for task_obj in downstream_task_objects:
            if not isinstance(task_obj, AbsTask):
                self.logger.warning(
                    f"task {task_obj} is not a valid task object. Skipping..."
                )
                continue
            task_name = task_obj.get_task_name()
            if task_name in loaded_tasks:
                self.logger.warning(
                    f"task by name {task_default_name} already loaded. this action will overwrite the previously loaded task. be careful as this may not be intended behavior!"
                )
            loaded_tasks[task_default_name] = task_class()
        return loaded_tasks

    def get_loaded_tasks(self) -> list[str]:
        '''
        Getter function for all the loaded tasks.
        
        Returns:
            a list of task names of the loaded tasks. if no tasks are loaded, return an empty list.
        '''
        if self.tasks != None:
            return list(self.tasks.keys())
        else:
            return []

    def create_dataset_info(
        self, tasks: dict[str, AbsTask]
    ) -> dict[str, DatasetConfigDataModel]:
        """
        After loading in the tasks, create the dataset information dictionary
        Parameters:
            tasks (dict[str, AbsTask]): a dictionary mapping task names to tasks.

        Returns:
            a dictionary mapping dataset names to dataset configs.
        """
        eval_dataset_config = {}
        for task_name, task_object in tasks.items():
            dataset_config = task_object.get_dataset_config()
            for dataset_name, config in dataset_config.items():
                if dataset_name not in eval_dataset_config:
                    eval_dataset_config[dataset_name] = config
        return eval_dataset_config

    def create_dataloaders(
        self, dataset_config: dict[str, DatasetConfigDataModel]
    ) -> dict[str, AbsDatasetLoader]:
        """
        Create the dataloaders according to the dataset config. Doesn't load the data until the tasks are actually being run.

        Parameters:
            dataset_config (dict[str, DatasetConfigDataModel]): A dictionary mapping dataset names to the config data models.

        Returns:
            a dictionary of dataloaders mapping dataset names to dataloader objects.
        """
        eval_dataloaders = {}
        for dataset_name, config in dataset_config.items():
            if isinstance(config, HFDatasetConfigDataModel):
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
        dataset_names: list[str],
        splits: str | list[str] = "test",
    ) -> dict[str, AbsDatasetLoader]:
        """
        Load the datasets through the dataloaders for a task.

        Parameters:
            dataset_names (list[str]): a list of names for the datasets to load.
            splits (str | list[str], optional): a single split or a list of splits.

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
                dataloader.load(splits=splits)
                dataloaders_for_task[dataset_name] = dataloader
        return dataloaders_for_task

    def setup_logger(
        self, persist_log: bool = True, log_file_path: str = None
    ) -> logging.Logger:
        """
        set up a logger for logging all evaluator actions.
        Parameters:
            persist_log (bool, optional): whether to persist the log to a file or not.
            log_file_path (string, optional): the path to persis the log to. if none is provided, default to target_run_log_<current time>.txt

        Returns:
            a logger with the correct file handling set up.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        if persist_log:
            if not log_file_path:
                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file_path = os.path.join("logs", f"./target_run_log_{time_str}.txt")
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

    def run(
        self,
        retriever: AbsRetrieverBase,
        splits: str | list[str] = "test",
        batch_size: int = 64,
        top_k: int = 5,
        **kwargs,
    ) -> dict[str, TaskResultsDataModel]:
        """
        Call this function to run the tasks! Woohoo!!!

        Parameters:
            retriever (AbsRetrieverBase): a retriever that either inherits from AbsStandardizedEmbeddingRetriever or AbsCustomEmbeddingRetriver.
            splits (str | list[str], optional): splits of data to run the tasks on.
            batch_size (int, optional): number of queries / number of tables to pass to the retriever at once. TODO: figure out if this is still relevant?
            top_k (int, optional): top k tables to retrieve.
        """
        all_results = {}
        loaded_datasets = set()
        for task_name, task in self.tasks.items():
            self.logger.info(f"Start running {task_name}...")
            self.logger.info("Start checking for new corpus to embed...")
            # load the datasets needed
            dataset_names = task.get_dataset_config().keys()
            dataloaders_for_task = self.load_datasets_for_task(
                dataset_names=dataset_names, splits=splits
            )

            # call embed corpus on the retriever to embed/preprocess the tables
            for dataset_name in dataset_names:
                if dataset_name not in loaded_datasets:
                    if isinstance(retriever, AbsStandardizedEmbeddingRetriever):
                        embeddings = retriever.embed_corpus(
                            dataset_name,
                            self.dataloaders[dataset_name].convert_corpus_table_to(
                                retriever.get_expected_corpus_format()
                            ),
                        )
                        # TODO: figure out what to do with the embedding
                    elif isinstance(retriever, AbsCustomEmbeddingRetriever):
                        retriever.embed_corpus(
                            dataset_name,
                            self.dataloaders[dataset_name].convert_corpus_table_to(
                                retriever.get_expected_corpus_format()
                            ),
                        )
                    else:
                        self.logger.warning(
                            "the retriever passed in is in the wrong format! it doens't inherit from any target retriever classes. "
                        )

            self.logger.info("Finished embedding all new corpus!")

            # run the task!
            task_result = task.task_run(
                retriever=retriever,
                dataset_loaders=dataloaders_for_task,
                logger=self.logger,
                batch_size=batch_size,
                splits=splits,
                top_k=top_k,
                **kwargs,
            )
            all_results[task_name] = task_result
        self.logger.info("Finished running all tasks!")
        return all_results
