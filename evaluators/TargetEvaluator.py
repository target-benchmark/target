from dataset_loaders.AbsTargetDatasetLoader import AbsTargetDatasetLoader
from dataset_loaders.HFDatasetLoader import HFDatasetLoader
from dataset_loaders.GenericDatasetLoader import GenericDatasetLoader
from dataset_loaders.LoadersDataModels import (
    DatasetConfigDataModel,
    GenericDatasetConfigDataModel,
    HFDatasetConfigDataModel,
)
from retrievers import (
    AbsTargetCustomEmbeddingRetriver,
    AbsTargetStandardizedEmbeddingRetriever,
)
from tasks.AbsTargetTask import AbsTargetTask
from tasks.TableRetrievalTask import TableRetrievalTask

from evaluators.utils import find_tasks
from datetime import datetime

import logging


class TargetEvaluator:
    def __init__(
        self,
        downstream_task_names: str | list[str] = [],
        downstream_task_objects: AbsTargetTask | list[AbsTargetTask] = [],
        persist_log: bool = True,
        log_file_path: str = None,
    ):
        """
        Pass in a list of task names for the evaluator to run. If no tasks are passed in, default table retrieval task will be created for running.

        Parameters:
            downstream_task_names (str | list[str], optional): name of the tasks.
            downstream_task_objects (AbsTargetTask | list[AbsTargetTask], optional): a list of custom tasks. for example, if a user wants to run some task with a custom dataset, they can first create the task object with the specified dataset configs, then simply pass the task object in here.
            persist_log (bool, optional): whether to persist the log to a file or not.
            log_file_path (string, optional): the path to persis the log to. if none is provided, default to target_run_log_<current time>.txt
        """

        # set up a logger for target
        self.logger = self.setup_logger(
            persist_log=persist_log, log_file_path=log_file_path
        )
        self.logger.info("Logger for TARGET is set up!")

        self.logger.info("Starting to load the specified tasks...")
        self.tasks: dict[str, AbsTargetTask] = self.load_tasks()
        self.logger.info(f"Finished loading tasks! Tasks loaded: {self.tasks.keys()}")

        self.logger.info("Started creating dataset information...")
        self.dataset_info = self.create_dataset_info(self.tasks)
        self.logger.info("Finished creating dataset config information.")

        self.logger.info("Started creating data loader objects...")
        self.dataloaders = self.create_dataloaders(self.dataset_info)
        self.logger.info("Finished creating dataset loaders. Finished setting up.")

    def load_tasks(
        self,
        downstream_task_names: str | list[str],
        downstream_task_objects: AbsTargetTask | list[AbsTargetTask],
    ) -> dict[str, AbsTargetTask]:
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
            return {TableRetrievalTask.get_default_task_name: TableRetrievalTask()}
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
            if not isinstance(task_obj):
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

    def create_dataset_info(
        self, tasks: dict[str, AbsTargetTask]
    ) -> dict[str, DatasetConfigDataModel]:
        """
        After loading in the tasks, create the dataset information dictionary
        Parameters:
            tasks (dict[str, AbsTargetTask]): a dictionary mapping task names to tasks.

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
    ) -> dict[str, AbsTargetDatasetLoader]:
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
                log_file_path = f"./target_run_log_{time_str}.txt"
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
