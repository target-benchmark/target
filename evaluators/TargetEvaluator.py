from dataset_loaders.AbsTargetDatasetLoader import AbsTargetDatasetLoader
from dataset_loaders.HFDatasetLoader import HFDatasetLoader
from dataset_loaders.GenericDatasetLoader import GenericDatasetLoader
from retrievers import (
    AbsTargetCustomEmbeddingRetriever,
    AbsTargetStandardizedEmbeddingRetriever,
)


class TargetEvaluator:
    def __init__(self, downstream_task_names: str | list[str] = None):
        """
        Pass in a list of task names for the evaluator to run. Table retrieval task will always be run no matter the downstream tasks specified because you need the results of the table retrieval anyways for all target downstream tasks.
        Parameters:
            downstream_task_names (str | list[str], optional): name of the tasks. by default the retrieval task
        """
        if downstream_task_names is None:
            downstream_task_names = []
        if isinstance(downstream_task_names, str):
            downstream_task_names = [downstream_task_names]
        self.downstream_task_names: list[str] = downstream_task_names
        self.datasets_info: list[dict[str, object]] = {}
        self.dataloaders: dict[str, AbsTargetDatasetLoader] = {}
        self.load_tasks()
        self.create_dataloaders()

    def load_tasks(self):
        """
        load the task objects specified in the list of downstream tasks. plus the table retrieval task.
        it should also create the self.datasets_info list of dictionaries. each dictionary should have the following keys:
            dataset_name: name of the dataset
            is_hf: if the dataset is a dataset on the huggingface hub or generic dataset
            if is_hf is true:
                corpus_path: path to corpus
                queries_path: path to queries
            else:
                dataset_path: path to the structured dataset
        """
        pass

    def create_dataloaders(self):
        for dataset_info in self.datasets_info:
            if dataset_info["is_hf"]:
                self.dataloaders[dataset_info["dataset_name"]] = HFDatasetLoader(
                    **dataset_info
                )
            else:
                self.dataloaders[dataset_info["dataset_name"]] = GenericDatasetLoader(
                    **dataset_info
                )
