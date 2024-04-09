from abc import ABC, abstractmethod

from generators.AbsTargetGenerator import AbsTargetGenerator
from generators.DefaultTargetGenerator import DefaultTargetGenerator

from retrievers.AbsTargetRetrieverBase import AbsTargetRetrieverBase
from retrievers.AbsTargetCustomEmbeddingRetriver import AbsTargetCustomEmbeddingRetriver as CustomEmbRetr
from retrievers.AbsTargetStandardizedEmbeddingRetriever import AbsTargetStandardizedEmbeddingRetriever as StandardizedEmbRetr

from dataset_loaders.AbsTargetDatasetLoader import AbsTargetDatasetLoader
from dataset_loaders.LoadersDataModels import QueryForTasksDataModel
from logging import Logger
from dictionary_keys import *

class AbsTargetTask(ABC):

    def __init__(
        self,
        task_name: str,
        datasets_config: dict[str, dict[str, str]] = None,
        overwrite_default_datasets: bool = False,
        task_generator: AbsTargetGenerator = DefaultTargetGenerator,
        **kwargs
    ):
        '''
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
        '''
        self.task_name: str = task_name
        self.dataset_config: dict[str, dict[str, str]] = self._construct_dataset_config(datasets_config, overwrite_default_datasets)
        self.task_generator: AbsTargetGenerator = task_generator

    def _construct_dataset_config(
        self, 
        datasets_config: dict[str, dict[str, str]],
        overwrite_default_datasets: bool,
    ) -> dict[str, dict[str, str]]:
        '''
        builds the dataset config according to the user inputted dataset config (if any) and the default for the class.

        Parameters:
            datasets_config (dict[str, dict[str, str]]): user inputted datasets config dictionary.
            overwrite_default_datasets (bool): whether to overwrite the default datasets or not if the same name dataset is provided.
        '''
        constructed_config: dict[str, dict[str, str]] = self._get_default_dataset_config()
        if datasets_config is not None:
            for key, value in datasets_config.items():

                if key not in constructed_config or overwrite_default_datasets:
                    assert(
                        (HF_DATASET_CONFIG_CORPUS_FIELD in value and HF_DATASET_CONFIG_QUERIES_FIELD in value) 
                        or GENERIC_DATASET_CONFIG_FIELD in value
                    ), f'user inputted data config for {key} is missing fields! (current config: {value})'
                    constructed_config[key] = value
                    
        return constructed_config
        
        

    @abstractmethod
    def _get_default_dataset_config(self) -> dict[str, dict[str, str]]:
        '''
        Returns the default dataset config for the class. MUST be implemented by any inherited task class.
        '''
        pass
        
    
    def task_run(
        self,
        retriever: AbsTargetRetrieverBase,
        dataset_loaders: dict[str, AbsTargetDatasetLoader],
        logger: Logger,
        batch_size: int = 64,
        splits: str | list[str] = 'test',
        top_k: int = 5
    ) -> dict:
        assert(self.dataset_config.keys() <= dataset_loaders.keys()), f'task\'s dataset config is not a subset of the dataset loaders passed in! \ntask dataset config: {self.dataset_config.keys()}\ndataset loaders passed in: {dataset_loaders.keys()}'

        assert(isinstance(retriever, CustomEmbRetr) or isinstance(retriever, StandardizedEmbRetr)), f'the passed in retriever doesn\'t correctly inherit from the standardized or custom retriever classes!'



        task_results = {}

        logger.info(f'start task {self.task_name}')
        
        for dataset_name, dataset_loader in dataset_loaders.items():
            logger.info(f'running task on dataset {dataset_name}')

            for query_batch in dataset_loader.get_queries_for_task(splits, batch_size):
                id_to_query = self._query_info_to_id_queries(query_batch)
                retrieved_tables, retrieval_performance = self._get_retrieval_results(retriever, id_to_query, dataset_name, top_k)
                id_to_answer = self._query_info_to_answers(query_batch)
                downstream_task_performance = self._get_downstream_task_results(retrieved_tables, id_to_answer)
                task_results[dataset_name] = {
                    'retrieval_performance': retrieval_performance,
                    'downstream_task_performance': downstream_task_performance
                }
        logger.info(f'finished running task {self.task_name}')
        return task_results


    def _query_info_to_id_queries(self, batch: list[QueryForTasksDataModel]) -> dict[str, str]:
        id_to_query = {}
        for query_info_dict in batch:
            id_to_query[query_info_dict.query_id] = query_info_dict.query
        return id_to_query
    
    def _query_info_to_answers(self, batch: list[QueryForTasksDataModel]) -> dict[str, str]:
        id_to_answer = {}
        for query_info_dict in batch:
            id_to_answer[query_info_dict.query_id] = query_info_dict.answer
        return id_to_answer

    def _get_retrieval_results(
            self, 
            retriever: AbsTargetRetrieverBase,
            id_to_query: dict[str, str],
            dataset_name: str,
            top_k: int
    ) -> tuple[dict[str, str], dict[str, object]]:
        # TODO: metrics
        is_standard = True
        if isinstance(retriever, CustomEmbRetr):
            is_standard = False
        if is_standard:
            # TODO: figure out what to do with embedding here
            # retreival_results = retriever.retrieve_batch(corpus_embedding=)
            pass
        else:
            retrieval_results = retriever.retrieve_batch(queries=id_to_query, dataset_name=dataset_name, top_k=top_k)
        
        return retrieval_results, {}
    
    @abstractmethod
    def _get_downstream_task_results(
        self,
        retrieval_results: dict[str, list[str]],
        id_to_answer: dict[str, str]
    ) -> dict[str, object]:
        '''
        TODO: how to pass through the tables? nested arrays, etc
        All downstreams tasks should fill out this method. ideally uses the retrieval results to generate the downstream answer, and return the performance of the downstream generation.
        '''
        pass