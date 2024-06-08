from dataset_loaders.LoadersDataModels import (
    DatasetConfigDataModel,
    QueryForTasksDataModel,
)
from dataset_loaders.AbsDatasetLoader import AbsDatasetLoader
from dataset_loaders.TargetDatasetConfig import *

from generators import DefaultGenerator
from generators.AbsGenerator import AbsGenerator
from generators.GeneratorsDataModels import DownstreamGeneratedResultDataModel
from generators.GeneratorPrompts import TEXT2SQL_SYSTEM_PROMPT, TEXT2SQL_USER_PROMPT

from retrievers.RetrieversDataModels import RetrievalResultDataModel

from tasks.AbsTask import AbsTask
from tasks.TasksDataModels import (
    Text2SQLTaskPerformanceDataModel,
)


import duckdb
import gdown
import os
import shutil
import sqlite3
from typing import List, Dict, Union


class Text2SQLTask(AbsTask):

    AVAILABLE_METRICS = set(["execution_accuracy", "query_match"])
    DEFAULT_METRICS = set(["execution_accuracy", "query_match"])

    def __init__(
        self,
        datasets_config: Dict[str, Dict[str, str]] = None,
        overwrite_default_datasets: bool = False,
        task_generator: AbsGenerator = None,
        metrics: Union[str, List[str]] = list(DEFAULT_METRICS),
        **kwargs,
    ):
        if task_generator == None:
            task_generator = DefaultGenerator(
                system_message=TEXT2SQL_SYSTEM_PROMPT,
                user_message=TEXT2SQL_USER_PROMPT,
            )        
        super().__init__(
            task_name=self.get_default_task_name(),
            datasets_config=datasets_config,
            overwrite_default_datasets=overwrite_default_datasets,
            task_generator=task_generator,
            **kwargs,
        )
        # set up the evaluator objects
        if isinstance(metrics, str):
            metrics = [metrics]

        self.evals = ""
        for metric in metrics:
            if metric not in Text2SQLTask.AVAILABLE_METRICS:
                raise ValueError(
                    f"the metric {metric} is not one of the available metrics!"
                )
        if "execution_accuracy" in metrics and "query_match" in metrics:
            self.evals = "all"
        elif "execution_accuracy" in metrics:
            self.evals = "exec"
        elif "query_match" in metrics:
            self.evals = "match"


        self.pred_answers = []
        self.ref_answers = []

    @classmethod
    def get_default_task_name(cls) -> str:
        return "Text to SQL Task"

    @classmethod
    def get_available_metrics(cls) -> str:
        return str(cls.AVAILABLE_METRICS)

    def _get_default_dataset_config(self) -> Dict[str, DatasetConfigDataModel]:
        """
        Returns the default dataset config for the class. MUST be implemented by any inherited task class.
        """
        # TODO: add more things here. this is for testing. carl note 4/24
        return {
            # this is for testing!!
            DEFAULT_DUMMY_DATASET_CONFIG.dataset_name: DEFAULT_DUMMY_DATASET_CONFIG,
        }

    def _task_setup(self, *args, **kwargs) -> None:
        '''
        Do any necessary setup you need in here.
        '''
        if self.evals == "match":
            self.dbs = {}
            return
        self.dataset_loaders: Dict[str, AbsDatasetLoader] = kwargs.get('dataset_loaders', {})
        self.splits = kwargs.get("splits", [])
        if isinstance(self.splits, str):
            self.splits = [self.splits]
        self.table_id_to_database_id = {dataset_name: dataloader.get_table_id_to_database_id(self.splits) for dataset_name, dataloader in self.dataset_loaders.items()}
        for dataset_name, dataloader in self.dataset_loaders.items():
            if DEFAULT_SPIDER_DATASET_CONFIG.dataset_name in dataset_name:
                self._spider_setup()
    
    def _spider_setup(self):

        # set up and clean up for downloading spider dataset
        # dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data")
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        # spider_dir = os.path.join(dir_path, "spider")
        # if not os.path.exists(spider_dir):
        #     dest_path = os.path.join(dir_path, "spider.zip")
        #     gdown.download(DEFAULT_SPIDER_DATASET_CONFIG.aux["spider_zip_gdrive_url"], dest_path) # download zip
            
        #     shutil.unpack_archive(dest_path)
        #     shutil.rmtree(os.path.join(dir_path, "__MACOSX"))
        
        # self.path_to_spider = spider_dir
        self.spider_in_mem = duckdb.connect(database=":memory:", read_only=False)


    def _get_downstream_task_results(
        self,
        query_batch: List[QueryForTasksDataModel],
        retrieval_results: List[RetrievalResultDataModel],
        dataset_name: str,
    ) -> List[DownstreamGeneratedResultDataModel]:
        """
        currently just markdown reps of table strings
        All downstreams tasks should fill out this method. ideally uses the retrieval results to generate the downstream answer, and return the performance of the downstream generation.
        """
        downstream_task_results = []
        for query, result in zip(query_batch, retrieval_results):
            database_ids = []
            for table_id in result.retrieval_results:
                for split in self.splits:
                    if table_id in self.table_id_to_database_id[dataset_name][split]:
                        database_ids.append(self.table_id_to_database_id[dataset_name][split][table_id])
            generated_results = self.task_generator.generate(
                table_str=,
                query=query.query
            )
            downstream_task_results.append(
                DownstreamGeneratedResultDataModel(
                    dataset_name=dataset_name,
                    query_id=query.query_id,
                    generated_results=generated_results,
                    query=query.query
                )
            )

        return downstream_task_results

    def _update_downstream_task_metrics(
        self,
        query_batch: List[QueryForTasksDataModel],
        downstream_results: List[DownstreamGeneratedResultDataModel],
    ) -> None:
        """
        Update any values you keep track of for the downstream tasks.
        """
        self.pred_answers.extend(
            [
                downstream_answer.generated_results
                for downstream_answer in downstream_results
            ]
        )
        self.ref_answers.extend([query.answer for query in query_batch])

    def _calculate_downstream_task_performance(
        self, **kwargs
    ) -> Text2SQLTaskPerformanceDataModel:
        """
        Calculate downstream task metrics for the question answering task.
        """
        scores = {}
        

        result = Text2SQLTaskPerformanceDataModel(scores=scores)

        self.pred_answers = []
        self.ref_answers = []
        return result
