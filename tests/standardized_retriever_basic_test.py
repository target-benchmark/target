import unittest
from unittest.mock import patch, MagicMock
import os
from dataset_loaders.LoadersDataModels import QueryForTasksDataModel
from tasks.TableRetrievalTask import TableRetrievalTask
from tasks.TasksDataModels import *
from dataset_loaders.HFDatasetLoader import HFDatasetLoader
from dataset_loaders.TargetDatasetConfig import HFDatasetConfigDataModel
from dataset_loaders.TargetDatasetConfig import (
    DEFAULT_WIKITQ_DATASET_CONFIG,
    DEFAULT_FETAQA_DATASET_CONFIG,
)
from retrievers import AbsStandardizedEmbeddingRetriever
from retrievers.RetrieversDataModels import RetrievalResultDataModel

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Get a logger
logger = logging.getLogger(__name__)

#TODO: Find a tool that works with this.

class TestTaskRunWithStdRetriever(unittest.TestCase):

    def setUp(self):
        self.retr_task = TableRetrievalTask()
        self.retriever = MagicMock()
        self.mock_retriever.__class__ = AbsStandardizedEmbeddingRetriever
        self.mock_retriever.retrieve_batch.return_value = [
            RetrievalResultDataModel(
                dataset_name="fetaqa",
                query_id=1,
                retrieval_results=["Table1", "Table2"],
            ),
            RetrievalResultDataModel(
                dataset_name="fetaqa",
                query_id=2,
                retrieval_results=["Table3", "Table4"],
            ),
        ]
        self.mock_dataset_loader = MagicMock()
        self.mock_dataset_loader.get_queries_for_task.side_effect = (
            lambda splits, batch_size: iter(
                [
                    [
                        QueryForTasksDataModel(
                            query_id=1,
                            query="Test query",
                            answer="Test answer",
                            table_id="Table1",
                            database_id=0,
                        ),
                        QueryForTasksDataModel(
                            query_id=2,
                            query="Test query 2",
                            answer="Test answer 2",
                            table_id="Table5",
                            database_id=0,
                        ),
                    ]
                ]
            )
        )

    def test_basic_task_run(self):

        results = self.retr_task.task_run(
            retriever=self.mock_retriever,
            dataset_loaders={"fetaqa": self.mock_dataset_loader},
            logger=logger,
            batch_size=1,
            splits="test",
            top_k=2,
        )
        self.mock_retriever.retrieve_batch.assert_called_once_with(
            queries=[
                QueryForTasksDataModel(
                    query_id=1,
                    query="Test query",
                    answer="Test answer",
                    table_id="Table1",
                    database_id=0,
                ),
                QueryForTasksDataModel(
                    query_id=2,
                    query="Test query 2",
                    answer="Test answer 2",
                    table_id="Table5",
                    database_id=0,
                ),
            ],
            dataset_name="fetaqa",
            top_k=2,
        )

    def test_tp_updating(self):
        results = self.retr_task.task_run(
            retriever=self.mock_retriever,
            dataset_loaders={"fetaqa": self.mock_dataset_loader},
            logger=logger,
            batch_size=1,
            splits="test",
            top_k=2,
        )

        fetaqa_results = results["fetaqa"]
        retr_perf = fetaqa_results.retrieval_performance
        downs_perf = fetaqa_results.downstream_task_performance
        self.assertIsInstance(fetaqa_results, TaskResultsDataModel)
        self.assertIsInstance(retr_perf, RetrievalPerformanceDataModel)
        self.assertIsInstance(downs_perf, DownstreamTaskPerformanceDataModel)
        self.assertEqual(
            retr_perf.model_dump(),
            {"k": 2, "accuracy": 0.5, "precision": None, "recall": None},
        )
        self.assertEqual(downs_perf.model_dump(), {"task_name": None})
        self.assertEqual(self.retr_task.true_positive, 0)
        self.assertEqual(self.retr_task.total_queries_processed, 0)

    def test_custom_dataset_config(self):

        new_task = TableRetrievalTask(
            datasets_config={
                "wikitq": {
                    "dataset_name": "wikitq",
                    "hf_corpus_dataset_path": "target-benchmark/wikitq-corpus",
                    "hf_queries_dataset_path": "target-benchmark/wikitq-queries",
                }
            }
        )
        constructed_task = new_task.get_dataset_config()
        self.assertEqual(
            constructed_task["wikitq"].model_dump(),
            DEFAULT_WIKITQ_DATASET_CONFIG.model_dump(),
        )
        self.assertEqual(
            constructed_task["fetaqa"].model_dump(),
            DEFAULT_FETAQA_DATASET_CONFIG.model_dump(),
        )



if __name__ == "__main__":
    unittest.main()