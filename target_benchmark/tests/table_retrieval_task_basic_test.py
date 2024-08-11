import unittest
from unittest.mock import MagicMock
from target_benchmark.tasks.TableRetrievalTask import TableRetrievalTask
from target_benchmark.tasks.TasksDataModels import *
from target_benchmark.dataset_loaders.TargetDatasetConfig import (
    DEFAULT_WIKITQ_DATASET_CONFIG,
    DEFAULT_FETAQA_DATASET_CONFIG,
)
from target_benchmark.retrievers.AbsCustomEmbeddingRetriever import (
    AbsCustomEmbeddingRetriever as CustomEmbRetr,
)
from target_benchmark.retrievers.RetrieversDataModels import RetrievalResultDataModel

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Get a logger
logger = logging.getLogger(__name__)


class TestTableRetriever(unittest.TestCase):

    def setUp(self):
        self.retr_task = TableRetrievalTask()
        self.mock_retriever = MagicMock()
        self.mock_retriever.__class__ = CustomEmbRetr
        self.mock_retriever.retrieve_batch.return_value = [
            RetrievalResultDataModel(
                dataset_name="fetaqa",
                query_id=1,
                retrieval_results=[(0, "Table1"), (0, "Table2")],
            ),
            RetrievalResultDataModel(
                dataset_name="fetaqa",
                query_id=2,
                retrieval_results=[(0, "Table3"), (0, "Table4")],
            ),
        ]
        self.mock_dataset_loader = MagicMock()
        self.mock_dataset_loader.get_queries_for_task.side_effect = (
            lambda batch_size: iter(
                [
                    {
                        "query_id": [1, 2],
                        "query": ["Test query", "Test query 2"],
                        "answer": ["Test answer", "Test answer 2"],
                        "table_id": ["Table1", "Table5"],
                        "database_id": [0, 0],
                    }
                ],
            )
        )

    def test_basic_task_run(self):

        results = self.retr_task.task_run(
            retriever=self.mock_retriever,
            dataset_loaders={"fetaqa": self.mock_dataset_loader},
            logger=logger,
            batch_size=1,
            top_k=2,
        )
        self.mock_retriever.retrieve_batch.assert_called_once_with(
            queries={
                "query_id": [1, 2],
                "query": ["Test query", "Test query 2"],
                "answer": ["Test answer", "Test answer 2"],
                "table_id": ["Table1", "Table5"],
                "database_id": [0, 0],
            },
            dataset_name="fetaqa",
            top_k=2,
        )

    def test_tp_updating(self):
        results = self.retr_task.task_run(
            retriever=self.mock_retriever,
            dataset_loaders={"fetaqa": self.mock_dataset_loader},
            logger=logger,
            batch_size=1,
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
        self.assertEqual(downs_perf.model_dump(), {"task_name": None, "scores": None})
        self.assertEqual(self.retr_task.true_positive, 0)
        self.assertEqual(self.retr_task.total_queries_processed, 0)

    def test_custom_dataset_config(self):

        new_task = TableRetrievalTask(
            datasets_config={
                "wikitq": {
                    "dataset_name": "wikitq",
                    "hf_corpus_dataset_path": "target-benchmark/wikitq-corpus",
                    "hf_queries_dataset_path": "target-benchmark/wikitq-queries",
                    "query_type": "Table Question Answering",
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

    def test_incomplete_and_wrong_dataset_config(self):

        hf_missing_corpus = {
            "wikitq": {
                "dataset_name": "wikitq",
                "hf_queries_dataset_path": "target-benchmark/wikitq-queries",
            }
        }
        hf_missing_queries = {
            "wikitq": {
                "dataset_name": "wikitq",
                "hf_corpus_dataset_path": "target-benchmark/wikitq-corpus",
            }
        }
        generic_missing_dataset_path = {
            "wikitq": {
                "dataset_name": "wikitq",
            }
        }
        with self.assertRaises(AssertionError):
            new_task = TableRetrievalTask(datasets_config=hf_missing_corpus)
            another_new_task = TableRetrievalTask(datasets_config=hf_missing_queries)
            third_new_task = TableRetrievalTask(
                datasets_config=generic_missing_dataset_path
            )


if __name__ == "__main__":
    unittest.main()
