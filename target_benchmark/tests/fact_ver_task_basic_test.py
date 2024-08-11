import unittest
from unittest.mock import MagicMock
from target_benchmark.tasks import FactVerificationTask
from target_benchmark.tasks.TasksDataModels import *

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
        self.fact_ver = FactVerificationTask()
        self.mock_retriever = MagicMock()
        self.mock_retriever.__class__ = CustomEmbRetr
        self.mock_retriever.retrieve_batch.return_value = [
            RetrievalResultDataModel(
                dataset_name="dummy-dataset",
                query_id=1,
                retrieval_results=[(0, "Table1"), (0, "Table3")],
            ),
            RetrievalResultDataModel(
                dataset_name="dummy-dataset",
                query_id=2,
                retrieval_results=[(0, "Table5"), (0, "Table4")],
            ),
        ]
        self.mock_dataset_loader = MagicMock()
        self.mock_dataset_loader.get_table_id_to_table.return_value = {
            (0, "Table1"): [["fact"], ["Anna is 10 years old."]],
            (0, "Table2"): [["fact"], ["I'm not a freak athlete."]],
            (0, "Table3"): [["fact"], ["Jaylen Brown went to Cal."]],
            (0, "Table4"): [["fact"], ["Minecraft is such a fun game."]],
            (0, "Table5"): [["fact"], ["Today's temperature is 22.1 celsius."]],
        }
        self.mock_dataset_loader.get_queries_for_task.side_effect = (
            lambda batch_size: iter(
                [
                    {
                        "query_id": [1, 2],
                        "query": [
                            "Jaylen Brown went to Stanford",
                            "Today's temperature is in the low 20s.",
                        ],
                        "answer": ["False", "True"],
                        "table_id": ["Table1", "Table5"],
                        "database_id": [0, 0],
                    }
                ],
            )
        )

    def test_fact_ver_task_run_key_error(self):

        with self.assertRaises(AssertionError):
            result = self.fact_ver.task_run(
                retriever=self.mock_retriever,
                dataset_loaders={"dummy": self.mock_dataset_loader},
                logger=logger,
                batch_size=1,
                top_k=2,
            )

    def test_fact_ver_task_run(self):

        results = self.fact_ver.task_run(
            retriever=self.mock_retriever,
            dataset_loaders={"tab-fact": self.mock_dataset_loader},
            logger=logger,
            batch_size=1,
            top_k=2,
        )
        self.mock_retriever.retrieve_batch.assert_called_once_with(
            queries={
                "query_id": [1, 2],
                "query": [
                    "Jaylen Brown went to Stanford",
                    "Today's temperature is in the low 20s.",
                ],
                "answer": ["False", "True"],
                "table_id": ["Table1", "Table5"],
                "database_id": [0, 0],
            },
            dataset_name="tab-fact",
            top_k=2,
        )
        self.assertDictEqual(
            results["tab-fact"].retrieval_performance.model_dump(),
            {"k": 2, "accuracy": 1.0, "precision": None, "recall": None},
        )
        self.assertDictEqual(
            results["tab-fact"].downstream_task_performance.model_dump(),
            {
                "task_name": "Fact Verification Task",
                "scores": {"accuracy": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0},
            },
        )


if __name__ == "__main__":
    unittest.main()
