import unittest
from unittest.mock import patch, MagicMock
import os
from dataset_loaders.LoadersDataModels import QueryForTasksDataModel
from tasks import FactVerificationTask
from tasks.TasksDataModels import *
from dataset_loaders.HFDatasetLoader import HFDatasetLoader
from dataset_loaders.TargetDatasetConfig import HFDatasetConfigDataModel
from dataset_loaders.TargetDatasetConfig import (
    DEFAULT_WIKITQ_DATASET_CONFIG,
    DEFAULT_FETAQA_DATASET_CONFIG,
)
from retrievers.AbsCustomEmbeddingRetriever import (
    AbsCustomEmbeddingRetriever as CustomEmbRetr,
)
from retrievers.RetrieversDataModels import RetrievalResultDataModel

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
                retrieval_results=["Table1", "Table3"],
            ),
            RetrievalResultDataModel(
                dataset_name="dummy-dataset",
                query_id=2,
                retrieval_results=["Table5", "Table4"],
            ),
        ]
        self.mock_dataset_loader = MagicMock()
        self.mock_dataset_loader.get_table_id_to_table.return_value = {
            "Table1": [["fact"], ["Anna is 10 years old."]],
            "Table2": [["fact"], ["I'm not a freak athlete."]],
            "Table3": [["fact"], ["Jaylen Brown went to Cal."]],
            "Table4": [["fact"], ["Minecraft is such a fun game."]],
            "Table5": [["fact"], ["Today's temperature is 22.1 celsius."]],
        }
        self.mock_dataset_loader.get_queries_for_task.side_effect = (
            lambda splits, batch_size: iter(
                [
                    [
                        QueryForTasksDataModel(
                            query_id=1,
                            query="Jaylen Brown went to Stanford",
                            answer="False",
                            table_id="Table1",
                            database_id=0,
                        ),
                        QueryForTasksDataModel(
                            query_id=2,
                            query="Today's temperature is in the low 20s.",
                            answer="True",
                            table_id="Table5",
                            database_id=0,
                        ),
                    ]
                ]
            )
        )

    def test_fact_ver_task_run(self):

        results = self.fact_ver.task_run(
            retriever=self.mock_retriever,
            dataset_loaders={"dummy-dataset": self.mock_dataset_loader},
            logger=logger,
            batch_size=1,
            splits="test",
            top_k=2,
        )
        self.mock_retriever.retrieve_batch.assert_called_once_with(
            queries=[
                QueryForTasksDataModel(
                    query_id=1,
                    query="Jaylen Brown went to Stanford",
                    answer="False",
                    table_id="Table1",
                    database_id=0,
                ),
                QueryForTasksDataModel(
                    query_id=2,
                    query="Today's temperature is in the low 20s.",
                    answer="True",
                    table_id="Table5",
                    database_id=0,
                ),
            ],
            dataset_name="dummy-dataset",
            top_k=2,
        )
        self.assertDictEqual(
            results["dummy-dataset"].retrieval_performance.model_dump(),
            {"k": 2, "accuracy": 1.0, "precision": None, "recall": None},
        )
        self.assertDictEqual(
            results["dummy-dataset"].downstream_task_performance.model_dump(),
            {
                "task_name": "Fact Verification Task",
                "scores": {"accuracy": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0},
            },
        )


if __name__ == "__main__":
    unittest.main()
