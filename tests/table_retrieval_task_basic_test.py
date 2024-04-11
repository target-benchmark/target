import unittest
from unittest.mock import patch, MagicMock
import os
from dataset_loaders.LoadersDataModels import QueryForTasksDataModel
from tasks.TableRetrievalTask import TableRetrievalTask
from dataset_loaders.HFDatasetLoader import HFDatasetLoader
from dataset_loaders.TargetDatasetConfig import HFDatasetConfigDataModel
from dataset_loaders.TargetDatasetConfig import DEFAULT_FETAQA_DATASET_CONFIG
from retrievers.AbsTargetCustomEmbeddingRetriver import (
    AbsTargetCustomEmbeddingRetriver as CustomEmbRetr,
)

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Get a logger
logger = logging.getLogger(__name__)


class TestOTTQARetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retr_task = TableRetrievalTask()
        cls.dataset_config = cls.retr_task.get_dataset_config()
        fetaqa_config = cls.dataset_config["test_dataset"]
        cls.fetaqa_loader = HFDatasetLoader(**fetaqa_config.model_dump())
        cls.fetaqa_loader.load()

    def test_basic_task_run(self):
        mock_retriever = MagicMock()
        mock_retriever.__class__ = CustomEmbRetr
        mock_retriever.retrieve_batch.return_value = {1: ["Table1, Table2"]}
        mock_dataset_loader = MagicMock()
        mock_dataset_loader.get_queries_for_task.side_effect = (
            lambda splits, batch_size: iter(
                [
                    [
                        QueryForTasksDataModel(
                            query_id=1,
                            query="Test query",
                            answer="Test answer",
                            table_id="Table1",
                            database_id=0,
                        )
                    ]
                ]
            )
        )

        self.retr_task.task_run(
            retriever=mock_retriever,
            dataset_loaders={"test_dataset": mock_dataset_loader},
            logger=logger,
            batch_size=1,
            splits="test",
            top_k=2,
        )
        mock_retriever.retrieve_batch.assert_called_once_with(
            queries={1: "Test query"},  # The exact structure depends on your setup
            dataset_name="test_dataset",  # Make sure this matches your test setup
            top_k=2,  # Or whatever value you're testing with
        )


if __name__ == "__main__":
    unittest.main()
