import unittest
from unittest.mock import patch, MagicMock
import os
from dataset_loaders.LoadersDataModels import QueryForTasksDataModel
from dictionary_keys import METADATA_KEY_NAME
from evaluators import TARGET
from tasks.TableRetrievalTask import TableRetrievalTask
from tasks.TasksDataModels import *
from dataset_loaders.HFDatasetLoader import HFDatasetLoader
from dataset_loaders.TargetDatasetConfig import HFDatasetConfigDataModel
from dataset_loaders.TargetDatasetConfig import (
    DEFAULT_WIKITQ_DATASET_CONFIG,
    DEFAULT_FETAQA_DATASET_CONFIG,
)
from retrievers import OAIEmbedder
from retrievers.RetrieversDataModels import RetrievalResultDataModel
from qdrant_client import QdrantClient, models

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Get a logger
logger = logging.getLogger(__name__)


class TestTaskRunWithStdRetriever(unittest.TestCase):

    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.dataset_name = "dummy-dataset"
        self.client.create_collection(
            collection_name=self.dataset_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        )
        self.test_dataset = {
            "Table1": [["some random table"], ["some item"]],
            "Table2": [["some other random table"], ["another item"]],
            "Table3": [["third random table"], ["third item"]],
            "Table4": [["fourth random table"], ["fourth item"]],
            "Table5": [["fifth random table"], ["fifth item"]],
        }
        self.retr_task = TableRetrievalTask()
        self.retriever = OAIEmbedder()

        vectors = []
        metadata = []
        for table_id, table in self.test_dataset.items():
            table_embedding = self.retriever.embed_corpus(self.dataset_name, table)
            vectors.append(list(table_embedding))
            metadata.append({METADATA_KEY_NAME: table_id})
        self.client.upload_collection(
            collection_name=self.dataset_name,
            vectors=vectors,
            payload=metadata,
        )

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
            retriever=self.retriever,
            dataset_loaders={"dummy-dataset": self.mock_dataset_loader},
            logger=logger,
            batch_size=1,
            splits="test",
            top_k=2,
            client=self.client,
        )

    def test_basic_full_run(self):
        # end to end test that includes the client being created and retrieval from the client
        targ = TARGET()
        results = targ.run(self.retriever, splits="train")


if __name__ == "__main__":
    unittest.main()
