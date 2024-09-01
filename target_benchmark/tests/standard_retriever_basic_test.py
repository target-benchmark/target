import unittest
from unittest.mock import MagicMock
from target_benchmark.dictionary_keys import METADATA_TABLE_ID_KEY_NAME, METADATA_DB_ID_KEY_NAME
from evaluators import TARGET
from target_benchmark.tasks.TableRetrievalTask import TableRetrievalTask
from target_benchmark.tasks.TasksDataModels import *
from target_benchmark.retrievers import OpenAIEmbedder
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
        self.dataset_name = "fetaqa"
        self.client.create_collection(
            collection_name=self.dataset_name,
            vectors_config=models.VectorParams(
                size=1536, distance=models.Distance.COSINE
            ),
        )
        self.test_dataset = {
            "Table1": [["some random table"], ["some item"]],
            "Table2": [["some other random table"], ["another item"]],
            "Table3": [["third random table"], ["third item"]],
            "Table4": [["fourth random table"], ["fourth item"]],
            "Table5": [["fifth random table"], ["fifth item"]],
        }
        self.retr_task = TableRetrievalTask()
        self.retriever = OpenAIEmbedder()

        vectors = []
        metadata = []
        for table_id, table in self.test_dataset.items():
            table_embedding = self.retriever.embed_corpus(
                self.dataset_name,
                {"database_id": 1, "table_id": table_id, "table": table, "context": {}},
            )
            vectors.append(list(table_embedding))
            metadata.append(
                {METADATA_TABLE_ID_KEY_NAME: table_id, METADATA_DB_ID_KEY_NAME: 1}
            )
        self.client.upload_collection(
            collection_name=self.dataset_name,
            vectors=vectors,
            payload=metadata,
        )

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
            retriever=self.retriever,
            dataset_loaders={"fetaqa": self.mock_dataset_loader},
            logger=logger,
            batch_size=1,
            top_k=2,
            client=self.client,
        )

    def test_basic_full_run(self):
        # end to end test that includes the client being created and retrieval from the client
        fetaqa_dummy_config = {
            "dataset_name": "fetaqa",
            "hf_corpus_dataset_path": "jixy2012/mock-hf-corpus-dataset",
            "hf_queries_dataset_path": "jixy2012/mock-hf-queries-dataset",
            "query_type": "Table Question Answering",
        }
        trt = TableRetrievalTask({"fetaqa": fetaqa_dummy_config}, True)
        targ = TARGET(downstream_tasks=trt)
        results = targ.run(self.retriever)
        print(results)


if __name__ == "__main__":
    unittest.main()
