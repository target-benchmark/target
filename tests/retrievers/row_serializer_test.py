import shutil
import unittest
from pathlib import Path

from target_benchmark.dataset_loaders import HFDatasetLoader
from target_benchmark.dataset_loaders.TargetDatasetConfig import (
    DEFAULT_DUMMY_DATASET_CONFIG,
)
from target_benchmark.retrievers import RowSerializationRetriever


class TestRowSerializationEmbeddingRetriever(unittest.TestCase):
    def setUp(self):
        self.retriever = RowSerializationRetriever(str(Path(__file__).parent / "data"))
        self.dataloader = HFDatasetLoader(**DEFAULT_DUMMY_DATASET_CONFIG.model_dump())
        self.dataloader.load()

    def test_basic_embed(self):
        self.retriever.embed_corpus("fetaqa", self.dataloader.convert_corpus_table_to())

        client = self.retriever.client
        self.assertEqual(client.count("fetaqa", exact=True).count, 11)

    def test_retrieve(self):
        self.retriever.embed_corpus("fetaqa", self.dataloader.convert_corpus_table_to())

        query = "who is the manager?"
        retrieval = self.retriever.retrieve(query, "fetaqa", 5)
        counts = {}
        for retrieved in retrieval:
            if retrieved not in counts:
                counts[retrieved] = 0
            counts[retrieved] += 1
        self.assertEqual(len(counts), 5)
        for key in counts:
            self.assertEqual(counts[key], 1)

    def tearDown(self):
        shutil.rmtree(self.retriever.embeddings_dir)


if __name__ == "__main__":
    unittest.main()
