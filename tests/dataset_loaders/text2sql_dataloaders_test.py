import unittest
from typing import Dict

from datasets import Dataset

from target_benchmark.dataset_loaders import Text2SQLDatasetLoader
from target_benchmark.dataset_loaders.AbsDatasetLoader import QueryType
from target_benchmark.dataset_loaders.TargetDatasetConfig import (
    DEFAULT_SPIDER_DATASET_CONFIG,
    get_default_beaver_dataset,
)


class T2SDataloadersTest(unittest.TestCase):
    def test_default_v2(self):
        beaver_loader = get_default_beaver_dataset()
        self.assertEqual(beaver_loader.query_type, QueryType.TEXT_2_SQL)
        self.assertEqual(beaver_loader.dataset_name, "beaver")

        beaver_loader.load()
        corpus = beaver_loader.get_corpus()
        corpus_headers = beaver_loader.get_corpus_header()
        queries = beaver_loader.get_queries()
        queries_headers = beaver_loader.get_queries_header()

        self.assertIsInstance(corpus, Dataset)
        self.assertSetEqual(set(corpus_headers), set(["table", "table_id", "database_id", "context"]))

        self.assertIsInstance(queries, Dataset)
        self.assertTrue(set(["query", "answer", "table_id", "database_id", "query_id", "difficulty"]) <= set(queries_headers))

        database_dir = beaver_loader.get_path_to_database()
        self.assertTrue("test_database" in str(database_dir))

    def test_default_spider(self):
        spider_loader = Text2SQLDatasetLoader(**DEFAULT_SPIDER_DATASET_CONFIG.model_dump())
        self.assertEqual(spider_loader.query_type, QueryType.TEXT_2_SQL)
        self.assertEqual(spider_loader.dataset_name, "spider")

        spider_loader.load()
        corpus = spider_loader.get_corpus()
        corpus_headers = spider_loader.get_corpus_header()
        queries = spider_loader.get_queries()
        queries_headers = spider_loader.get_queries_header()
        self.assertIsInstance(corpus, Dataset)
        self.assertSetEqual(set(corpus_headers), set(["table", "table_id", "database_id", "context"]))
        self.assertIsInstance(corpus["context"][0], Dict)
        self.assertIn("foreign_keys", corpus["context"][0])
        self.assertIn("primary_key", corpus["context"][0])
        print(len(corpus["context"]))

        self.assertIsInstance(queries, Dataset)
        self.assertSetEqual(
            set(queries_headers),
            set(["query", "answer", "table_id", "database_id", "query_id", "difficulty"]),
        )

    def test_convert_to_format(self):
        spider_loader = Text2SQLDatasetLoader(**DEFAULT_SPIDER_DATASET_CONFIG.model_dump())
        spider_loader._load_corpus()
        num_tables = 0
        for batch in spider_loader.get_corpus_iter():
            num_tables += 1
        self.assertEqual(num_tables, 180)
        print(spider_loader.path_to_database_dir)


if __name__ == "__main__":
    unittest.main()
