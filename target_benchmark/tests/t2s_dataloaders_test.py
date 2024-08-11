from typing import Dict
import unittest
from target_benchmark.dataset_loaders import Text2SQLDatasetLoader
from target_benchmark.dataset_loaders.AbsDatasetLoader import QueryType
from target_benchmark.dataset_loaders.TargetDatasetConfig import DEFAULT_SPIDER_TEST_DATASET_CONFIG
from datasets import Dataset


class T2SDataloadersTest(unittest.TestCase):
    def test_default_spider(self):
        spider_loader = Text2SQLDatasetLoader(
            **DEFAULT_SPIDER_TEST_DATASET_CONFIG.model_dump()
        )
        self.assertEqual(spider_loader.query_type, QueryType.TEXT_2_SQL)
        self.assertEqual(spider_loader.dataset_name, "spider")

        spider_loader.load()
        corpus = spider_loader.get_corpus()
        corpus_headers = spider_loader.get_corpus_header()
        queries = spider_loader.get_queries()
        queries_headers = spider_loader.get_queries_header()
        self.assertIsInstance(corpus, Dict)
        self.assertSetEqual(
            set(corpus_headers), set(["table", "table_id", "database_id", "context"])
        )
        self.assertIsInstance(corpus["context"][0], Dict)
        self.assertIn("foreign_keys", corpus["context"][0])
        self.assertIn("primary_key", corpus["context"][0])
        print(len(corpus["context"]))

        self.assertIsInstance(queries, Dataset)
        self.assertSetEqual(
            set(queries_headers),
            set(
                ["query", "answer", "table_id", "database_id", "query_id", "difficulty"]
            ),
        )

    def test_text2sql(self):
        spider_loader = Text2SQLDatasetLoader(
            **DEFAULT_SPIDER_TEST_DATASET_CONFIG.model_dump()
        )
        spider_loader._load_corpus()
        # spider_loader._download_bird()
        self.assertIsNotNone(spider_loader.corpus)
        self.assertIsInstance(spider_loader.corpus, dict)
        self.assertIn("table", spider_loader.corpus)
        self.assertIn("context", spider_loader.corpus)
        self.assertIn("table_id", spider_loader.corpus)
        self.assertIn("database_id", spider_loader.corpus)

    def test_convert_to_format(self):
        spider_loader = Text2SQLDatasetLoader(
            **DEFAULT_SPIDER_TEST_DATASET_CONFIG.model_dump()
        )
        spider_loader._load_corpus()
        num_tables = 0
        for batch in spider_loader.convert_corpus_table_to():
            num_tables += 1
        self.assertEqual(num_tables, 180)
        print(spider_loader.path_to_database_dir)


if __name__ == "__main__":
    unittest.main()
