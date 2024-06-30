import unittest
from dataset_loaders import Text2SQLDatasetLoader
from dataset_loaders.AbsDatasetLoader import QueryType
from dataset_loaders.TargetDatasetConfig import DEFAULT_SPIDER_TEST_DATASET_CONFIG


class T2SDataloadersTest(unittest.TestCase):
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
