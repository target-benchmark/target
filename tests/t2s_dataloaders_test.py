import unittest
from dataset_loaders import Text2SQLDatasetLoader
from dataset_loaders.TargetDatasetConfig import (
    DEFAULT_SPIDER_DATASET_CONFIG,
    DEFAULT_TABFACT_DATASET_CONFIG,
)
from dataset_loaders.AbsDatasetLoader import QueryType


class T2SDataloadersTest(unittest.TestCase):
    def test_text2sql(self):
        spider_loader = Text2SQLDatasetLoader(
            dataset_name="spider",
        )
        spider_loader._download_spider()


if __name__ == "__main__":
    unittest.main()
