from typing import Dict
import unittest
from dataset_loaders import HFDatasetLoader, Text2SQLDatasetLoader
from dataset_loaders.TargetDatasetConfig import (
    DEFAULT_SPIDER_TEST_DATASET_CONFIG,
    DEFAULT_TABFACT_DATASET_CONFIG,
)
from dataset_loaders.AbsDatasetLoader import QueryType
from datasets import Dataset


class DataloadersTest(unittest.TestCase):

    def test_factver(self):
        tabfact_loader = HFDatasetLoader(**DEFAULT_TABFACT_DATASET_CONFIG.model_dump())
        self.assertEqual(tabfact_loader.query_type, QueryType.FACT_VERIFICATION)


if __name__ == "__main__":
    unittest.main()
