import unittest
from target_benchmark.dataset_loaders import HFDatasetLoader
from target_benchmark.dataset_loaders.TargetDatasetConfig import (
    DEFAULT_TABFACT_DATASET_CONFIG,
)
from target_benchmark.dataset_loaders.AbsDatasetLoader import QueryType


class DataloadersTest(unittest.TestCase):

    def test_factver(self):
        tabfact_loader = HFDatasetLoader(**DEFAULT_TABFACT_DATASET_CONFIG.model_dump())
        self.assertEqual(tabfact_loader.query_type, QueryType.FACT_VERIFICATION)


if __name__ == "__main__":
    unittest.main()
