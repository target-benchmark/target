import unittest
from target_benchmark.dataset_loaders import HFDatasetLoader
from target_benchmark.dataset_loaders.TargetDatasetConfig import (
    DEFAULT_TABFACT_DATASET_CONFIG,
)
from target_benchmark.dataset_loaders.AbsDatasetLoader import QueryType


class DataloadersTest(unittest.TestCase):
    def setUp(self):
        self.tabfact_loader = HFDatasetLoader(**DEFAULT_TABFACT_DATASET_CONFIG.model_dump())
        self.tabfact_loader.load()

    def test_query_dataset_start_index(self):
        start_idx = 100
        num_queries = 0
        for _ in self.tabfact_loader.get_queries_for_task(start_idx=start_idx, batch_size=1):
            num_queries += 1
        self.assertEqual(num_queries, self.tabfact_loader.get_queries_size() - start_idx)

    def test_factver(self):
        self.assertEqual(self.tabfact_loader.query_type, QueryType.FACT_VERIFICATION)


if __name__ == "__main__":
    unittest.main()
