import unittest

from pandas import DataFrame

from target_benchmark.dataset_loaders import HFDatasetLoader
from target_benchmark.dataset_loaders.AbsDatasetLoader import QueryType
from target_benchmark.dataset_loaders.TargetDatasetConfig import (
    DEFAULT_TABFACT_DATASET_CONFIG,
)


class DataloadersTest(unittest.TestCase):
    def test_factver(self):
        tabfact_loader = HFDatasetLoader(**DEFAULT_TABFACT_DATASET_CONFIG.model_dump())
        self.assertEqual(tabfact_loader.query_type, QueryType.FACT_VERIFICATION)

    def test_itertable(self):
        test_dataset = HFDatasetLoader(
            dataset_name="test",
            hf_corpus_dataset_path="jixy2012/mock-hf-corpus-dataset",
            hf_queries_dataset_path="jixy2012/mock-hf-queries-dataset",
            query_type="Fact Verification",
        )
        test_dataset.load()

        for batch in test_dataset.get_corpus_iter():
            for row in batch["table"][0]:
                self.assertIsInstance(row, list)
        for batch in test_dataset.get_corpus_iter(output_format="dataframe"):
            self.assertIsInstance(batch["table"][0], DataFrame)
        for batch in test_dataset.get_corpus_iter(output_format="json"):
            for row in batch["table"][0]:
                self.assertIsInstance(row, dict)


if __name__ == "__main__":
    unittest.main()
