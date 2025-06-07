import time
import unittest

from pandas import DataFrame

from target_benchmark.dataset_loaders import HFDatasetLoader, NeedleInHaystackDataLoader
from target_benchmark.dataset_loaders.AbsDatasetLoader import QueryType
from target_benchmark.dataset_loaders.TargetDatasetConfig import (
    DEFAULT_GITTABLES_DATASET_CONFIG,
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

    def test_nih_latency(self):
        nih_dataset = NeedleInHaystackDataLoader(**DEFAULT_GITTABLES_DATASET_CONFIG.model_dump())
        nih_dataset.load()
        start_time = time.time()
        times = [start_time]
        for batch in nih_dataset.get_corpus_iter():
            inner_start = time.time()
            times.append(inner_start)
            continue

        length = len(times) - 1
        avg_per_batch = sum([times[i + 1] - times[i] for i in range(length)]) / length

        end_time = time.time()
        print(f"elapsed time to iterate through 50k tables: {end_time - start_time}\navg time each batch: {avg_per_batch}")
        start_time = time.time()
        for batch in nih_dataset.get_corpus_iter():
            start_loop_time = time.time()
            break
        print(f"elapsed startup time: {start_loop_time - start_time}")


if __name__ == "__main__":
    unittest.main()
