import logging
import unittest

from target_benchmark.dataset_loaders.TargetDatasetConfig import (
    DEFAULT_GITTABLES_DATASET_CONFIG,
    DEFAULT_TABFACT_DATASET_CONFIG,
)
from target_benchmark.tasks import TableRetrievalTask

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("bert_score").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)

# Get a logger
logger = logging.getLogger(__name__)


class TestBasicTaskInit(unittest.TestCase):
    def test_default_datasets(self):
        def_datasets = TableRetrievalTask._get_default_dataset_config()
        self.assertIn("tabfact", def_datasets)
        self.assertIn("fetaqa", def_datasets)
        self.assertIn("ottqa", def_datasets)
        self.assertNotIn("gittables", def_datasets)

    def test_available_datasets(self):
        def_datasets = TableRetrievalTask.get_available_datasets()
        self.assertIn("tabfact", def_datasets)
        self.assertIn("fetaqa", def_datasets)
        self.assertIn("ottqa", def_datasets)
        self.assertIn("gittables", def_datasets)

    def test_task_creation_with_dict(self):
        task = TableRetrievalTask(
            {
                "tabfact": {
                    "dataset_name": "tabfact",
                    "split": "test",
                    "query_type": "Fact Verification",
                    "hf_corpus_dataset_path": "target-benchmark/tabfact-corpus",
                    "hf_queries_dataset_path": "target-benchmark/tabfact-queries",
                },
                "gittables": {
                    "dataset_name": "gittables",
                    "split": "train",
                    "query_type": "Needle in Haystack",
                    "hf_corpus_dataset_path": "target-benchmark/gittables-corpus",
                },
            }
        )

        configs = task.get_dataset_config()
        self.assertIn("gittables", configs)
        self.assertIn("tabfact", configs)
        self.assertEqual(configs["gittables"].query_type, "Needle in Haystack")
        self.assertEqual(configs["tabfact"].query_type, "Fact Verification")

    def test_task_creation_with_config(self):
        task = TableRetrievalTask(
            {
                "tabfact": DEFAULT_TABFACT_DATASET_CONFIG,
                "gittables": DEFAULT_GITTABLES_DATASET_CONFIG,
            }
        )

        configs = task.get_dataset_config()
        self.assertIn("gittables", configs)
        self.assertIn("tabfact", configs)
        self.assertEqual(configs["gittables"].query_type, "Needle in Haystack")
        self.assertEqual(configs["tabfact"].query_type, "Fact Verification")

    def test_task_creation_with_error(self):
        with self.assertRaises(AssertionError):
            TableRetrievalTask(
                {
                    "gittables": DEFAULT_GITTABLES_DATASET_CONFIG,
                }
            )


if __name__ == "__main__":
    unittest.main()
