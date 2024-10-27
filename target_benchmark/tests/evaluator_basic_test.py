import unittest
from unittest.mock import patch, MagicMock
from target_benchmark.dataset_loaders.LoadersDataModels import HFDatasetConfigDataModel
from target_benchmark.evaluators.TARGET import TARGET
from target_benchmark.retrievers.AbsCustomEmbeddingRetriever import (
    AbsCustomEmbeddingRetriever as CustomEmbRetr,
)

from target_benchmark.retrievers.RetrieversDataModels import RetrievalResultDataModel
from target_benchmark.tasks import TableRetrievalTask


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.fetaqa_dummy_config = {
            "dataset_name": "fetaqa",
            "hf_corpus_dataset_path": "jixy2012/mock-hf-corpus-dataset",
            "hf_queries_dataset_path": "jixy2012/mock-hf-queries-dataset",
            "query_type": "Table Question Answering",
        }
        self.trt = TableRetrievalTask({"fetaqa": self.fetaqa_dummy_config}, True)
        self.evaluator = TARGET(downstream_tasks=self.trt)

    def test_input_dataset_config(self):
        target = TARGET(
            downstream_tasks=[
                ("Text to SQL Task", "spider-test"),
                ("Table Question Answering Task", ["fetaqa", "ottqa"]),
                self.trt,
            ]
        )
        self.assertSetEqual(
            set(target.get_loaded_tasks()),
            set(
                [
                    "Table Question Answering Task",
                    "Text to SQL Task",
                    "Table Retrieval Task",
                ]
            ),
        )
        dataset_info = target.dataset_info
        self.assertSetEqual(
            set(dataset_info.keys()), set(["spider-test", "fetaqa", "ottqa"])
        )
        self.assertEqual(dataset_info["spider-test"].split, "test")
        self.assertIsInstance(dataset_info["ottqa"], HFDatasetConfigDataModel)

    def test_duplicate_input_dataset_config(self):
        target = TARGET(downstream_tasks=["Table Retrieval Task", self.trt])
        # check if warning was tracked in log

    def test_default_evaluator_creation(self):
        default_evaluator = TARGET()
        self.assertEqual(self.evaluator.get_loaded_tasks(), ["Table Retrieval Task"])

    def test_with_task_evaluator_creation(self):
        self.assertEqual(self.evaluator.get_loaded_tasks(), ["Table Retrieval Task"])
        self.assertEqual(list(self.evaluator.dataset_info.keys()), ["fetaqa"])

    def test_dataset_loaders_creation(self):
        feta_loader = self.evaluator.create_dataloaders(
            self.evaluator.dataset_info, "test"
        )["fetaqa"]
        self.assertEqual(feta_loader.dataset_name, "fetaqa")
        self.assertEqual(feta_loader.split, "test")

    def test_dataset_loaders_loading(self):
        feta_loader = self.evaluator.create_dataloaders(
            self.evaluator.dataset_info, "test"
        )["fetaqa"]
        self.assertEqual(feta_loader.corpus, None)
        self.assertEqual(feta_loader.queries, None)
        feta_loader.load()
        self.assertNotEqual(feta_loader.corpus, None)
        self.assertNotEqual(feta_loader.queries, None)
        expected = {
                "database_id": 1,
                "table_id": "event_schedule.csv",
                "table": [
                    ["Event", "Date", "Expected Attendees"],
                    ["Tech Conference", "2023-10-15", "150"],
                    ["Music Festival", "2023-11-05", "2000"],
                ],
            }
        self.assertTrue(expected.items() <= feta_loader.corpus[0].items())
        self.assertDictEqual(
            feta_loader.queries[0],
            {
                "query_id": 1,
                "database_id": 2,
                "table_id": "car_maintenance.csv",
                "query": 'What is the cost for "Oil Change" service?',
                "answer": "60",
            },
        )

    def test_basic_run_task(self):
        with patch("target_benchmark.evaluators.TARGET.TARGET.create_dataloaders") as mock_func:
            mock_retriever = MagicMock()
            mock_retriever.__class__ = CustomEmbRetr
            mock_retriever.retrieve_batch.return_value = [
                RetrievalResultDataModel(
                    dataset_name="fetaqa",
                    query_id=1,
                    retrieval_results=[("0", "Table1"), ("0", "Table2")],
                ),
                RetrievalResultDataModel(
                    dataset_name="fetaqa",
                    query_id=2,
                    retrieval_results=[("0", "Table3"), ("0", "Table4")],
                ),
            ]
            mock_dataset_loader = MagicMock()
            mock_dataset_loader.get_queries_size.return_value = 2
            mock_dataset_loader.get_queries_for_task.side_effect = (
                lambda batch_size: iter(
                    [
                        {
                            "query_id": [1, 2],
                            "query": ["Test query", "Test query 2"],
                            "answer": ["Test answer", "Test answer 2"],
                            "table_id": ["Table1", "Table5"],
                            "database_id": [0, 0],
                        }
                    ],
                )
            )
            mock_func.return_value = {"fetaqa": mock_dataset_loader}
            results = self.evaluator.run(mock_retriever, top_k=2)
            self.assertIsInstance(results, dict)
            self.assertEqual(list(results.keys()), ["Table Retrieval Task"])
            expected_vals = {"k": 2, "accuracy": 0.5, "precision": None, "recall": None}
            actual_vals = results["Table Retrieval Task"][
                "fetaqa"
            ].retrieval_performance.model_dump()

            for key, value in expected_vals.items():
                self.assertIn(key, actual_vals)
                self.assertEqual(value, actual_vals[key])
    def test_needle_in_haystack(self):
        eval = TARGET(("Table Retrieval Task", ["fetaqa", "gittables"]))
        tasks = eval.tasks
        self.assertIn("Table Retrieval Task", tasks)
        task = tasks["Table Retrieval Task"]
        dls = task.get_dataset_config()
        self.assertIn("gittables", dls)
        self.assertIn("fetaqa", dls)

if __name__ == "__main__":
    unittest.main()
