import unittest
from unittest.mock import patch, MagicMock
from dataset_loaders.LoadersDataModels import QueryForTasksDataModel
from evaluators.TARGET import TARGET
from dataset_loaders.TargetDatasetConfig import DEFAULT_FETAQA_DATASET_CONFIG
from retrievers.AbsCustomEmbeddingRetriever import (
    AbsCustomEmbeddingRetriever as CustomEmbRetr,
)
from retrievers.RetrieversDataModels import RetrievalResultDataModel
from tasks import TableRetrievalTask


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.fetaqa_dummy_config = {
            "dataset_name": "fetaqa",
            "hf_corpus_dataset_path": "jixy2012/mock-hf-corpus-dataset",
            "hf_queries_dataset_path": "jixy2012/mock-hf-queries-dataset",
            "query_type": "Table Question Answering",
        }
        self.trt = TableRetrievalTask({"fetaqa": self.fetaqa_dummy_config}, True)
        self.evaluator = TARGET(downstream_task_objects=self.trt)

    def test_default_evaluator_creation(self):
        default_evaluator = TARGET()
        self.assertEqual(self.evaluator.get_loaded_tasks(), ["Table Retrieval Task"])

    def test_with_task_evaluator_creation(self):
        self.assertEqual(self.evaluator.get_loaded_tasks(), ["Table Retrieval Task"])
        self.assertEqual(list(self.evaluator.dataset_info.keys()), ["fetaqa"])

    def test_dataset_loaders_creation(self):
        feta_loader = self.evaluator.create_dataloaders(
            self.evaluator.dataset_info, "train"
        )["fetaqa"]
        self.assertEqual(feta_loader.dataset_name, "fetaqa")
        self.assertEqual(feta_loader.table_col_name, "table")
        self.assertEqual(feta_loader.table_id_col_name, "table_id")
        self.assertEqual(feta_loader.database_id_col_name, "database_id")
        self.assertEqual(feta_loader.query_col_name, "query")
        self.assertEqual(feta_loader.query_id_col_name, "query_id")
        self.assertEqual(feta_loader.answer_col_name, "answer")
        self.assertEqual(feta_loader.split, "train")

    def test_dataset_loaders_loading(self):
        feta_loader = self.evaluator.create_dataloaders(
            self.evaluator.dataset_info, "train"
        )["fetaqa"]
        self.assertEqual(feta_loader.corpus, None)
        self.assertEqual(feta_loader.queries, None)
        feta_loader.load()
        self.assertNotEqual(feta_loader.corpus, None)
        self.assertNotEqual(feta_loader.queries, None)
        self.assertDictEqual(
            feta_loader.corpus[0],
            {
                "database_id": 1,
                "table_id": "event_schedule.csv",
                "table": [
                    ["Event", "Date", "Expected Attendees"],
                    ["Tech Conference", "2023-10-15", "150"],
                    ["Music Festival", "2023-11-05", "2000"],
                ],
            },
        )
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
        with patch("evaluators.TARGET.TARGET.create_dataloaders") as mock_func:
            mock_retriever = MagicMock()
            mock_retriever.__class__ = CustomEmbRetr
            mock_retriever.retrieve_batch.return_value = [
                RetrievalResultDataModel(
                    dataset_name="fetaqa",
                    query_id=1,
                    retrieval_results=["Table1", "Table2"],
                ),
                RetrievalResultDataModel(
                    dataset_name="fetaqa",
                    query_id=2,
                    retrieval_results=["Table3", "Table4"],
                ),
            ]
            mock_dataset_loader = MagicMock()
            mock_dataset_loader.get_queries_for_task.side_effect = (
                lambda batch_size: iter(
                    [
                        [
                            QueryForTasksDataModel(
                                query_id=1,
                                query="Test query",
                                answer="Test answer",
                                table_id="Table1",
                                database_id=0,
                            ),
                            QueryForTasksDataModel(
                                query_id=2,
                                query="Test query 2",
                                answer="Test answer 2",
                                table_id="Table5",
                                database_id=0,
                            ),
                        ]
                    ]
                )
            )
            mock_func.return_value = {"fetaqa": mock_dataset_loader}

            results = self.evaluator.run(mock_retriever, top_k=2)
            self.assertIsInstance(results, dict)
            self.assertEqual(list(results.keys()), ["Table Retrieval Task"])
            self.assertDictEqual(
                results["Table Retrieval Task"][
                    "fetaqa"
                ].retrieval_performance.model_dump(),
                {"k": 2, "accuracy": 0.5, "precision": None, "recall": None},
            )
            self.assertDictEqual(
                results["Table Retrieval Task"][
                    "fetaqa"
                ].downstream_task_performance.model_dump(),
                {"task_name": None, "scores": None},
            )


if __name__ == "__main__":
    unittest.main()
