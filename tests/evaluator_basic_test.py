import unittest
from unittest.mock import patch, MagicMock
from dataset_loaders.LoadersDataModels import QueryForTasksDataModel
from evaluators.TargetEvaluator import TargetEvaluator
from dataset_loaders.TargetDatasetConfig import DEFAULT_FETAQA_DATASET_CONFIG
from retrievers.AbsCustomEmbeddingRetriever import (
    AbsCustomEmbeddingRetriever as CustomEmbRetr,
)
from retrievers.RetrieversDataModels import RetrievalResultDataModel


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = TargetEvaluator()


    def test_default_task_creation(self):
        self.assertEqual(self.evaluator.get_loaded_tasks(), ["Table Retrieval Task"])
        self.assertEqual(list(self.evaluator.dataset_info.keys()), ["fetaqa"])
        self.assertDictEqual(self.evaluator.dataset_info["fetaqa"].model_dump(), DEFAULT_FETAQA_DATASET_CONFIG.model_dump())
        self.assertEqual(list(self.evaluator.dataloaders.keys()), ["fetaqa"])
    
    def test_dataset_loaders_creation(self):
        feta_loader = self.evaluator.dataloaders["fetaqa"]
        self.assertEqual(feta_loader.dataset_name, "fetaqa")
        self.assertEqual(feta_loader.table_col_name, "table")
        self.assertEqual(feta_loader.table_id_col_name, "table_id")
        self.assertEqual(feta_loader.database_id_col_name, "database_id")
        self.assertEqual(feta_loader.query_col_name, "query")
        self.assertEqual(feta_loader.query_id_col_name, "query_id")
        self.assertEqual(feta_loader.answer_col_name, "answer")
        self.assertEqual(feta_loader.splits, ["test"])

    def test_dataset_loaders_loading(self):
        feta_loader = self.evaluator.dataloaders["fetaqa"]
        self.assertEqual(feta_loader.corpus, None)
        self.assertEqual(feta_loader.queries, None)
        feta_loader.load()
        self.assertNotEqual(feta_loader.corpus, None)
        self.assertNotEqual(feta_loader.queries, None)
        self.assertDictEqual(feta_loader.queries["test"][0], {'query_id': 2206, 'database_id': 0, 'table_id': 'totto_source/dev_json/example-2205.json', 'query': 'What TV shows was Shagun Sharma seen in 2019?', 'answer': 'In 2019, Shagun Sharma played in the roles as Pernia in Laal Ishq, Vikram Betaal Ki Rahasya Gatha as Rukmani/Kashi and Shaadi Ke Siyape as Dua.'})

    def test_basic_run_task(self):
        self.mock_retriever = MagicMock()
        self.mock_retriever.__class__ = CustomEmbRetr
        self.mock_retriever.retrieve_batch.return_value = [
            RetrievalResultDataModel(
                dataset_name="fetaqa",
                query_id=1,
                retrieval_results=["Table1", "Table2"]
            ),
            RetrievalResultDataModel(
                dataset_name="fetaqa",
                query_id=2,
                retrieval_results=["Table3", "Table4"]
            ),
        ]
        self.mock_dataset_loader = MagicMock()
        self.mock_dataset_loader.get_queries_for_task.side_effect = (
            lambda splits, batch_size: iter(
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
        self.evaluator.dataloaders["fetaqa"] = self.mock_dataset_loader
        results = self.evaluator.run(self.mock_retriever, top_k=2)
        self.assertIsInstance(results, dict)
        self.assertEqual(list(results.keys()), ["Table Retrieval Task"])
        self.assertDictEqual(results["Table Retrieval Task"]["fetaqa"].retrieval_performance.model_dump(), {"k": 2, "accuracy": 0.5, "precision": None, "recall": None})
        self.assertDictEqual(results["Table Retrieval Task"]["fetaqa"].downstream_task_performance.model_dump(), {"task_name": None})

    
if __name__ == "__main__":
    unittest.main()