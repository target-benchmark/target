import shutil
import unittest
from pathlib import Path

from target_benchmark import Target
from target_benchmark.retrievers.AbsCustomEmbeddingRetriever import (
    AbsCustomEmbeddingRetriever as CustomEmbRetr,
)
from target_benchmark.tasks import TableRetrievalTask


class DummyRetriever(CustomEmbRetr):
    def __init__(self, expected_corpus_format="nested array"):
        super().__init__(expected_corpus_format=expected_corpus_format)
        self.retrieve_called_times = 0

    def reset_called_times(self):
        self.retrieve_called_times = 0

    def retrieve(self, query, dataset_name, top_k, **kwargs):
        self.retrieve_called_times += 1
        return [("0", "0")] * top_k

    def embed_corpus(self, dataset_name, corpus):
        pass


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.fetaqa_dummy_config = {
            "dataset_name": "fetaqa",
            "hf_corpus_dataset_path": "jixy2012/mock-hf-corpus-dataset",
            "hf_queries_dataset_path": "jixy2012/mock-hf-queries-dataset",
            "query_type": "Table Question Answering",
        }
        self.trt = TableRetrievalTask({"fetaqa": self.fetaqa_dummy_config}, True)
        self.evaluator = Target(downstream_tasks=self.trt)
        self.retriever = DummyRetriever()
        self.retrieval_results_dir = Path(__file__).parent / "test_retrieval_files"
        self.fetaqa_out_path = self.retrieval_results_dir / "fetaqa"
        self.k_5_out_path = self.fetaqa_out_path / "5.jsonl"

    def test_basic_task_run(self):
        _ = self.evaluator.run(retriever=self.retriever, retrieval_results_dir=str(self.retrieval_results_dir))
        self.assertTrue(self.fetaqa_out_path.exists())
        self.assertTrue(self.k_5_out_path.exists())

    def test_resume(self):
        _ = self.evaluator.run(retriever=self.retriever, retrieval_results_dir=str(self.retrieval_results_dir))
        self.assertTrue(self.k_5_out_path.exists())
        with open(self.k_5_out_path, "r") as file:
            lines = file.readlines()

        self.assertEqual(len(lines), 3)
        lines = lines[:-2]

        with open(self.k_5_out_path, "w") as file:
            file.writelines(lines)
        self.assertEqual(self.retriever.retrieve_called_times, 3)
        self.retriever.reset_called_times()
        # rerun to make sure resume happens correctly
        _ = self.evaluator.run(retriever=self.retriever, retrieval_results_dir=str(self.retrieval_results_dir))
        # should have retrieved 2 more times
        self.assertEqual(self.retriever.retrieve_called_times, 2)
        with open(self.k_5_out_path, "r") as file:
            lines = file.readlines()
        self.assertEqual(len(lines), 3)

    def tearDown(self):
        if self.retrieval_results_dir.exists() and self.retrieval_results_dir.is_dir():
            shutil.rmtree(self.retrieval_results_dir)


if __name__ == "__main__":
    unittest.main()
