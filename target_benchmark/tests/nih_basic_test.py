from target_benchmark.dataset_loaders.TargetDatasetConfig import (
    DEFAULT_FETAQA_DATASET_CONFIG,
    DEFAULT_GITTABLES_DATASET_CONFIG,
)
from target_benchmark.evaluators import TARGET
from target_benchmark.retrievers import AbsCustomEmbeddingRetriever
from target_benchmark.tasks import TableRetrievalTask
import unittest
class DummyRetriever(AbsCustomEmbeddingRetriever):
    def retrieve(
        self,
        query: str,
        dataset_name: str,
        top_k: int,
        **kwargs,
    ):
        return [("", "")] * top_k

    def embed_corpus(self, dataset_name, corpus) -> None:
        self.num_tables = 0
        for entry in corpus:
            self.num_tables += 1
    def reset(self) -> int:
        corps = self.num_tables
        self.num_tables = 0
        return corps

class TestNIHTask(unittest.TestCase):

    def setUp(self):
        self.retriever = DummyRetriever()

    def _create_target_evaluation(self, num_tables):
        gittables_config = DEFAULT_GITTABLES_DATASET_CONFIG.model_copy()
        gittables_config.num_tables = num_tables
        return TARGET(
            TableRetrievalTask(
                {
                    "fetaqa": DEFAULT_FETAQA_DATASET_CONFIG,
                    "gittables": gittables_config,
                }
            )
        )

    def test_add_no_nih_tables(self):
        eval_0 = self._create_target_evaluation(0)

        _ = eval_0.run(retriever=self.retriever)
        num_corps_0 = self.retriever.reset()
        self.assertEqual(2003, num_corps_0)

    def test_added_nih_tables(self):
        eval_0 = self._create_target_evaluation(0)
        _ = eval_0.run(retriever=self.retriever)
        num_corps_0 = self.retriever.reset()

        eval_500 = self._create_target_evaluation(500)
        _ = eval_500.run(retriever=self.retriever)
        num_corps_500 = self.retriever.reset()
        
        eval_1k = self._create_target_evaluation(1000)
        _ = eval_1k.run(retriever=self.retriever)
        num_corps_1k = self.retriever.reset()
        
        self.assertEqual(500, num_corps_1k - num_corps_500)
        self.assertEqual(500, num_corps_500 - num_corps_0)

    def test_added_all_gittables(self):
        eval_0 = self._create_target_evaluation(0)
        _ = eval_0.run(retriever=self.retriever)
        num_corps_0 = self.retriever.reset()

        eval_full = self._create_target_evaluation(None)
        _ = eval_full.run(retriever=self.retriever)
        num_corps_full = self.retriever.reset()
        self.assertEqual(int(5e4), num_corps_full - num_corps_0)

if __name__ == "__main__":
    unittest.main()
