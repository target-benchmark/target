import unittest

from evaluators.TARGET import TARGET
from retrievers.AbsCustomEmbeddingRetriever import (
    AbsCustomEmbeddingRetriever as CustomEmbRetr,
)
from retrievers.llama_index.LlamaIndexRetriever import LlamaIndexRetriever
from retrievers.RetrieversDataModels import RetrievalResultDataModel
from tasks import TableRetrievalTask
import os


class TestLlamaIndexBasics(unittest.TestCase):
    def setUp(self):
        self.fetaqa_dummy_config = {
            "dataset_name": "fetaqa",
            "hf_corpus_dataset_path": "jixy2012/mock-hf-corpus-dataset",
            "hf_queries_dataset_path": "jixy2012/mock-hf-queries-dataset",
            "query_type": "Table Question Answering",
        }
        self.trt = TableRetrievalTask({"fetaqa": self.fetaqa_dummy_config}, True)
        self.evaluator = TARGET(downstream_tasks=self.trt)

    def test_run_llama_on_dummy(self):
        llama = LlamaIndexRetriever()

        res = self.evaluator.run(llama)
        print(res)


if __name__ == "__main__":
    unittest.main()
