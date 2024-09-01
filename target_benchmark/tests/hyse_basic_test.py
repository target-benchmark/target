import unittest

from target_benchmark.evaluators.TARGET import TARGET

from target_benchmark.retrievers.hyse.HySERetriever import HySERetriever
from target_benchmark.tasks import TableRetrievalTask


class TestHyseBasics(unittest.TestCase):
    def setUp(self):
        self.fetaqa_dummy_config = {
            "dataset_name": "fetaqa",
            "hf_corpus_dataset_path": "jixy2012/mock-hf-corpus-dataset",
            "hf_queries_dataset_path": "jixy2012/mock-hf-queries-dataset",
            "query_type": "Table Question Answering",
        }
        self.trt = TableRetrievalTask({"fetaqa": self.fetaqa_dummy_config}, True)
        self.evaluator = TARGET(downstream_tasks=self.trt)

    def test_run_hyse_on_dummy(self):
        hyse = HySERetriever()

        res = self.evaluator.run(hyse)
        print(res)


if __name__ == "__main__":
    unittest.main()
