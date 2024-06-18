from retrievers.ottqa.OTTQARetriever import OTTQARetriever
from dataset_loaders.HFDatasetLoader import HFDatasetLoader
from evaluators.TARGET import TARGET
from tasks.QuestionAnsweringTask import QuestionAnsweringTask
import unittest
import os


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.ottqa_retriever = OTTQARetriever()

        self.target = TARGET()

    def test_run(self):
        res = self.target.run(self.ottqa_retriever, top_k=20, split="train")
        print(res)


if __name__ == "__main__":
    unittest.main()
