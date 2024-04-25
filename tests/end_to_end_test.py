from retrievers.ottqa.OTTQARetriever import OTTQARetriever
from dataset_loaders.HFDatasetLoader import HFDatasetLoader
from evaluators.TARGET import TARGET
from tasks.QuestionAnsweringTask import QuestionAnsweringTask
import unittest
import os

class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.ottqa_retriever = OTTQARetriever(
            script_dir=os.path.dirname(os.path.abspath(__file__))
        )

        self.target = TARGET("Question Answering Task")

    def test_run(self):
        res = self.target.run(self.ottqa_retriever)
        print(res)

if __name__ == "__main__":
    print(os.path.dirname(os.path.abspath(__file__)))
    unittest.main()