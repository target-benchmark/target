from target_benchmark.retrievers.ottqa.OTTQARetriever import OTTQARetriever
from target_benchmark.evaluators.TARGET import TARGET
import unittest


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.ottqa_retriever = OTTQARetriever()
        self.target = TARGET()

    def test_run(self):
        res = self.target.run(self.ottqa_retriever, top_k=20, split="train")
        print(res)


if __name__ == "__main__":
    unittest.main()
