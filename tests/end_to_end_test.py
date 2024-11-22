import unittest

from target_benchmark.evaluators.TARGET import TARGET
from target_benchmark.retrievers.ottqa.OTTQARetriever import OTTQARetriever


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.ottqa_retriever = OTTQARetriever()
        self.target = TARGET()

    # def test_run(self):
    #     res = self.target.run(self.ottqa_retriever, top_k=20, split="train")
    #     print(res)
    def test_run_needle_in_haystack(self):
        eval = TARGET(("Table Retrieval Task", ["fetaqa", "gittables"]))
        res = eval.run(self.ottqa_retriever, top_k=20, split="test")
        print(res)


if __name__ == "__main__":
    unittest.main()
