import unittest

from target_benchmark.evaluators.TARGET import TARGET
from target_benchmark.retrievers.ottqa.OTTQARetriever import OTTQARetriever


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.ottqa_retriever = OTTQARetriever()
        self.target = TARGET()

    def test_run_needle_in_haystack(self):
        eval = TARGET(("Table Retrieval Task", ["fetaqa"]))
        res = eval.run(self.ottqa_retriever, top_k=20, split="test")
        self.assertIn("Table Retrieval Task", res)
        self.assertIn("fetaqa", res["Table Retrieval Task"])


if __name__ == "__main__":
    unittest.main()
