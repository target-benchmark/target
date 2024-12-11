import unittest

from target_benchmark import Target
from target_benchmark.retrievers.ottqa.OTTQARetriever import OTTQARetriever


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.ottqa_retriever = OTTQARetriever()
        self.target = Target()

    def test_run_needle_in_haystack(self):
        eval = Target(("Table Retrieval Task", ["fetaqa"]))
        res = eval.run(self.ottqa_retriever, top_k=20, split="test")
        self.assertIn("Table Retrieval Task", res)
        self.assertIn("fetaqa", res["Table Retrieval Task"])


if __name__ == "__main__":
    unittest.main()
