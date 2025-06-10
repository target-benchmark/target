import unittest
import os

from target_benchmark.evaluators.TARGET import TARGET
from target_benchmark.retrievers.murre.MurreRetriever import MurreRetriever
from target_benchmark.tasks import TableRetrievalTask


# Similar to llama_index_basic_test.py
class TestMurreBasics(unittest.TestCase):
    def setUp(self):
        self.fetaqa_dummy_config = {
            "dataset_name": "fetaqa",
            "hf_corpus_dataset_path": "jixy2012/mock-hf-corpus-dataset",
            "hf_queries_dataset_path": "jixy2012/mock-hf-queries-dataset",
            "query_type": "Table Question Answering",
        }
        self.trt = TableRetrievalTask({"fetaqa": self.fetaqa_dummy_config}, True)
        self.evaluator = TARGET(downstream_tasks=self.trt)

    def test_run_murre_on_dummy(self):
        murre = MurreRetriever(
            model="text-embedding-3-small",
            embedding_type="openai",
            use_rewriting=False,
            beam_size=2,
            max_hops=1
        )

        res = self.evaluator.run(murre)
        self.assertIn("Table Retrieval Task", res)
        print(res)

    def test_murre_instance_creation(self):
        
        murre_default = MurreRetriever()
        self.assertEqual(murre_default.model, "text-embedding-3-small")
        self.assertEqual(murre_default.embedding_type, "openai")
        self.assertEqual(murre_default.beam_size, 3)
        self.assertEqual(murre_default.max_hops, 2)
        self.assertFalse(murre_default.use_rewriting)
        
        murre_custom = MurreRetriever(
            model="text-embedding-ada-002",
            embedding_type="openai",
            use_rewriting=True,
            beam_size=5,
            max_hops=3,
            rewriter_model="gpt-4o"
        )
        self.assertEqual(murre_custom.model, "text-embedding-ada-002")
        self.assertTrue(murre_custom.use_rewriting)
        self.assertEqual(murre_custom.beam_size, 5)
        self.assertEqual(murre_custom.max_hops, 3)
        self.assertEqual(murre_custom.rewriter_model, "gpt-4o")

    def test_table_conversion(self):
        murre = MurreRetriever()
        
        table_data = [
            ["Name", "Age", "City"],
            ["John", "25", "New York"],
            ["Jane", "30", "Los Angeles"]
        ]
        result = murre._convert_table_to_string(table_data)
        expected = "Columns: Name | Age | City \\n Data: John | 25 | New York \\n Data: Jane | 30 | Los Angeles"
        self.assertEqual(result, expected)
        
        empty_table = []
        result_empty = murre._convert_table_to_string(empty_table)
        self.assertEqual(result_empty, "[]")
        
        result_none = murre._convert_table_to_string(None)
        self.assertEqual(result_none, "None")

    @unittest.skipIf(not os.getenv("OPENAI_API_KEY"), "OpenAI API key not available")
    def test_embedding_functionality(self):
        murre = MurreRetriever(
            model="text-embedding-3-small",
            embedding_type="openai"
        )
        
        murre._ensure_embedding_client()
        self.assertIsNotNone(murre.embedding_client)
        
        test_texts = ["Sample table with data"]
        embeddings = murre._get_embeddings_batch(test_texts, is_query=True)
        
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(embeddings[0]), 1536)  

    def test_beam_state_creation(self):
        from target_benchmark.retrievers.murre.MurreRetriever import BeamState
        
        beam = BeamState(
            retrieved_tables=[1, 2, 3],
            current_query="test query",
            score=0.85,
            hop=1
        )
        
        self.assertEqual(beam.retrieved_tables, [1, 2, 3])
        self.assertEqual(beam.current_query, "test query")
        self.assertEqual(beam.score, 0.85)
        self.assertEqual(beam.hop, 1)
        
        beam_none = BeamState(
            retrieved_tables=None,
            current_query="test",
            score=0.0,
            hop=0
        )
        self.assertEqual(beam_none.retrieved_tables, [])


if __name__ == "__main__":
    unittest.main()
