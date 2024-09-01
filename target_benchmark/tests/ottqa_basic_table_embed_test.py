from target_benchmark.retrievers.ottqa.OTTQARetriever import OTTQARetriever
from target_benchmark.dataset_loaders.HFDatasetLoader import HFDatasetLoader
import unittest


class TestOTTQARetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This setup happens once for the entire test class
        test_dataset_name = "fetaqa"
        test_hf_corpus_dataset_path = "target-benchmark/fetaqa-corpus"
        test_hf_queries_dataset_path = "target-benchmark/fetaqa-queries"

        # Initialize and load the dataset
        cls.fetaqa_loader = HFDatasetLoader(
            dataset_name=test_dataset_name,
            hf_corpus_dataset_path=test_hf_corpus_dataset_path,
            hf_queries_dataset_path=test_hf_queries_dataset_path,
        )
        cls.fetaqa_loader.load()  # Load the fetaqa data

        # Convert the corpus table
        dict_iter = cls.fetaqa_loader.convert_corpus_table_to()

        # Initialize the OTTQA retriever
        cls.ottqa_retriever = OTTQARetriever()

        # Embed the corpus
        cls.ottqa_retriever.embed_corpus(cls.fetaqa_loader.dataset_name, dict_iter)

    def test_corpus_keys(self):
        # Test to check if corpus keys are as expected
        expected_keys = ["database_id", "table_id", "table", "context"]
        self.assertEqual(
            set(self.fetaqa_loader.corpus.column_names), set(expected_keys)
        )

    def test_single_retrieve(self):
        # Test the retrieve function
        query = self.fetaqa_loader.queries[0]["query"]
        query_id = self.fetaqa_loader.queries[0]["query_id"]
        table_id = self.fetaqa_loader.queries[0]["table_id"]
        results = self.ottqa_retriever.retrieve(
            query, self.fetaqa_loader.dataset_name, 10
        )
        print(
            f"\nretrieving on query: {query}\n\ncorrect table: {table_id}\n\n actual results: {results}"
        )
        # Perform assertions on the results
        self.assertIsInstance(results, list)
        # More assertions depending on the expected structure of the results

    def test_query_table_id(self):
        # Test to check the query's table ID
        expected_table_id = (
            "totto_source/dev_json/example-2205.json"  # the expected table ID here
        )
        query_table_id = self.fetaqa_loader.queries[0]["table_id"]
        self.assertEqual(query_table_id, expected_table_id)


if __name__ == "__main__":
    unittest.main()
