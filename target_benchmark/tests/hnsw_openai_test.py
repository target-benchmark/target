import unittest
import os
from pathlib import Path
from target_benchmark.retrievers import HNSWOpenAIEmbeddingRetriever
from target_benchmark.dictionary_keys import (
    DATABASE_ID_COL_NAME,
    TABLE_COL_NAME,
    TABLE_ID_COL_NAME,
    CONTEXT_COL_NAME,
)
# Import the HNSWOpenAIEmbeddingRetriever from the module where it is defined
# from your_module import HNSWOpenAIEmbeddingRetriever, DATABASE_ID_COL_NAME, TABLE_ID_COL_NAME, TABLE_COL_NAME

class TestHNSWOpenAIEmbeddingRetriever(unittest.TestCase):
    def setUp(self):
        # Set up the retriever with a test output directory
        self.retriever = HNSWOpenAIEmbeddingRetriever(out_dir="test_output")
        self.dataset_name = "test_dataset"

        # Minimal test corpus with two tables
        self.corpus = [
            {
                DATABASE_ID_COL_NAME: ["db1", "db2"],
                TABLE_ID_COL_NAME: ["table1", "table2"],
                TABLE_COL_NAME: [
                    [["row1_col1", "row1_col2"], ["row2_col1", "row2_col2"]],
                    [["row1_col1", "row1_col2"], ["row2_col1", "row2_col2"]]
                ],
                CONTEXT_COL_NAME: [{}, {}],
            }
        ]

        # Expected identifier based on corpus setup
        self.corpus_identifier = f"{self.dataset_name}_numrows_all"

    def tearDown(self):
        # Clean up test output files
        idx_path, db_table_ids_path = self.retriever._construct_persistence_paths(self.corpus_identifier)
        if idx_path.exists():
            os.remove(idx_path)
        if db_table_ids_path.exists():
            os.remove(db_table_ids_path)
        if Path("test_output").exists():
            Path("test_output").rmdir()

    def test_embedding_and_retrieving(self):

        # Run the embedding process
        self.retriever.embed_corpus(self.dataset_name, self.corpus)

        # Verify that the files were created
        idx_path, db_table_ids_path = self.retriever._construct_persistence_paths(self.corpus_identifier)
        self.assertTrue(idx_path.exists())
        self.assertTrue(db_table_ids_path.exists())

        # Test retrieval
        retrieved_ids = self.retriever.retrieve(query="test query", dataset_name=self.dataset_name, top_k=1)
        self.assertEqual(self.retriever.corpus_identifier, self.corpus_identifier)
        self.assertIsNotNone(self.retriever.corpus_index)
        self.assertIsNotNone(self.retriever.db_table_ids)
        self.assertEqual(len(retrieved_ids), 1)  # Verify we get 1 result as specified by top_k

    def test_corpus_identifier_update(self):
        # Check that changing num_rows updates the corpus_identifier
        self.retriever.num_rows = 5
        new_identifier = self.retriever._get_corpus_identifier(self.dataset_name)
        self.assertNotEqual(new_identifier, self.corpus_identifier)
        self.assertIn("numrows_5", new_identifier)

# Run the test
if __name__ == "__main__":
    unittest.main()
