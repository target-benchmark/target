import os
from typing import Dict, Iterable

from dotenv import load_dotenv

from target_benchmark.retrievers import AbsCustomEmbeddingRetriever

file_dir = os.path.dirname(os.path.realpath(__file__))
default_out_dir = os.path.join(file_dir, "retrieval_files", "analysis")


class NoContextRetriever(AbsCustomEmbeddingRetriever):
    """
    The difference with the DefaultOpenAIEmbeddingRetriever is that it uses the
    HNSW index for consistent evaluation with the HySE retriever.
    """

    def __init__(
        self,
        out_dir: str = default_out_dir,
        expected_corpus_format: str = "nested array",
    ):
        super().__init__(expected_corpus_format=expected_corpus_format)

        load_dotenv()

        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)

    def retrieve(
        self,
        query: str,
        dataset_name: str,
        top_k: int,
        **kwargs,
    ):
        return []

    def embed_query(self, query: str):
        return

    def embed_corpus(self, dataset_name: str, corpus: Iterable[Dict]):
        """
        Function to embed the given corpus. This will be called in the evaluation pipeline before any retrieval.

        Parameters:
            dataset_name (str): the name of the corpus dataset.
            corpus (Iterable[Dict[str, List]]): an iterable of dicts, each being a batch of entries in the corpus dataset, containing database id, table id, the table contents (which the user can assume is in the format of self.expected_corpus_format), and context metadata (in these exact keys).
        Returns:
            nothing. the indexed embeddings are stored in a file.
        """
        return
