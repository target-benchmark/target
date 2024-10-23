from typing import Dict, List

from target_benchmark.retrievers import AbsStandardEmbeddingRetriever


class T5EmbeddingRetriever(AbsStandardEmbeddingRetriever):
    def __init__(
        self,
        expected_corpus_format: str = "nested array",
    ):
        super().__init__(expected_corpus_format=expected_corpus_format)

    def embed_corpus(self, dataset_name: str, corpus_entry: Dict) -> List[float]:
        pass

    def retrieve(
        self,
        query: str,
    ):
        pass
