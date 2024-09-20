from typing import Dict, Iterable, List, Tuple

from target_benchmark.retrievers import AbsCustomEmbeddingRetriever


class NoContextRetriever(AbsCustomEmbeddingRetriever):
    def __init__(
        self,
        expected_corpus_format: str = "nested array",
    ):
        super().__init__(expected_corpus_format=expected_corpus_format)

    def retrieve(
        self,
        query: str,
        dataset_name: str,
        top_k: int,
        **kwargs,
    ) -> List[Tuple]:
        return [("", "") for _ in range(top_k)]

    def embed_corpus(self, dataset_name: str, corpus: Iterable[Dict]) -> None:
        return
