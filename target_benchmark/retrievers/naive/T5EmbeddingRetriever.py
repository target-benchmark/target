from typing import Dict, List

from langchain_openai import OpenAIEmbeddings

from target_benchmark.retrievers import AbsStandardEmbeddingRetriever
from target_benchmark.retrievers.utils import json_table_str


class OpenAIEmbeddingRetriever(AbsStandardEmbeddingRetriever):
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        expected_corpus_format: str = "nested array",
    ):
        super().__init__(expected_corpus_format=expected_corpus_format)
        self.embedding_model = OpenAIEmbeddings(model=embedding_model)

    def embed_corpus(self, dataset_name: str, corpus_entry: Dict) -> List[float]:
        table_str = json_table_str(corpus_entry["table"])
        return self.embedding_model.embed_query(table_str)

    def retrieve(
        self,
        query: str,
    ):
        self.embedding_model.embed_query(query)
