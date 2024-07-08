from retrievers import AbsStandardEmbeddingRetriever
from typing import Dict, List
from langchain_openai import OpenAIEmbeddings
from dataset_loaders.utils import markdown_table_with_headers


class OpenAIEmbedder(AbsStandardEmbeddingRetriever):

    def __init__(self, expected_corpus_format: str = "nested array"):
        super().__init__(expected_corpus_format=expected_corpus_format)
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    def embed_query(
        self,
        query: str,
        dataset_name: str,
        **kwargs,
    ) -> List[float]:
        return self.embedding_model.embed_query(query)

    def embed_corpus(self, dataset_name: str, corpus_entry: Dict) -> List[float]:
        table_str = markdown_table_with_headers(corpus_entry["table"])
        return self.embedding_model.embed_query(table_str)