from typing import Dict, Union

import numpy as np
from langchain_openai import OpenAIEmbeddings

from target_benchmark.retrievers import AbsStandardEmbeddingRetriever
from target_benchmark.retrievers.utils import markdown_table_str


class OpenAIEmbedder(AbsStandardEmbeddingRetriever):
    def __init__(
        self,
        expected_corpus_format: str = "nested array",
        num_rows: Union[int, None] = None,
    ):
        super().__init__(expected_corpus_format=expected_corpus_format)
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.num_rows = num_rows

    def embed_query(
        self,
        query: str,
        dataset_name: str,
        **kwargs,
    ) -> np.ndarray:
        emb = self.embedding_model.embed_query(query)
        return np.array(emb)

    def embed_corpus(self, dataset_name: str, corpus_entry: Dict) -> np.ndarray:
        table_str = markdown_table_str(corpus_entry["table"], self.num_rows)
        emb = self.embedding_model.embed_query(table_str)
        return np.array(emb)
