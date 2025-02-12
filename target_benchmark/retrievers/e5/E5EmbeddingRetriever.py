from typing import Dict, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from target_benchmark.dictionary_keys import TABLE_COL_NAME
from target_benchmark.retrievers import AbsStandardEmbeddingRetriever
from target_benchmark.retrievers.utils import markdown_table_str

# high performing lightweight embedding model from NovaSearch
# https://huggingface.co/NovaSearch/stella_en_400M_v5


class E5EmbeddingRetriever(AbsStandardEmbeddingRetriever):
    def __init__(
        self,
        num_rows: Union[int, None] = None,
    ):
        super().__init__("nested array")
        self.num_rows = num_rows
        self.model = SentenceTransformer("intfloat/multilingual-e5-large-instruct", trust_remote_code=True).cuda()

    def embed_query(self, query: str, dataset_name: str) -> np.ndarray:
        return self.model.encode(query, convert_to_tensor=True, normalize_embeddings=True)

    def embed_corpus(self, dataset_name: str, corpus_entry: Dict) -> np.ndarray:
        table = corpus_entry[TABLE_COL_NAME]
        num_rows_to_include = len(table) - 1
        if self.num_rows is not None:
            num_rows_to_include = self.num_rows
        table_str = markdown_table_str(table, num_rows=num_rows_to_include)

        if self.num_rows and num_rows_to_include != self.num_rows:
            print(f"truncated input due to context length constraints, included {num_rows_to_include} rows")
        return self.model.encode(table_str, convert_to_tensor=True, normalize_embeddings=True)
