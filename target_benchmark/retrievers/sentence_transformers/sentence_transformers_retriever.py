from typing import Dict, Union
from sentence_transformers import SentenceTransformer
import numpy as np

from target_benchmark.dictionary_keys import TABLE_COL_NAME
from target_benchmark.retrievers import AbsStandardEmbeddingRetriever
from target_benchmark.retrievers.utils import markdown_table_str


class SentenceTransformersRetriever(AbsStandardEmbeddingRetriever):
    def __init__(
            self,
            model_name_or_path: str,
            num_rows: Union[int, None] = None,
            **kwargs
    ):
        """
        Args:
            model_name_or_path (str, optional): If it is a filepath on disc, it loads the model from that path. If it is not a path,
                it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model
                from the Hugging Face Hub with that name.
            **kwargs: Any additional kwargs we want to pass to the SentenceTransformer class initialization.
        """
        super().__init__("nested array")
        self.num_rows = num_rows
        self.model = SentenceTransformer(model_name_or_path, **kwargs)

    def embed_query(self, query: str, dataset_name: str) -> np.ndarray:
        return self.model.encode(query, convert_to_tensor=True, normalize_embeddings=True).cpu().numpy()

    def embed_corpus(self, dataset_name: str, corpus_entry: Dict) -> np.ndarray:
        table = corpus_entry[TABLE_COL_NAME]
        num_rows_to_include = self.num_rows or len(table) - 1
        table_str = markdown_table_str(table, num_rows=num_rows_to_include)

        if num_rows_to_include != self.num_rows:
            print(f"truncated input due to context length constraints, included {num_rows_to_include} rows")
        return self.model.encode(table_str, convert_to_tensor=True, normalize_embeddings=True).cpu().numpy()

# high performing lightweight embedding model from NovaSearch
# https://huggingface.co/NovaSearch/stella_en_400M_v5
class StellaEmbeddingRetriever(SentenceTransformersRetriever):

    def __init__(self, num_rows: Union[int, None] = None, **kwargs):
        super().__init__(
            model_name_or_path="NovaSearch/stella_en_400M_v5",
            num_rows=num_rows,
            **kwargs
        )

    # Override the default `embed_query` from subclassed `SentenceTransformersRetriever`
    #   to add the unique `prompt_name` arg
    def embed_query(self, query: str, dataset_name: str) -> np.ndarray:
        return self.model.encode(query, prompt_name="s2p_query")