from typing import Dict, Union

import numpy as np
import pandas as pd
from transformers import TapasModel, TapasTokenizer

from target_benchmark.dictionary_keys import TABLE_COL_NAME
from target_benchmark.retrievers import AbsStandardEmbeddingRetriever

# high performing lightweight embedding model from NovaSearch
# https://huggingface.co/NovaSearch/stella_en_400M_v5


class TapasTableEncoderRetriever(AbsStandardEmbeddingRetriever):
    def __init__(
        self,
        num_rows: Union[int, None] = None,
    ):
        super().__init__("pandas")
        self.num_rows = num_rows
        self.model = TapasModel.from_pretrained("google/tapas-base")
        self.tokenizer = TapasTokenizer.from_pretrained("google/tapas-base", drop_rows_to_fit=True)
        self.model.eval()

    def _create_embedding(self, table: pd.DataFrame, query: str) -> np.ndarray:
        input = self.tokenizer(
            table=table,
            queries=[query],
            return_tensors="pt",
        )
        return self.model(**input).last_hidden_state[0].mean(axis=0).detach().numpy()

    def embed_query(self, query: str, dataset_name: str):
        return self._create_embedding(table=pd.DataFrame(), query=query)

    def embed_corpus(self, dataset_name: str, corpus_entry: Dict):
        table: pd.DataFrame = corpus_entry[TABLE_COL_NAME]
        if self.num_rows is not None:
            table = table.head(self.num_rows)
        return self._create_embedding(table=table, query="")
