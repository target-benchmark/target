import os
import pickle
from typing import Dict, Iterable, Union

import numpy as np
import tiktoken
import tqdm
from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

from target_benchmark.dictionary_keys import (
    DATABASE_ID_COL_NAME,
    TABLE_COL_NAME,
    TABLE_ID_COL_NAME,
)
from target_benchmark.retrievers import AbsCustomEmbeddingRetriever
from target_benchmark.retrievers.utils import (
    construct_embedding_index,
    markdown_table_str,
)

file_dir = os.path.dirname(os.path.realpath(__file__))
default_out_dir = os.path.join(file_dir, "retrieval_files", "openai")


class HNSWOpenAIEmbeddingRetriever(AbsCustomEmbeddingRetriever):
    """
    The difference with the DefaultOpenAIEmbeddingRetriever is that it uses the
    HNSW index for consistent evaluation with the HySE retriever.
    """

    def __init__(
        self,
        out_dir: str = default_out_dir,
        embedding_model_id: str = "text-embedding-3-small",
        expected_corpus_format: str = "nested array",
        num_rows: Union[int, None] = None,
    ):
        super().__init__(expected_corpus_format=expected_corpus_format)

        load_dotenv()

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.out_dir = out_dir
        self.corpus_identifier = ""
        self.embedding_model_id = embedding_model_id
        # TODO: need to get this dynamically according to model id
        self.embedding_model_encoding = tiktoken.get_encoding("cl100k_base")
        self.num_rows = num_rows

    def retrieve(
        self,
        query: str,
        dataset_name: str,
        top_k: int,
        **kwargs,
    ):
        # TODO: add split to file!
        with open(
            os.path.join(self.out_dir, f"corpus_index_{self.corpus_identifier}.pkl"),
            "rb",
        ) as f:
            corpus_index = pickle.load(f)

        with open(
            os.path.join(self.out_dir, f"db_table_ids_{self.corpus_identifier}.pkl"),
            "rb",
        ) as f:
            # stored separately as hnsw only takes int indices
            db_table_ids = pickle.load(f)

        query_embedding = self.embed_query(query)

        # Query dataset
        retrieved_ids, distances = corpus_index.knn_query(
            np.array(query_embedding),
            k=top_k,
        )

        # Get original table_ids (table names) from the retrieved integer identifiers for each query
        retrieved_full_ids = [db_table_ids[id] for id in retrieved_ids[0]]

        return retrieved_full_ids

    def embed_query(self, query: str):
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model_id,
                input=query,
            )
            return response.data[0].embedding
        except BadRequestError as e:
            print(type(query), len(query))
            raise e

    def embed_corpus(self, dataset_name: str, corpus: Iterable[Dict]):
        """
        Function to embed the given corpus. This will be called in the evaluation pipeline before any retrieval.

        Parameters:
            dataset_name (str): the name of the corpus dataset.
            corpus (Iterable[Dict[str, List]]): an iterable of dicts, each being a batch of entries in the corpus dataset, containing database id, table id, the table contents (which the user can assume is in the format of self.expected_corpus_format), and context metadata (in these exact keys).
        Returns:
            nothing. the indexed embeddings are stored in a file.
        """
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)

        self.corpus_identifier = f"{dataset_name}_numrows_all"
        if self.num_rows is not None:
            self.corpus_identifier = f"{dataset_name}_numrows_{self.num_rows}"

        if os.path.exists(
            os.path.join(self.out_dir, f"corpus_index_{self.corpus_identifier}.pkl")
        ):
            return

        embedded_corpus = {}
        for corpus_dict in tqdm.tqdm(corpus):
            for db_id, table_id, table in zip(
                corpus_dict[DATABASE_ID_COL_NAME],
                corpus_dict[TABLE_ID_COL_NAME],
                corpus_dict[TABLE_COL_NAME],
            ):
                tup_id = (db_id, table_id)
                num_rows_to_include = self.num_rows
                while num_rows_to_include >= 0:
                    table_str = markdown_table_str(table, num_rows=num_rows_to_include)
                    num_tokens = len(self.embedding_model_encoding.encode(table_str))
                    if (
                        num_tokens < 8192
                    ):  # this is not great, need to remove hardcode in future
                        break
                    num_rows_to_include -= 10

                if num_rows_to_include != self.num_rows:
                    print(
                        f"truncated input due to context length constraints, included {num_rows_to_include} rows"
                    )
                embedded_corpus[tup_id] = self.embed_query(table_str)

        corpus_index = construct_embedding_index(list(embedded_corpus.values()))

        # Store table embedding index and table ids in distinct files
        with open(
            os.path.join(self.out_dir, f"corpus_index_{self.corpus_identifier}.pkl"),
            "wb",
        ) as f:
            pickle.dump(corpus_index, f)

        with open(
            os.path.join(self.out_dir, f"db_table_ids_{self.corpus_identifier}.pkl"),
            "wb",
        ) as f:
            pickle.dump(list(embedded_corpus.keys()), f)
