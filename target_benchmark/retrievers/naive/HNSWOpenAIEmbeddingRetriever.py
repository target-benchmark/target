import os
import pickle
import numpy as np
import tqdm

from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, Iterable, Iterator, List, Tuple

from target_benchmark.dictionary_keys import TABLE_COL_NAME, TABLE_ID_COL_NAME, DATABASE_ID_COL_NAME

from target_benchmark.retrievers import AbsCustomEmbeddingRetriever, utils

file_dir = os.path.dirname(os.path.realpath(__file__))
default_out_dir = os.path.join(file_dir, "retrieval_files", "openai")


class HNSWOpenAIEmbeddingRetriever(AbsCustomEmbeddingRetriever):

    def __init__(
        self,
        out_dir: str = default_out_dir,
        embedding_model_id: str = "text-embedding-3-small",
        expected_corpus_format: str = "nested array",
        num_rows: int = 0,
    ):
        super().__init__(expected_corpus_format=expected_corpus_format)

        load_dotenv()

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)

        self.embedding_model_id = embedding_model_id

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
        response = self.client.embeddings.create(
            model=self.embedding_model_id,
            input=query,
        )
        return response.data[0].embedding

    def embed_corpus(self, dataset_name: str, corpus: Iterable[Dict]):
        """
        Function to embed the given corpus. This will be called in the evaluation pipeline before any retrieval.

        Parameters:
            dataset_name (str): the name of the corpus dataset.
            corpus (Iterable[Dict[str, List]]): an iterable of dicts, each being a batch of entries in the corpus dataset, containing database id, table id, the table contents (which the user can assume is in the format of self.expected_corpus_format), and context metadata (in these exact keys).
        Returns:
            nothing. the indexed embeddings are stored in a file.
        """

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
                table_str = utils.markdown_table_str(table, num_rows=self.num_rows)
                embedded_corpus[tup_id] = self.embed_query(table_str)

        corpus_index = utils.construct_embedding_index(list(embedded_corpus.values()))

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
