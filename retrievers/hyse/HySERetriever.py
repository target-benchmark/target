import json
import os

import hnswlib
import instructor
import numpy as np
import pandas as pd
import pickle
import tqdm

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import Dict, Iterable, Iterator, List, Tuple, Optional

from dictionary_keys import TABLE_COL_NAME, TABLE_ID_COL_NAME, DATABASE_ID_COL_NAME
from retrievers import AbsCustomEmbeddingRetriever, utils

file_dir = os.path.dirname(os.path.realpath(__file__))
default_out_dir = os.path.join(file_dir, "retrieval_files", "hyse")



class ResponseFormat(BaseModel):
    headers: List[List[str]]


class HySERetriever(AbsCustomEmbeddingRetriever):

    def __init__(
        self,
        out_dir: str = default_out_dir,
        expected_corpus_format: str = "nested array",
    ):
        super().__init__(expected_corpus_format)

        load_dotenv()

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)

    def retrieve(
        self,
        query: str,
        dataset_name: str,
        top_k: int,
        **kwargs,
    ) -> List[Tuple[int, str]]:
        """
        Directly retrieves the predicted relevant tables for the query.

        Parameters:
            query (str): the actual query string.

            dataset_name (str): identifier for the dataset that these queries come from.
            since retrieval evaluation can be done for multiple datasets,
            use this as a way of choosing which dataset's corpus to retrieve from.

            top_k (int): the top k tables to retrieve for each query

            any additional kwargs you'd like to include.

        Returns:
            List[str]: the list of table ids of the retrieved tables.
        """
        with open(
            os.path.join(self.out_dir, f"corpus_index_{dataset_name}.pkl"), "rb"
        ) as f:
            corpus_index = pickle.load(f)

        with open(
            os.path.join(self.out_dir, f"db_table_ids_{dataset_name}.pkl"), "rb"
        ) as f:
            # stored separately as hnsw only takes int indices
            db_table_ids = pickle.load(f)

        s = 2
        # generate s hypothetical schemas for given query
        hypothetical_schemas_query = self.generate_hypothetical_schemas(query, s)

        # embed each hypothetical schema
        hypothetical_schema_embeddings = []
        for hypothetical_schema in hypothetical_schemas_query:
            hypothetical_schema_embeddings += [
                self.embed_query(table=hypothetical_schema)
            ]

        # Query dataset, k - number of the closest elements (returns 2 numpy arrays)
        retrieved_ids, distances = corpus_index.knn_query(
            np.array(hypothetical_schema_embeddings),
            # retrieves s*10 tables, 10 tables per hypothetical schema
            k=int(top_k / s),
        )

        # Get original table_ids (table names) from the retrieved integer identifiers for each in s hypothetical schemas
        retrieved_full_ids = []
        for i in range(s):
            retrieved_full_ids += [db_table_ids[id] for id in retrieved_ids[i]]

        return retrieved_full_ids


    def embed_corpus(self, dataset_name: str, corpus: Iterable[Dict]):
        """
        Function to embed the given corpus. This will be called in the evaluation pipeline before any retrieval.

        Parameters:
            dataset_name (str): the name of the corpus dataset.
            corpus (Iterable[Dict[str, List]]): an iterable of dicts, each being a batch of entries in the corpus dataset, containing database id, table id, the table contents (which the user can assume is in the format of self.expected_corpus_format), and context metadata (in these exact keys).
        Returns:
            nothing. the indexed embeddings are stored in a file.
        """
        if os.path.exists(os.path.join(self.out_dir, f"corpus_index_{dataset_name}.pkl")):
            return

        embedded_corpus = {}
        for corpus_dict in tqdm.tqdm(corpus):
            for db_id, table_id, table in zip(corpus_dict[DATABASE_ID_COL_NAME], corpus_dict[TABLE_ID_COL_NAME], corpus_dict[TABLE_COL_NAME]):
                tup_id = (db_id, table_id)
                embedded_corpus[tup_id] = self.embed_query(table=table, id=tup_id)

        corpus_index = utils.construct_embedding_index(list(embedded_corpus.values()))

        # Store table embedding index and table ids in distinct files
        with open(
            os.path.join(self.out_dir, f"corpus_index_{dataset_name}.pkl"), "wb"
        ) as f:
            pickle.dump(corpus_index, f)

        with open(
            os.path.join(self.out_dir, f"db_table_ids_{dataset_name}.pkl"), "wb"
        ) as f:
            pickle.dump(list(embedded_corpus.keys()), f)


    def embed_query(self, table: List[List], id: Tuple[int, str] = None) -> List[List]:
        """Embed table using default openai embedding model, only using table header for now."""
        try:
            table_str = utils.json_table_str(table)
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                # current: embed schema only
                # todo: add table values
                input=table_str,
            )
            return response.data[0].embedding
        except Exception as e:
            print("error on: ", id, e)
            return []


    def generate_hypothetical_schemas(self, query: str, s: int) -> List[List]:
        """Generate a hypothetical schema relevant to answer the query."""

        client = instructor.from_openai(self.client)
        response_model = ResponseFormat

        # TODO: ensure correctness of output type and dimension
        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {
                    "role": "system",
                    # todo: add table values
                    # todo: expand to multi-table
                    "content": f"""You are an assistant skilled in database schemas.""",
                },
                {
                    "role": "user",
                    "content": f"""
                        Generate exactly {s} table headers, which are different in semantics and size,
                        of tables which could answer the following query: {query}.

                        Return a nested list of size {s} in which each list contains table attributes (strings),
                        without any surrounding numbers or text.
                    """
                },
            ],
            response_model=response_model,
            temperature=0,
        )

        hypothetical_schemas = response.headers

        return hypothetical_schemas
