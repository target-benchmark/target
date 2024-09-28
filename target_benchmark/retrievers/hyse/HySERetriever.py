import os
import pickle
from typing import Dict, Iterable, List, Tuple

import instructor
import numpy as np
import pandas as pd
import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from transformers import TapasModel, TapasTokenizer

from target_benchmark.dictionary_keys import (
    DATABASE_ID_COL_NAME,
    TABLE_COL_NAME,
    TABLE_ID_COL_NAME,
)
from target_benchmark.retrievers import AbsCustomEmbeddingRetriever, utils

file_dir = os.path.dirname(os.path.realpath(__file__))
default_out_dir = os.path.join(file_dir, "retrieval_files", "hyse")


class ResponseFormat(BaseModel):
    schemas: List[List[List]]


class HySERetriever(AbsCustomEmbeddingRetriever):
    def __init__(
        self,
        out_dir: str = default_out_dir,
        expected_corpus_format: str = "nested array",
        model_name: str = "openai",
        num_rows: int = 2,
        num_schemas: int = 2,
        with_query: bool = True,
        aggregated: bool = False,
    ):
        super().__init__(expected_corpus_format)

        load_dotenv()

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        if model_name == "tapas":
            self.model = TapasModel.from_pretrained(
                "google/tapas-base"  # check performance difference with fine-tuned model take qa as base
            )
            self.tokenizer = TapasTokenizer.from_pretrained(
                "google/tapas-base", drop_rows_to_fit=True
            )
            self.model.eval()

        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)

        self.model_name = model_name
        self.num_rows = num_rows
        self.num_schemas = num_schemas
        self.with_query = with_query
        self.aggregated = aggregated

    def retrieve(
        self,
        query: str,
        dataset_name: str,
        top_k: int,
        **kwargs,
    ) -> List[Tuple]:
        """
        Directly retrieves the predicted relevant tables for the query.

        Parameters:
            query (str): the actual query string.

            dataset_name (str): identifier for the dataset that these queries come from.
            since retrieval evaluation can be done for multiple datasets,
            use this as a way of choosing which dataset's corpus to retrieve from.

            top_k (int): the top k tables to retrieve for each query

            num_schemas: number of hypothetical schemas to generate for a given input query

            num_rows: number of (hypothetical) rows generated per table in the schema.
            Be aware that this parameter should be aligned with the #rows used for embedding tables in the corpus.

            any additional kwargs you'd like to include.

        Returns:
            List[str]: the list of table ids of the retrieved tables.
        """
        try:
            with open(
                os.path.join(
                    self.out_dir, f"corpus_index_{self.corpus_identifier}.pkl"
                ),
                "rb",
            ) as f:
                corpus_index = pickle.load(f)

            with open(
                os.path.join(
                    self.out_dir, f"db_table_ids_{self.corpus_identifier}.pkl"
                ),
                "rb",
            ) as f:
                # stored separately as hnsw only takes int indices
                db_table_ids = pickle.load(f)

            # generate s hypothetical schemas for given query
            hypothetical_schemas_query = self.generate_hypothetical_schemas(query)

            hypothetical_schema_embeddings = []
            for hypothetical_schema in hypothetical_schemas_query:
                hypothetical_schema_embeddings += [
                    self.embed_table(table=hypothetical_schema)
                ]

            if self.with_query:
                hypothetical_schema_embeddings += [self.embed_query(query=query)]

            # print("input query is: ", query)
            # print("generated the following schemas: ", hypothetical_schemas_query)

            query_vector = np.array(hypothetical_schema_embeddings)

            # adapt inputs conditioned on single aggregated vector query or multi-table query
            if self.aggregated:
                # construct single 'prototype' table query vector (as in original HyDE method)
                query_vector = query_vector.mean(axis=0)
            else:
                # if doing multi-table retrieval, reset k per table to ensure total maximum is top_k
                top_k = int(top_k / self.num_schemas)

            # Retrieve tables most similar to input query
            # Returns nested vector of size len(query_vector)
            retrieved_ids, distances = corpus_index.knn_query(
                query_vector,
                k=top_k,
            )

            if not self.aggregated:
                # Get original table_ids (table names) from the retrieved
                # integer identifiers for each in s hypothetical schemas
                retrieved_full_ids = []
                for i in range(self.num_schemas):
                    retrieved_full_ids += [db_table_ids[id] for id in retrieved_ids[i]]
            else:
                retrieved_full_ids = [db_table_ids[id] for id in retrieved_ids[0]]

            return retrieved_full_ids
        except Exception as e:
            print(f"encountered error on dataset: {dataset_name} and query: {query}", e)

            return []

    def embed_query(self, query: str, id: Tuple[int, str] = None) -> List[List]:
        """Embed query using given embedding model."""
        try:
            if self.model_name == "openai":
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=query,
                )

                return response.data[0].embedding

            if self.model_name == "tapas":
                input = self.tokenizer(
                    table=pd.DataFrame(),
                    queries=[query],
                    return_tensors="pt",
                )

                embedding = (
                    self.model(**input)
                    .last_hidden_state[0]
                    .mean(axis=0)
                    .detach()
                    .numpy()
                    .tolist()
                )

                return embedding

        except Exception as e:
            print("error on: ", id, e)
            return []

    def embed_table(self, table: List[list], id: Tuple[int, str] = None) -> List[List]:
        """Embed table using given embedding model."""
        try:
            if self.model_name == "openai":
                table_str = utils.markdown_table_str(table, num_rows=self.num_rows)

                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=table_str,
                )

                return response.data[0].embedding

            if self.model_name == "tapas":
                # here, the table is actually a nested list
                table_df = pd.DataFrame(table[1 : self.num_rows + 1], columns=table[0])

                input = self.tokenizer(
                    table=table_df.astype(str),
                    queries=[""],
                    truncation=True,
                    return_tensors="pt",
                )

                embedding = (
                    self.model(**input)
                    .last_hidden_state[0]
                    .mean(axis=0)
                    .detach()
                    .numpy()
                    .tolist()
                )

                return embedding

        except Exception as e:
            print("error on: ", id, e)
            return []

    def generate_hypothetical_schemas(self, query: str) -> List[List]:
        """Generate a hypothetical schema relevant to answer the query."""

        client = instructor.from_openai(self.client)
        response_model = ResponseFormat

        # TODO: ensure correctness of output type and dimension
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {
                    "role": "system",
                    # todo: add table values
                    # todo: expand to multi-table
                    "content": """You are an assistant skilled in generating real-world database schemas and tables.""",
                },
                {
                    "role": "user",
                    "content": self.get_hyse_prompt(
                        query=query, with_rows=self.num_rows > 0
                    ),
                },
            ],
            response_model=response_model,
            temperature=0,
        )

        hypothetical_schemas = response.schemas

        return hypothetical_schemas

    def embed_corpus(self, dataset_name: str, corpus: Iterable[Dict]):
        """
        Function to embed the given corpus. This will be called in the evaluation pipeline before any retrieval.

        Parameters:
            dataset_name (str): the name of the corpus dataset.

            corpus (Iterable[Dict[str, List]]): an iterable of dicts, each being a batch of entries in the corpus dataset,
            containing database id, table id, the table contents (which the user can assume is in the format of self.expected_corpus_format), and context metadata (in these exact keys).

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
                embedded_corpus[tup_id] = self.embed_table(table=table, id=tup_id)

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

    def get_hyse_prompt(self, query: str, with_rows: bool = False):
        # with rows or without
        hyse_prompts = {
            False: f"""
                Generate {self.num_schemas} table headers with different semantics and at least 5 columns of tables that could answer or verify the following query: {query}.

                Return a nested list of size {self.num_schemas} in which each list contains a nested list representing the table header.
                Only generate table headers, do not provide surrounding numbers or text.

                For example, for two table headers to answer queries about the business performance of an energy company, you generate:
                [[['Quarter', 'Total Revenue (USD)', 'Net Profit (USD)', 'EBITDA (USD)', 'Total Energy Sold (MWh)', 'Customer Growth Rate (%)']],
                [['Period', 'Product', 'Sales Volume', 'Revenue', 'Market Share']]]
            """,
            True: f"""
                Generate exactly {self.num_schemas} tables with different semantics and at least 5 columns, but exactly {self.num_rows} rows that could answer or verify the following query: {query}.

                Return a nested list of size {self.num_schemas} in which each list contains a nested list representing a table.
                Each nested table list starts with a list column names, followed by lists representing the rows.
                Rows cannot contain 'None' values. Only generate tables, do not provide surrounding numbers or text.

                For example, for 2 tables with 1 row to answer queries about the business performance of an energy company, you could generate:

                [[['Quarter', 'Total Revenue (USD)', 'Net Profit (USD)', 'EBITDA (USD)', 'Total Energy Sold (MWh)', 'Customer Growth Rate (%)'], ['Q1 2024', '300M', '45M', '80M', '2.5M', 3]],
                [['Quarter', 'Product', 'Sales Volume', 'Revenue', 'Market Share'],['Q1', 2024, 'Electricity', 1200000, 150000000, 25]]]
            """,
        }

        return hyse_prompts[with_rows]
