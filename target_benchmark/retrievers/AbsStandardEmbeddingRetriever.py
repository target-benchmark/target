from abc import abstractmethod
from typing import Dict, List

import numpy as np
from qdrant_client import QdrantClient

from target_benchmark.dictionary_keys import (
    CLIENT_KEY_NAME,
    METADATA_DB_ID_KEY_NAME,
    METADATA_TABLE_ID_KEY_NAME,
    QUERY_COL_NAME,
    QUERY_ID_COL_NAME,
)
from target_benchmark.retrievers.AbsRetrieverBase import AbsRetrieverBase
from target_benchmark.retrievers.RetrieversDataModels import RetrievalResultDataModel


class AbsStandardEmbeddingRetriever(AbsRetrieverBase):
    """
    This retriever class includes both an embed query and an embed corpus method. If the user choose to inherit their retriever after this class, they need to implement both functions. The user implementated retriever is not expected to persist any data. Instead, as long as the embed corpus and embed query functions return a vector (list of floats) of the same dimension, the storage of the data for the evaluation will be dealt with automatically.

    NOTE: Storage of data is done with an in memory instance of a Qdrant vector database, which means it is not persisted across calls to `TARGET.run`.

    Some reasons to inherit from this class as opposed to `AbsCustomEmbeddingRetreiver`
    - the embedding of your tool is simply a vector or array of floats. example: the retriever is just an embedding model that produces vectors.
    - your retrieval system doesn't need any specific persistence formats or folder structures to work.

    To inherit from this class, fill out the `embed_query` and `embed_corpus` functions.
    """

    def __init__(self, expected_corpus_format: str = "nested array"):
        """
        Parameters:
            expected_corpus_format (str, optional): a string indicating what corpus format (ie nested array, dictionary, pandas df, etc.) the `embed_corpus` function expects from its input.
        """
        super().__init__(expected_corpus_format=expected_corpus_format)

    def retrieve_batch(
        self,
        queries: Dict[str, List],
        dataset_name: str,
        top_k: int,
        **kwargs,
    ) -> List[RetrievalResultDataModel]:
        retrieval_results = []
        if CLIENT_KEY_NAME not in kwargs:
            raise KeyError(
                f"missing key {CLIENT_KEY_NAME} in kwargs. must be included to use standardized embedding retriever."
            )
        client: QdrantClient = kwargs.get(CLIENT_KEY_NAME)
        for query_id, query_str in zip(
            queries[QUERY_ID_COL_NAME], queries[QUERY_COL_NAME]
        ):
            result = client.search(
                collection_name=dataset_name,
                query_vector=self.embed_query(query_str, dataset_name, **kwargs),
                limit=top_k,
                with_payload=True,
            )
            retrieval_results.append(
                RetrievalResultDataModel(
                    dataset_name=dataset_name,
                    query_id=query_id,
                    retrieval_results=[
                        (
                            scored_point.payload[METADATA_DB_ID_KEY_NAME],
                            scored_point.payload[METADATA_TABLE_ID_KEY_NAME],
                        )
                        for scored_point in result
                    ],
                )
            )
        return retrieval_results

    @abstractmethod
    def embed_query(
        self,
        query: str,
        dataset_name: str,
        **kwargs,
    ) -> np.ndarray:
        """
        Given a query, return the query embedding for searching.

        Parameters:

            queries (str): the actual query string.

            dataset_name (str): identifier for the dataset that these queries come from. since retrieval evaluation can be done for multiple datasets, use this as a way of choosing which dataset's corpus to retrieve from.

        Returns:
            the embeddings for the query
        """
        pass

    @abstractmethod
    def embed_corpus(self, dataset_name: str, corpus_entry: Dict) -> np.ndarray:
        """
        The function to embed the given corpus. This will be called in the evaluation pipeline before any retrieval. The corpus given will be in the same format as self.expected_corpus_format for flexibility.

        Parameters:
            dataset_name (str): the name of the corpus dataset.
            corpus (Dict): entry in the corpus dataset, containing database id, table id, the table contents (which the user can assume is in the format of self.expected_corpus_format), and context metadata (with these exact keys in the dictionary).

        Returns:
            List[float]: embedding of the passed in table
        """
        pass
