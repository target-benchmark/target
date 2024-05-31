from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest, ScoredPoint
from dataset_loaders.LoadersDataModels import QueryForTasksDataModel
from dictionary_keys import CLIENT_KEY_NAME, METADATA_KEY_NAME
from retrievers.AbsRetrieverBase import AbsRetrieverBase
from retrievers.RetrieversDataModels import RetrievalResultDataModel

from abc import abstractmethod
from typing import List, Dict, Iterable


class AbsStandardizedEmbeddingRetriever(AbsRetrieverBase):
    """
    This retriever class provides both a retrieve and embed method. If the user choose to inherit their custom class after this, they need to implement both functions. The retrieve class will now take in an additional `corpus_embedding` parameter, so they don't need to deal with embedded persistence explicitly here, as the embeddings will be provided at retrieval time.

    Some reasons to inherit from this class as opposed to `AbsCustomEmbeddingRetreiver`
    - the embedding of your tool is simply a vector or array of floats.
    - your retrieval system doesn't need any specific persistence formats or folder structure to work.
    """

    def __init__(self, expected_corpus_format: str = "nested array"):
        """
        Parameters:
            expected_corpus_format (str, optional): a string indicating what corpus format (ie nested array, dictionary, pandas df, etc.) the `embed_corpus` function expects from its input.
        """
        super().__init__(expected_corpus_format=expected_corpus_format)

    def retrieve_batch(
        self,
        queries: List[QueryForTasksDataModel],
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
        for query in queries:
            result = client.search(
                collection_name=dataset_name,
                query_vector=self.embed_query(query.query, dataset_name, **kwargs),
                limit=top_k,
                with_payload=True,
            )
            retrieval_results.append(
                RetrievalResultDataModel(
                    dataset_name=dataset_name,
                    query_id=query.query_id,
                    retrieval_results=[
                        scored_point.payload[METADATA_KEY_NAME]
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
    ) -> List[float]:
        """
        Given a query, return the query embedding for searching.

        Parameters:

            queries (str): the actual query string.

            dataset_name (str): identifier for the dataset that these queries come from. since retrieval evaluation can be done for multiple datasets, use this as a way of choosing which dataset's corpus to retrieve from.
            any additional kwargs you'd like to include.

        Returns:
            the embeddings for the query
        """
        pass

    @abstractmethod
    def embed_corpus(self, dataset_name: str, table) -> List[float]:
        """
        The function to embed the given corpus. This will be called in the evaluation pipeline before any retrieval. The corpus given will be in the same format as self.expected_corpus_format for flexibility.

        Parameters:
            dataset_name (str): the name of the corpus dataset.
            corpus (object): the table object (which the user can assume is in the format of self.expected_corpus_format).

        Returns:
            List[float]: embedding of the passed in table
        """
        pass
