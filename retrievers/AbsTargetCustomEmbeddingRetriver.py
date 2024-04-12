from typing import Iterable, Iterator
from retrievers.AbsTargetRetrieverBase import AbsTargetRetrieverBase
from dataset_loaders.AbsTargetDatasetLoader import AbsTargetDatasetLoader
from abc import abstractmethod


class AbsTargetCustomEmbeddingRetriver(AbsTargetRetrieverBase):
    """
    This interface includes the retrieve method and an encode method that doesn't expect a return value. If your retrieval tool already has table embedding/encoding persistence built in, this is the preferred class to inherit from for your custom retriever, as you can just ignore the encode method. At retrieval time, it is assumed that the **table embeddings are no longer needed to be provided** for the retrieval to work.
    Reasons for providing this encoding method is:
    - it's not expected of the users to deal with setting up `TargetDatasetloaders` directly, since at the time of instantiation it may be unclear which datasets needs to be preprocessed. We'd like to delegate this responsibilty to the `TargetEvaluator` class during the eval process.
    - remain symmetric to the `AbsTargetStandardizedEmbeddingRetriever`
    Some possible reasons to inherit from this class and not `AbsTargetRetrieverWithEncoding`:
    - you have a custom format of embedding for the tables (ie directory structure, file formats, etc).
    - your tool already deals with the persistence of the embedding, in which case the embedding method can just pass & do nothing.
    """

    def __init__(self, expected_corpus_format: str = "nested array"):
        """
        Parameters:
            expected_corpus_format (str, optional): a string indicating what corpus format (ie nested array, dictionary, pandas df, etc.) the `embed_corpus` function expects from its input.
        """
        self.expected_corpus_format = expected_corpus_format

    def retrieve_batch(
        self,
        queries: dict[int, str],
        dataset_name: str,
        top_k: int,
        **kwargs,
    ) -> dict[int, list[str]]:
        retrieval_results = {}
        for query_id, query_str in queries.items():
            retrieval_results[query_id] = self.retrieve(
                query_str, dataset_name, top_k, kwargs
            )
        return retrieval_results

    @abstractmethod
    def retrieve(
        self,
        query: str,
        dataset_name: str,
        top_k: int,
        **kwargs,
    ) -> list[str]:
        """
        Directly retrieves the corresponding tables for the query. Works under the assumption that the embeddings are available when this function is called, and the retriever should be able to get the right tables with the query provided without any additional information about the corpus.

        Parameters:
            query (str): the actual query string.

            dataset_name (str): identifier for the dataset that these queries come from. since retrieval evaluation can be done for multiple datasets, use this as a way of choosing which dataset's corpus to retrieve from.

            top_k (int): the top k tables to retrieve for each query

            any additional kwargs you'd like to include.

        Returns:
            list[str]: the list of table ids of the retrieved tables.
        """
        pass

    @abstractmethod
    def embed_corpus(self, dataset_name: str, corpus: Iterable[dict]):
        """
        The function to embed the given corpus. This will be called in the evaluation pipeline before any retrieval. The corpus given will be in the same format as self.expected_corpus_format for flexibility.

        Parameters:
            dataset_name (str): the name of the corpus dataset.
            corpus (Iterable[dict[str, object]]): an iterable of dictionaries, each dictionary mapping the table id to the table object (which the user can assume is in the format of self.expected_corpus_format).

        Returns:
            nothing. the persistence of the embedding must be dealt with the logic of this function itself, and the `retrieve` function should also know about the embedding results of this function so that retrieval can be done.
        """
        pass
