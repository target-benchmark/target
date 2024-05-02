from retrievers.RetrieversDataModels import RetrievalResultDataModel
from typing import List

from abc import ABC, abstractmethod


class AbsRetrieverBase(ABC):
    """
    A base class for all Target Retrievers. Serves as a organization class, no function signatures are defined here, delegated to `AbsStandardizedEmbeddingRetriever` and `AbsCustomEmbeddingRetriever`
    """

    def __init__(self, expected_corpus_format: str = "nested array"):
        """
        Parameters:
            expected_corpus_format (str, optional): a string indicating what corpus format (ie nested array, dictionary, pandas df, etc.) the `embed_corpus` function expects from its input.
        """
        self.expected_corpus_format = expected_corpus_format

    def get_expected_corpus_format(self) -> str:
        """
        Returns the expected corpus format.
        """
        return self.expected_corpus_format

    @abstractmethod
    def retrieve(self, *args, **kwargs) -> List[str]:
        """
        The essential function for any Target Retriever. User have to implement this for the retriever class to work with evaluation pipeline.
        Returns:
            a dictionary mapping the query IDs to the list of possible tables retrieved.
        """
        pass

    @abstractmethod
    def retrieve_batch(self, *args, **kwargs) -> List[RetrievalResultDataModel]:
        pass
