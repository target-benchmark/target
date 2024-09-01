from abc import ABC, abstractmethod
from typing import List

from target_benchmark.retrievers.RetrieversDataModels import RetrievalResultDataModel


class AbsRetrieverBase(ABC):
    """
    A base class for all Target Retrievers. Serves as the super class to `AbsStandardizedEmbeddingRetriever` and `AbsCustomEmbeddingRetriever`
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
    def retrieve_batch(self, *args, **kwargs) -> List[RetrievalResultDataModel]:
        pass
