from typing import Iterable
from dataset_loaders.LoadersDataModels import QueryForTasksDataModel
from retrievers.AbsRetrieverBase import AbsRetrieverBase
from retrievers.RetrieversDataModels import RetrievalResultDataModel
from abc import abstractmethod
from numpy.typing import NDArray, ArrayLike
from typing import List, Dict, Iterable


class AbsStandardizedEmbeddingRetriever(AbsRetrieverBase):
    """
    This retriever class provides both a retrieve and embed method. If the user choose to inherit their custom class after this, they need to implement both functions. The retrieve class will now take in an additional `corpus_embedding` parameter, so they don't need to deal with embedded persistence explicitly here, as the embeddings will be provided at retrieval time.

    Some reasons to inherit from this class as opposed to `AbsCustomEmbeddingRetreiver`
    - the embedding of your tool is simply a vector or array like object.
    - your retrieval system doesn't need any specific persistence formats or folder structure to work, as long as the corpus embedding is given that's all you need.
    """

    def __init__(self, expected_corpus_format: str = "nested array"):
        """
        Parameters:
            expected_corpus_format (str, optional): a string indicating what corpus format (ie nested array, dictionary, pandas df, etc.) the `embed_corpus` function expects from its input.
        """
        super().__init__(expected_corpus_format=expected_corpus_format)

    def retrieve_batch(
        self,
        corpus_embedding,
        queries: List[QueryForTasksDataModel],
        dataset_name: str,
        top_k: int,
        **kwargs,
    ) -> List[RetrievalResultDataModel]:
        retrieval_results = []
        for query in queries:
            retrieval_results.append(
                RetrievalResultDataModel(
                    dataset_name=dataset_name,
                    query_id=query.query_id,
                    retrieval_results=self.retrieve(
                        corpus_embedding, query.query_str, dataset_name, top_k, kwargs
                    ),
                )
            )
        return retrieval_results

    @abstractmethod
    def retrieve(
        self,
        corpus_embedding,
        query: str,
        dataset_name: str,
        top_k: int,
        **kwargs,
    ) -> List[str]:
        """
        Given a corpus embedding, retrieves the corresponding tables for the given query.

        Parameters:
            corpus_embedding: embedding of the corpus (created by `embed_corpus`). TODO: figure out the format for this

            queries (str): the actual query string.

            dataset_name (str): identifier for the dataset that these queries come from. since retrieval evaluation can be done for multiple datasets, use this as a way of choosing which dataset's corpus to retrieve from.

            top_k (int): the top k tables to retrieve for each query

            any additional kwargs you'd like to include.

        Returns:
            List[str]: the list of table ids of the retrieved tables.
        """
        pass

    @abstractmethod
    def embed_corpus(
        self, dataset_name: str, corpus: Iterable[dict]
    ) -> Dict[str, ArrayLike]:
        """
        The function to embed the given corpus. This will be called in the evaluation pipeline before any retrieval. The corpus given will be in the same format as self.expected_corpus_format for flexibility.

        Parameters:
            dataset_name (str): the name of the corpus dataset.
            corpus (Dict[str, object]): a dictionary mapping the table id to the table object (which the user can assume is in the format of self.expected_corpus_format).

        Returns:
            Dict[str, ArrayLike]: a mapping between the table id and the embedding of that table. the embedding is restricted to an ArrayLike
        """
        pass
