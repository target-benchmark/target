from typing import Dict, Iterable, List, Tuple
from retrievers import AbsCustomEmbeddingRetriever


class LlamaIndexRetriever(AbsCustomEmbeddingRetriever):

    def __init__(self):
        """
        Parameters:
            expected_corpus_format (str, optional): a string indicating what corpus format (ie nested array, dictionary, pandas df, etc.) the `embed_corpus` function expects from its input.
        """
        super().__init__(expected_corpus_format="dataframe")

    def retrieve(
        self,
        query: str,
        dataset_name: str,
        top_k: int,
        **kwargs,
    ) -> List[Tuple]:
        pass

    def embed_corpus(self, dataset_name: str, corpus: Iterable[Dict]) -> None:
        pass
