from dataset_loaders.AbsDatasetLoader import AbsDatasetLoader
from dictionary_keys import *
from typing import Literal

from datasets import load_dataset


class HFDatasetLoader(AbsDatasetLoader):
    def __init__(
        self,
        dataset_name: str,
        hf_corpus_dataset_path: str,
        hf_queries_dataset_path: str,
        table_col_name: str = TABLE_COL_NAME,
        table_id_col_name: str = TABLE_ID_COL_NAME,
        database_id_col_name: str = DATABASE_ID_COL_NAME,
        context_col_name:str = CONTEXT_COL_NAME,
        query_col_name: str = QUERY_COL_NAME,
        query_id_col_name: str = QUERY_ID_COL_NAME,
        answer_col_name: str = ANSWER_COL_NAME,
        split: Literal["test", "train", "validation"] = "test",
        data_directory: str = None,
        query_type: str = "",
        **kwargs
    ):
        super().__init__(
            dataset_name=dataset_name,
            table_col_name=table_col_name,
            table_id_col_name=table_id_col_name,
            database_id_col_name=database_id_col_name,
            context_col_name=context_col_name,
            query_col_name=query_col_name,
            query_id_col_name=query_id_col_name,
            answer_col_name=answer_col_name,
            split=split,
            data_directory=data_directory,
            query_type=query_type,
            **kwargs
        )
        """
        Constructor for a generic dataset loader that loads from a huggingface dataset.
        Parameters:
            hf_corpus_dataset_path (str): the path to your huggingface hub corpus dataset. it will look something like target-benchmark/fetaqa-corpus (namespace/dataset-name)
            hf_queries_dataset_path (str): the path to your huggingface hub queries dataset path. 
        """

        self.hf_corpus_dataset_path = hf_corpus_dataset_path
        self.hf_queries_dataset_path = hf_queries_dataset_path

    def _load_corpus(self) -> None:
        if not self.corpus:
            self.corpus = load_dataset(
                path=self.hf_corpus_dataset_path, split=self.split
            )

    def _load_queries(self) -> None:
        if not self.queries:
            self.queries = load_dataset(
                path=self.hf_queries_dataset_path, split=self.split
            )
