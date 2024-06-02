from dataset_loaders.AbsDatasetLoader import AbsDatasetLoader
from dictionary_keys import *
from typing import Union, List

from datasets import load_dataset, DatasetDict
from pathlib import Path


class GenericDatasetLoader(AbsDatasetLoader):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        datafile_ext: str = None,
        table_col_name: str = TABLE_COL_NAME,
        table_id_col_name: str = TABLE_ID_COL_NAME,
        database_id_col_name: str = DATABASE_ID_COL_NAME,
        query_col_name: str = QUERY_COL_NAME,
        query_id_col_name: str = QUERY_ID_COL_NAME,
        answer_col_name: str = ANSWER_COL_NAME,
        splits: Union[str, List[str]] = "test",
        data_directory: str = None,
        query_type: str = "",
        **kwargs
    ):
        """
        Constructor for a generic dataset loader that loads from a local directory

        Parameters:
            dataset_path (str): the path to the directory where the dataset files reside. an example dataset files orgnanization for target:
                dataset_path/
                ├── corpus/
                │   ├── train.csv
                │   └── test.csv
                └── queries/
                    ├── train.csv
                    └── test.csv
            Dataset file formats supported: csv, json, parquet, etc
        """
        super().__init__(
            dataset_name=dataset_name,
            table_col_name=table_col_name,
            table_id_col_name=table_id_col_name,
            database_id_col_name=database_id_col_name,
            query_col_name=query_col_name,
            query_id_col_name=query_id_col_name,
            answer_col_name=answer_col_name,
            splits=splits,
            data_directory=data_directory,
            query_type=query_type,
            **kwargs,
        )
        self.dataset_path = Path(dataset_path)
        self.corpus_path = self.dataset_path / "corpus"
        self.queries_path = self.dataset_path / "queries"
        self.datafile_ext = datafile_ext

    def _load_corpus(self) -> None:
        if not self.corpus:
            self.corpus = DatasetDict()
        for split in self.splits:
            if split not in self.corpus:
                self.corpus[split] = load_dataset(
                    path=str(self.corpus_path), split=split
                )

    def _load_queries(self) -> None:
        if not self.queries:
            self.queries = DatasetDict()
        for split in self.splits:
            if split not in self.queries:
                self.queries[split] = load_dataset(
                    path=str(self.queries_path), split=split
                )
