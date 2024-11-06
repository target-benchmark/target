from pathlib import Path
from typing import Literal

from datasets import DatasetDict, load_dataset

from target_benchmark.dataset_loaders.AbsDatasetLoader import AbsDatasetLoader


class GenericDatasetLoader(AbsDatasetLoader):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        num_tables: int = None,
        datafile_ext: str = None,
        split: Literal["test", "train", "validation"] = "test",
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
            split=split,
            data_directory=data_directory,
            query_type=query_type,
            num_tables=num_tables**kwargs,
        )
        self.dataset_path = Path(dataset_path)
        self.corpus_path = self.dataset_path / "corpus"
        self.queries_path = self.dataset_path / "queries"
        self.datafile_ext = datafile_ext

    def _load_corpus(self) -> None:
        if not self.corpus:
            self.corpus = DatasetDict()
        if self.split not in self.corpus:
            self.corpus[self.split] = load_dataset(path=str(self.corpus_path), split=self.split)

    def _load_queries(self) -> None:
        if not self.queries:
            self.queries = DatasetDict()
        if self.split not in self.queries:
            self.queries[self.split] = load_dataset(path=str(self.queries_path), split=self.split)
