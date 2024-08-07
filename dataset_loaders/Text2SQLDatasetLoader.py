import json
import requests
from dataset_loaders.utils import (
    InMemoryDataFormat,
    array_of_arrays_to_df,
    array_of_arrays_to_dict,
    set_in_memory_data_format,
    write_table_to_path,
)
from dictionary_keys import *


from dataset_loaders import HFDatasetLoader

from huggingface_hub import snapshot_download

from pathlib import Path
from typing import Dict, Iterable, List, Literal


class Text2SQLDatasetLoader(HFDatasetLoader):
    """
    The abstrack super class of target dataset loaders.
    This class contains implementations of utility functions shared by subclasses,
    but the loading functions are left as abstract so the subclasses will need to implement them separately.
    """

    def __init__(
        self,
        dataset_name: str,
        hf_corpus_dataset_path: str,
        hf_queries_dataset_path: str,
        split: Literal["test", "train", "validation"] = "test",
        data_directory: str = None,
        **kwargs,
    ):
        """
        Constructor for the abstract data loader class.

        Parameters:
            dataset_name (str): a string for the name of the dataset. required

            split (Literal["test", "train", "validation"], optional): split of the data you want to load. defaults to test, since some models may use the train split of existing datasets for training, we opt to use test for our evaluation purposes.

            data_directory (str, optional): a directory where data files are stored. you don't have to provide one if you don't need to persist the file after loading it.

            query_type (str, optional): the type of queries that the dataset focuses on, for example fact verification, table QA, text to sql, etc. defaults to None.

        Instance Variables:
            aside from instance variables of the same name & purposes as the parameters, there's also:

            self.corpus: a huggingface Dataset object containing the corpus dataset, remains None until load corpus is complete.
            self.queries: a huggingface Dataset object containing the queries dataset, remains None until load queries is complete.
        """
        assert (
            dataset_name == "spider" or dataset_name == "bird"
        ), f"we don't allow customized text2sql datasets yet. try spider or bird instead"

        super().__init__(
            dataset_name=dataset_name,
            hf_corpus_dataset_path=hf_corpus_dataset_path,
            hf_queries_dataset_path=hf_queries_dataset_path,
            split=split,
            data_directory=data_directory,
            query_type="Text to SQL",
            kwargs=kwargs,
        )
        self.corpus: Dict = None
        self.path_to_database_dir: str = None

    def _load_corpus(self) -> None:
        path_to_data_dir = snapshot_download(
            repo_id=self.hf_corpus_dataset_path, repo_type="dataset"
        )
        path_to_context = Path(
            path_to_data_dir, f"{self.dataset_name}-corpus-{self.split}.json"
        )
        self.path_to_database_dir = Path(path_to_data_dir, f"{self.split}_database")
        with open(path_to_context, "r") as file:
            self.corpus = json.load(file)

    def persist_corpus_to(
        self, format: Literal["csv", "json"], path: str = None
    ) -> None:
        """
        Saves the tables in the corpus to a specified location and format.

        Parameters:
            format (Literal["csv", "json"]): The format in which to save the corpus (e.g., 'csv', 'json').
            path (str, optional): The file system path where the corpus should be saved. creates directory if the directory doesn't exist. if no path is given, defaults to self.data_directory. if self.data_directory is also None, throw ValueError.

        Raises:
            Runtime error if the corpus has not been loaded.
        """
        if not self.corpus:
            raise RuntimeError("Corpus has not been loaded!")

        if not path:
            if not self.data_directory:
                raise ValueError(f"No path for persistence is specified!")
            path = self.data_directory

        path_to_write_to = Path(path)
        if path_to_write_to.suffix:
            raise ValueError(
                f"this path {path_to_write_to} looks like a path to a file."
            )
        if not path_to_write_to.exists():
            path_to_write_to.mkdir(parents=True, exist_ok=True)

        split_path = path_to_write_to / self.split
        if not split_path.exists():
            split_path.mkdir(parents=True, exist_ok=True)
        for i in range(len(self.corpus[TABLE_COL_NAME])):
            table_name = Path(self.corpus[TABLE_ID_COL_NAME][i])
            nested_array = self.corpus[TABLE_ID_COL_NAME][i]
            write_table_to_path(format, table_name, split_path, nested_array)

    def convert_corpus_table_to(
        self,
        output_format: str = "nested array",
        batch_size: int = 1,
    ) -> Iterable[Dict]:
        """
        convert the corpus table to a specific format in memory.

        Parameters:

            output_format (str): the output class name, can be nest_array, pandas, etc.
            batch_size (int): number of tables to be outputted at once

        Returns:
            a generator. each yield produces a dictionary with keys "table_id", "table", "database_id", "context".
        """

        if not self.corpus:
            raise RuntimeError("Corpus has not been loaded!")

        in_memory_format = set_in_memory_data_format(output_format)
        converted_corpus = self.corpus.copy()
        if in_memory_format == InMemoryDataFormat.DF:
            df_tables = list(map(array_of_arrays_to_df, self.corpus[TABLE_COL_NAME]))
            converted_corpus[TABLE_COL_NAME] = df_tables
        elif in_memory_format == InMemoryDataFormat.DICTIONARY:
            dict_tables = list(
                map(array_of_arrays_to_dict, self.corpus[TABLE_COL_NAME])
            )
            converted_corpus[TABLE_COL_NAME] = dict_tables
        else:
            converted_corpus = self.corpus
        for i in range(0, len(self.corpus[TABLE_COL_NAME]), batch_size):
            res = {}
            # Use list comprehensions to extract each column
            res[TABLE_COL_NAME] = self.corpus[TABLE_COL_NAME][i : i + batch_size]
            res[DATABASE_ID_COL_NAME] = self.corpus[DATABASE_ID_COL_NAME][
                i : i + batch_size
            ]
            res[TABLE_ID_COL_NAME] = self.corpus[TABLE_ID_COL_NAME][i : i + batch_size]
            res[CONTEXT_COL_NAME] = self.corpus[CONTEXT_COL_NAME][i : i + batch_size]
            yield res

    def get_corpus(self) -> List[Dict]:
        """
        get the corpus of the loaded dataset. if the dataset has not been loaded, raise an error.

        Returns:
            a Dataset object

        Raises:
            a runtime error if corpus has not been loaded yet.

        """
        if not self.corpus:
            raise RuntimeError("Corpus datasets have not been loaded!")
        return self.corpus

    def get_corpus_size(self) -> int:
        """
        Get the number of tables in the corpus.

        Returns:
            number of tables.

        Raises:
            a runtime error if corpus has not been loaded yet.
        """
        if not self.corpus:
            raise RuntimeError("Corpus datasets have not been loaded!")
        return len(self.corpus)

    def get_corpus_header(self) -> List[str]:
        """
        returns the header of this dataset's corpus

        Returns:
            a dictionary containing the headers of the corpus
        """
        if not self.corpus:
            raise RuntimeError("Corpus datasets have not been loaded!")
        return list(self.corpus.keys())