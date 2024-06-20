import requests
from dataset_loaders.utils import (
    InMemoryDataFormat,
    QueryType,
    array_of_arrays_to_df,
    array_of_arrays_to_dict,
    set_in_memory_data_format,
    set_query_type,
    enforce_split_literal,
    write_table_to_path,
    convert_corpus_entry_to_df,
    convert_corpus_entry_to_dict,
)
from dictionary_keys import *

from abc import ABC, abstractmethod
from datasets import Dataset
from dataset_loaders.AbsDatasetLoader import AbsDatasetLoader
import gdown
import os
from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Literal, Tuple
import zipfile

file_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
default_download_dir = os.path.join(file_dir, ".text_2_sql_datasets")
default_spider_path = os.path.join(default_download_dir, "spider")
default_spider_database_path = os.path.join(default_spider_path, "test_database")
spider_download_url = "https://drive.google.com/uc?id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m"

default_bird_path = os.path.join(default_download_dir, "bird")
default_bird_database_path = os.path.join(default_bird_path, "dev_databases")
bird_download_url = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"


class Text2SQLDatasetLoader(AbsDatasetLoader):
    """
    The abstrack super class of target dataset loaders.
    This class contains implementations of utility functions shared by subclasses,
    but the loading functions are left as abstract so the subclasses will need to implement them separately.
    """

    def __init__(
        self,
        dataset_name: str,
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
        assert (
            split == "test"
        ), f"currently only the test split is supported for text2sql"
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            data_directory=data_directory,
            query_type="Text to SQL",
            kwargs=kwargs,
        )
        self.corpus: Dict = None
        self.queries: Dict = None
        if dataset_name == "spider":
            self._download_spider()
        elif dataset_name == "bird":
            self._download_bird()

    def _download_spider(self):
        if not os.path.exists(default_download_dir):
            os.makedirs(default_download_dir, exist_ok=True)
        if os.path.exists(default_spider_path):
            return
        path_to_zip = os.path.join(default_download_dir, "spider.zip")
        gdown.download(spider_download_url, output=path_to_zip, quiet=False)
        with zipfile.ZipFile(path_to_zip, "r") as zip_ref:
            # Unzip all the contents
            zip_ref.extractall(default_download_dir)
        os.remove(path_to_zip)
        mac_configs = os.path.join(default_download_dir, "__MACOSX")
        if os.path.exists(mac_configs):
            shutil.rmtree(mac_configs)

    def _download_bird(self):
        if not os.path.exists(default_download_dir):
            os.makedirs(default_download_dir, exist_ok=True)
        if os.path.exists(default_bird_path):
            return
        path_to_zip = os.path.join(default_download_dir, "bird.zip")
        with requests.get(bird_download_url, stream=True) as response:
            response.raise_for_status()  # This will raise an exception for HTTP error codes
            with open(path_to_zip, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=8192
                ):  # Adjust chunk size as needed
                    file.write(chunk)
        with zipfile.ZipFile(path_to_zip, "r") as zip_ref:
            # Unzip all the contents
            zip_ref.extractall(default_download_dir)
        os.rename(os.path.join(default_download_dir, "dev"), default_bird_path)

        path_to_db_zip = os.path.join(default_bird_path, "dev_databases.zip")
        with zipfile.ZipFile(path_to_db_zip, "r") as zip_ref:
            # Unzip all the contents
            zip_ref.extractall(default_bird_path)
        os.remove(path_to_zip)
        os.remove(path_to_db_zip)
        mac_configs = os.path.join(default_download_dir, "__MACOSX")
        if os.path.exists(mac_configs):
            shutil.rmtree(mac_configs)

    def _load_corpus(self) -> None:
        pass

    def _load_queries(self) -> None:
        pass

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

    def get_queries_for_task(self, batch_size: int = 64) -> Iterable[Dict]:
        if not self.queries:
            raise RuntimeError("Queries has not been loaded!")

        for i in range(0, len(self.queries[QUERY_COL_NAME]), batch_size):
            res = {}
            # Use list comprehensions to extract each column
            res[QUERY_COL_NAME] = self.queries[QUERY_COL_NAME][i : i + batch_size]
            res[QUERY_ID_COL_NAME] = self.queries[QUERY_ID_COL_NAME][i : i + batch_size]
            res[DATABASE_ID_COL_NAME] = self.queries[DATABASE_ID_COL_NAME][
                i : i + batch_size
            ]
            res[TABLE_ID_COL_NAME] = self.queries[TABLE_ID_COL_NAME][i : i + batch_size]
            res[ANSWER_COL_NAME] = self.queries[ANSWER_COL_NAME][i : i + batch_size]
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

    def get_queries(self) -> List[Dict]:
        """
        get the queries of the loaded dataset. if the dataset has not been loaded, raise an error.

        Returns:
            a Dataset object

        Raises:
            a runtime error if queries has not been loaded yet.
        """
        if not self.queries:
            raise RuntimeError("Queries datasets have not been loaded!")
        return self.queries

    def get_corpus_header(self) -> List[str]:
        """
        returns the header of this dataset's corpus

        Returns:
            a dictionary containing the headers of the corpus
        """
        if not self.corpus:
            raise RuntimeError("Corpus datasets have not been loaded!")
        return list(self.corpus[0].keys)

    def get_queries_header(self) -> List[str]:
        """
        returns the header of this dataset's queries
        Returns:
            a dictionary containing the headers of the queries
        """
        if not self.queries:
            raise RuntimeError("Queries datasets have not been loaded!")
        return list(self.queries[0].keys)
