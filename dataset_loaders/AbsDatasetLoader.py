from dataset_loaders.utils import (
    InMemoryDataFormat,
    array_of_arrays_to_df,
    QueryType,
    set_in_memory_data_format,
    set_query_type,
    enforce_split_literal,
    write_table_to_path,
    convert_corpus_entry_to_df,
    convert_corpus_entry_to_dict,
)
from dataset_loaders.LoadersDataModels import QueryForTasksDataModel
from dictionary_keys import *

from abc import ABC, abstractmethod
import csv
from datasets import Dataset
from pathlib import Path
from typing import Iterable, Iterator, Literal, Tuple
from typing import Union, List, Dict


class AbsDatasetLoader(ABC):
    """
    The abstrack super class of target dataset loaders.
    This class contains implementations of utility functions shared by subclasses,
    but the loading functions are left as abstract so the subclasses will need to implement them separately.
    """

    def __init__(
        self,
        dataset_name: str,
        table_col_name: str = TABLE_COL_NAME,
        table_id_col_name: str = TABLE_ID_COL_NAME,
        database_id_col_name: str = DATABASE_ID_COL_NAME,
        context_col_name:str = CONTEXT_COL_NAME,
        query_col_name: str = QUERY_COL_NAME,
        query_id_col_name: str = QUERY_ID_COL_NAME,
        answer_col_name: str = ANSWER_COL_NAME,
        split: Literal["test", "train", "validation"] = "test",
        data_directory: str = None,
        query_type: str = None,
        **kwargs,
    ):
        """
        Constructor for the abstract data loader class.

        Parameters:
            dataset_name (str): a string for the name of the dataset. required

            table_col_name (str, optional): name of the column that contains the nested array tables. defaults to "table", you can leave as default if working with any TARGET datasets. If user were to use custom datasets with different header names, this name can be adjusted accordingly.

            table_id_col_name (str, optional): name of the column that contains the unique identifiers of the tables. defaults to "table_id" across all TARGET datasets.

            database_id_col_name (str, optional): name of the column that contains the database id, this column exists in both queries and corpus datasets, and TARGET datasets loaders work on the assumption that the column name remains the same across the queries and corpus datasets. defaults to "database_id".

            query_col_name (str, optional): name of the column that contains the query strings. defaults to "query".

            query_id_col_name (str, optional): name of the column that contains the query ids. defaults to "query_id".

            answer_col_name (str, optional): name of the column that contains the answer str. defaults to "answer".

            split (Literal["test", "train", "validation"], optional): split of the data you want to load. defaults to test, since some models may use the train split of existing datasets for training, we opt to use test for our evaluation purposes.

            data_directory (str, optional): a directory where data files are stored. you don't have to provide one if you don't need to persist the file after loading it.

            query_type (str, optional): the type of queries that the dataset focuses on, for example fact verification, table QA, text to sql, etc. defaults to None.

        Instance Variables:
            aside from instance variables of the same name & purposes as the parameters, there's also:

            self.corpus: a huggingface Dataset object containing the corpus dataset, remains None until load corpus is complete.
            self.queries: a huggingface Dataset object containing the queries dataset, remains None until load queries is complete.
        """

        self.dataset_name: str = dataset_name
        self.split = enforce_split_literal(split)
        if data_directory and Path(data_directory).suffix:
            raise ValueError(f"this path {data_directory} looks like a path to a file.")
        self.data_directory: str = data_directory
        self.table_col_name: str = table_col_name
        self.table_id_col_name: str = table_id_col_name
        self.database_id_col_name: str = database_id_col_name
        self.context_col_name: str = context_col_name
        self.query_col_name: str = query_col_name
        self.query_id_col_name: str = query_id_col_name
        self.answer_col_name: str = answer_col_name
        self.query_type: QueryType = set_query_type(query_type)
        self.corpus: Dataset = None
        self.queries: Dataset = None
        self.alt_corpus = None

    def load(self) -> None:
        if not self.corpus:
            self._load_corpus()
        if not self.queries:
            self._load_queries()

    @abstractmethod
    def _load_corpus(self) -> None:
        pass

    @abstractmethod
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
        cur_split_dataset = self.corpus[self.split]
        for batch in cur_split_dataset.iter(1):
            table_name = Path(batch[self.table_id_col_name][0])
            nested_array = batch[self.table_col_name][0]
            write_table_to_path(format, table_name, split_path, nested_array)

    def corpus_batch_to_tuple(self, batch):
        return (
                    batch[self.database_id_col_name],
                    batch[self.table_id_col_name],
                    batch[self.table_col_name],
                    batch[self.context_col_name]
                )

    def convert_corpus_table_to(
        self,
        output_format: str = "nested array",
        batch_size: int = 1,
    ) -> Iterable[Tuple]:
        """
        convert the corpus table to a specific format in memory.

        Parameters:

            output_format (str): the output class name, can be nest_array, pandas, etc.
            batch_size (int): number of tables to be outputted at once

        Returns:
            a generator. depending on the batch size, each yield produces a tuple of
            - a single values if batch size is 0
            - a list of values if batch size is greater than 0
            the order of the values in the yielded tuple will be
            (database id, table id, table content, context/metadata)
        """

        if not self.corpus:
            raise RuntimeError("Corpus has not been loaded!")

        in_memory_format = set_in_memory_data_format(output_format)
        if in_memory_format == InMemoryDataFormat.DF:
            converted_corpus = self.corpus.map(
                lambda entry: convert_corpus_entry_to_df(self.table_col_name, entry)
            )
        elif in_memory_format == InMemoryDataFormat.DICTIONARY:
            converted_corpus = self.corpus.map(
                lambda entry: convert_corpus_entry_to_dict(self.table_col_name, entry)
            )
        else:
            converted_corpus = self.corpus
        for batch in converted_corpus.iter(batch_size):
            yield self.corpus_batch_to_tuple(batch)

    def get_table_id_to_table(
        self,
    ) -> Dict[Tuple[int, str], List[List]]:
        mapping_dict = {}
        for entry in self.convert_corpus_table_to():
            key = (entry[self.database_id_col_name], entry[self.table_id_col_name]) 
            mapping_dict[entry[self.table_id_col_name]] = entry[self.table_col_name]
        return mapping_dict

    def get_queries_for_task(
        self, batch_size: int = 64
    ) -> Iterable[List[QueryForTasksDataModel]]:
        if not self.queries:
            raise RuntimeError("Queries has not been loaded!")

        for batch in self.queries.iter(batch_size):
            res_list = []
            query_ids = batch[self.query_id_col_name]
            queries = batch[self.query_col_name]
            database_ids = batch[self.database_id_col_name]
            table_ids = batch[self.table_id_col_name]
            answers = batch[self.answer_col_name]
            for query_id, query, database_id, table_id, answer in zip(
                query_ids, queries, database_ids, table_ids, answers
            ):
                res_list.append(
                    QueryForTasksDataModel(
                        table_id=table_id,
                        database_id=database_id,
                        query_id=query_id,
                        query=query,
                        answer=answer,
                    )
                )
            yield res_list

    def get_dataset_name(self) -> str:
        """
        returns the name of the dataset

        Returns:
            a string which is the name of the dataset
        """
        return self.dataset_name

    def get_corpus(self) -> Dataset:
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

    def get_queries(self) -> Dataset:
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
        return self.corpus.column_names

    def get_queries_header(self) -> List[str]:
        """
        returns the header of this dataset's queries
        Returns:
            a dictionary containing the headers of the queries
        """
        if not self.queries:
            raise RuntimeError("Queries datasets have not been loaded!")
        return self.queries.column_names
