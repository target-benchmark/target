from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

from datasets import Dataset

from target_benchmark.dataset_loaders.utils import (
    InMemoryDataFormat,
    QueryType,
    array_of_arrays_to_df,
    array_of_arrays_to_dict,
    enforce_split_literal,
    get_random_tables,
    set_in_memory_data_format,
    set_query_type,
    write_table_to_path,
)
from target_benchmark.dictionary_keys import (
    CONTEXT_COL_NAME,
    DATABASE_ID_COL_NAME,
    TABLE_COL_NAME,
    TABLE_ID_COL_NAME,
)


class AbsDatasetLoader(ABC):
    """
    The abstrack super class of target dataset loaders.
    This class contains implementations of utility functions shared by subclasses,
    but the loading functions are left as abstract so the subclasses will need to implement them separately.
    """

    def __init__(
        self,
        dataset_name: str,
        num_tables: int = None,
        split: Literal["test", "train", "validation"] = "test",
        data_directory: str = None,
        query_type: str = None,
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

        self.dataset_name: str = dataset_name
        self.num_tables: int = num_tables
        self.split = enforce_split_literal(split)
        if data_directory and Path(data_directory).suffix:
            raise ValueError(f"this path {data_directory} looks like a path to a file.")
        self.data_directory: str = data_directory
        self.query_type: QueryType = set_query_type(query_type)
        self.corpus: Dataset = None
        self.queries: Dataset = None

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

    def persist_corpus_to(self, format: Literal["csv", "json"], path: str = None) -> None:
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
                raise ValueError("No path for persistence is specified!")
            path = self.data_directory

        path_to_write_to = Path(path)
        if path_to_write_to.suffix:
            raise ValueError(f"this path {path_to_write_to} looks like a path to a file.")
        if not path_to_write_to.exists():
            path_to_write_to.mkdir(parents=True, exist_ok=True)

        split_path = path_to_write_to / self.split
        if not split_path.exists():
            split_path.mkdir(parents=True, exist_ok=True)
        for entry in self.corpus:
            table_name = Path(entry[TABLE_ID_COL_NAME])
            nested_array = entry[TABLE_COL_NAME]
            write_table_to_path(format, table_name, split_path, nested_array)

    def _convert_corpus_to_dict(self):
        return self.corpus.to_dict()

    def convert_corpus_table_to(
        self,
        output_format: str = "nested array",
        batch_size: int = 1,
        num_tables: int = None,
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
        converted_corpus = self._convert_corpus_to_dict()
        if in_memory_format == InMemoryDataFormat.DF:
            df_tables = list(map(array_of_arrays_to_df, self.corpus[TABLE_COL_NAME]))
            converted_corpus[TABLE_COL_NAME] = df_tables
        elif in_memory_format == InMemoryDataFormat.DICTIONARY:
            dict_tables = list(map(array_of_arrays_to_dict, self.corpus[TABLE_COL_NAME]))
            converted_corpus[TABLE_COL_NAME] = dict_tables
        count_tables = num_tables or self.num_tables
        if count_tables is not None:
            converted_corpus = get_random_tables(converted_corpus, max(0, min(count_tables, self.get_corpus_size())))
        for i in range(0, len(converted_corpus[TABLE_COL_NAME]), batch_size):
            batch = {}
            # Use list comprehensions to extract each column
            batch[TABLE_COL_NAME] = converted_corpus[TABLE_COL_NAME][i : i + batch_size]
            batch[DATABASE_ID_COL_NAME] = converted_corpus[DATABASE_ID_COL_NAME][i : i + batch_size]
            batch[TABLE_ID_COL_NAME] = converted_corpus[TABLE_ID_COL_NAME][i : i + batch_size]
            batch[CONTEXT_COL_NAME] = converted_corpus[CONTEXT_COL_NAME][i : i + batch_size]
            yield batch

    def get_table_id_to_table(
        self,
    ) -> Dict[Tuple[str, str], List[List]]:
        mapping_dict = {}
        for entry in self.convert_corpus_table_to():
            for database_id, table_id, table in zip(
                entry[DATABASE_ID_COL_NAME],
                entry[TABLE_ID_COL_NAME],
                entry[TABLE_COL_NAME],
            ):
                key = (str(database_id), str(table_id))
                mapping_dict[key] = table
        return mapping_dict

    def get_queries_for_task(self, batch_size: int = 64) -> Iterable[Dict]:
        if not self.queries:
            raise RuntimeError("Queries has not been loaded!")
        for batch in self.queries.iter(batch_size):
            yield batch

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
        return self.corpus.num_rows

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

    def get_queries_size(self) -> int:
        """
        Get the number of entries in the queries.

        Returns:
            number of queries.

        Raises:
            a runtime error if queries has not been loaded yet.
        """
        if not self.queries:
            raise RuntimeError("Queries datasets have not been loaded!")
        return self.queries.num_rows

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
