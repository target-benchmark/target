from dataset_loaders.utils import array_of_arrays_to_df, interpret_numbers
from dataset_loaders.LoadersDataModels import QueryForTasksDataModel
from dictionary_keys import *

from abc import ABC, abstractmethod
import csv
from datasets import DatasetDict
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator
from typing import Union, List, Dict

class QueryType(Enum):
    TEXT_2_SQL = "Text to SQL"
    FACT_VERIFICATION = "Fact Verification"
    TABLE_QA = "Table Question Answering"
    OTHER = "Other"

def set_query_type(string_rep: str) -> QueryType:
    string_rep = string_rep.lower()
    if string_rep in QueryType.FACT_VERIFICATION.value.lower():
        return QueryType.FACT_VERIFICATION
    elif string_rep in QueryType.TABLE_QA.value.lower():
        return QueryType.TABLE_QA
    elif string_rep in QueryType.TEXT_2_SQL.value.lower():
        return QueryType.TEXT_2_SQL
    else:
        return QueryType.OTHER


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
        query_col_name: str = QUERY_COL_NAME,
        query_id_col_name: str = QUERY_ID_COL_NAME,
        answer_col_name: str = ANSWER_COL_NAME,
        splits: Union[str, List[str]] = "test",
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

            splits (Union[str, List[str]], optional): splits of the data you want to load. defaults to test, since some models may use the train split of existing datasets for training, we opt to use test for our evaluation purposes.

            data_directory (str, optional): a directory where data files are stored. you don't have to provide one if you don't need to persist the file after loading it.

            query_type (str, optional): the type of queries that the dataset focuses on, for example fact verification, table QA, text to sql, etc. defaults to None.

        Instance Variables:
            aside from instance variables of the same name & purposes as the parameters, there's also:

            self.corpus: a huggingface DatasetDict object containing the corpus dataset, remains None until load corpus is complete.
            self.queries: a huggingface DatasetDict object containing the queries dataset, remains None until load queries is complete.
        """

        self.dataset_name: str = dataset_name
        self.splits = splits
        if isinstance(splits, str):
            self.splits = [splits]
        if data_directory and Path(data_directory).suffix:
            raise ValueError(f"this path {data_directory} looks like a path to a file.")
        self.data_directory: str = data_directory
        self.table_col_name: str = table_col_name
        self.table_id_col_name: str = table_id_col_name
        self.database_id_col_name: str = database_id_col_name
        self.query_col_name: str = query_col_name
        self.query_id_col_name: str = query_id_col_name
        self.answer_col_name: str = answer_col_name
        self.query_type: QueryType = set_query_type(query_type)
        self.corpus: DatasetDict = None
        self.queries: DatasetDict = None
        self.alt_corpus = None

    
    def load(self, splits: Union[str, List[str]] = None) -> None:
        if not self.corpus or not self.queries:
            self._load(splits=splits)

    def _load(self, splits: Union[str, List[str]] = None) -> None:
        """
        Load specific splits of a dataset, such as 'train', 'test', or 'validation'. It can accept either a single split as a string or a list of splits.

        Parameters:
            splits (Union[str, List[str]], optional): The dataset split or splits to load. Defaults to None, which will load train split or the split specified when constructing this Generic Dataset Loader object
        """
        if splits:
            if isinstance(splits, str):
                splits = [splits]
            self.splits = splits
        self._load_corpus()
        self._load_queries()
        if self.query_type == QueryType.TEXT_2_SQL:
            self.setup_text2sql_corpus()

    def setup_text2sql_corpus(self) -> None:
        '''
        For text-2-sql datasets, have to convert some columns of the tables from string to numbers. 
        self.alt_corpus will be used instead of self.corpus for passing data to other objects.
        '''
        if not self.corpus:
            raise ValueError("Corpus has not been loaded yet!")
        number_converted_corpus = {}
        for split_name, split in self.corpus.items():
            number_converted_corpus[split_name] = [interpret_numbers(entry, self.table_col_name) for entry in split]
        self.alt_corpus = number_converted_corpus

    @abstractmethod
    def _load_corpus(self) -> None:
        pass

    @abstractmethod
    def _load_queries(self) -> None:
        pass

    def _write_table_to_path(self, table_name: Path, split_path: Path, nested_array: List[List]) -> None:
        if format.lower() == "csv":
            if "csv" not in table_name.suffix:
                table_name = table_name / ".csv"
            table_path = split_path / table_name
            with open(table_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(nested_array)
        if format.lower() == "json":
            if "json" not in table_name.suffix:
                table_name = table_name / ".json"
            table_path = split_path / table_name
            # TODO: write JSON persistence logic
            pass

    def persist_corpus_to(
        self, format: str, path: str = None, splits: Union[str, List[str]] = None
    ) -> None:
        """
        Saves the tables in the corpus to a specified location and format.

        Parameters:
            format (str): The format in which to save the corpus (e.g., 'csv', 'json').
            path (str, optional): The file system path where the corpus should be saved. creates directory if the directory doesn't exist. if no path is given, defaults to self.data_directory. if self.data_directory is also None, throw ValueError.
            splits (Union[str, List[str]]): split names to persist. if non is provided, all splits will be persisted.

        Raises:
            Runtime error if the corpus has not been loaded, or if the splits specified doesn't exist in the data dicts.
        """
        if not self.corpus:
            raise RuntimeError("Corpus has not been loaded!")

        if not splits:
            splits = self.splits
        else:
            splits = self._check_all_split_names_exist(splits)
            if not isinstance(splits, list):
                raise ValueError(
                    f"split {splits} doesn't exist for the current dataset!"
                )

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

        for split in splits:
            split_path = path_to_write_to / split
            if not split_path.exists():
                split_path.mkdir(parents=True, exist_ok=True)
            if self.alt_corpus:
                cur_split_dataset = self.alt_corpus[split]
                for entry in cur_split_dataset:
                    table_name = Path(entry[self.table_id_col_name])
                    nested_array = entry[self.table_col_name]
                    self._write_table_to_path(table_name, split_path, nested_array)
            else:
                cur_split_dataset = self.corpus[split]
                for batch in cur_split_dataset.iter(1):
                    table_name = Path(batch[self.table_id_col_name][0])
                    nested_array = batch[self.table_col_name][0]
                    self._write_table_to_path(table_name, split_path, nested_array)

    def convert_corpus_table_to(
        self,
        output_format: str = "nested array",
        splits: Union[str, List[str]] = None,
        batch_size: int = 64,
    ) -> Iterable[Dict]:
        """
        convert the corpus table to a specific format in memory.

        Parameters:

            output_format (str): the output class name, can be nest_array, pandas, etc.
            splits (Union[str, List[str]]): split names to convert. if non is provided, all splits will be converted.
            batch_size (int): number of tables to be outputted at once

        Returns:
            a generator that contains a dictionary with the keys being the table_ids and the values being the corresponding table as an output_format object
        """

        if not self.corpus:
            raise RuntimeError("Corpus has not been loaded!")

        if not splits:
            splits = self.splits
        else:
            splits = self._check_all_split_names_exist(splits)
            if not isinstance(splits, list):
                raise ValueError(
                    f"split {splits} doesn't exist for the current dataset!"
                )

        output_format = output_format.lower()
        for split in splits:
            cur_split_dataset = self.corpus[split]
            if self.alt_corpus:
                for i in range(0, len(cur_split_dataset), batch_size):
                    cur_batch = cur_split_dataset[i : i + batch_size]
                    res_dict = {}
                    for entry in cur_batch:
                        if "array" in output_format:
                            res_dict[entry[self.table_id_col_name]] = entry[self.table_col_name]
                        elif "dataframe" in output_format:
                            res_dict[entry[self.table_id_col_name]] = array_of_arrays_to_df(entry[self.table_col_name])
                    yield res_dict
            
            else:
                for batch in cur_split_dataset.iter(batch_size):
                    table_ids = batch[self.table_id_col_name]
                    tables = []
                    if "array" in output_format:
                        tables = batch[self.table_col_name]
                    elif "dataframe" in output_format:
                        tables = map(array_of_arrays_to_df, batch[self.table_col_name])
                    res_dict = {}
                    for key, value in zip(table_ids, tables):
                        res_dict[key] = value
                    yield res_dict

    def get_table_id_to_table(
        self,
        splits: Union[str, List[str]] = None,
    ) -> Dict[str, List[List]]:
        mapping_dict = {}
        for batch in self.convert_corpus_table_to(splits=splits):
            for table_id, table in batch.items():
                mapping_dict[table_id] = table
        return mapping_dict
    
    def get_table_id_to_database_id(
        self,
        splits: Union[str, List[str]] = None,
    ) -> Dict[str, Dict[str, str]]:
        mapping_dict = {}
        if isinstance(splits, str):
            splits = [splits]
        elif splits == None:
            splits = self.splits
        for split in splits:
            dict_for_cur_split = {}
            for entry in self.corpus[split]:
                dict_for_cur_split[entry[self.table_id_col_name]] = entry[self.database_id_col_name]
            mapping_dict[split] = dict_for_cur_split
        return mapping_dict

    def get_queries_for_task(
        self, splits: Union[str, List[str]] = None, batch_size: int = 64
    ) -> Iterable[List[QueryForTasksDataModel]]:
        if not self.queries:
            raise RuntimeError("Queries has not been loaded!")

        if not splits:
            splits = self.splits
        else:
            splits = self._check_all_split_names_exist(splits)
            if not isinstance(splits, List):
                raise ValueError(
                    f"split {splits} doesn't exist for the current dataset!"
                )
        for split in splits:
            cur_queries_split = self.queries[split]
            for batch in cur_queries_split.iter(batch_size):
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

    def get_corpus(self, splits: Union[str, List[str]] = None) -> DatasetDict:
        """
        get the corpus of the loaded dataset. if the dataset has not been loaded, raise an error.

        Parameters:
            splits(Union[str, List[str]], optional): optional, either a string or a list of strings, each string is a split name. if none is provided, the entire DatasetDict object is returned.

        Returns:
            a DatasetDict object containing the corresponding splits from the dataset's corpus

        Raises:
            a runtime error if a specified split doesn't exist within the existing_dataset_dict.

        """
        if not self.corpus:
            raise RuntimeError("Corpus datasets have not been loaded!")
        return self._get_dataset_dict_from_split(self.corpus, splits)

    def get_queries(self, splits: Union[str, List[str]] = None) -> DatasetDict:
        """
        get the queries of the loaded dataset. if the dataset has not been loaded, raise an error.

        Parameters:
            splits(Union[str, List[str]], optional): optional, either a string or a list of strings, each string is a split name. if none is provided, the entire DatasetDict object is returned.

        Returns:
            a DatasetDict object containing the corresponding splits from the dataset's queries.

        Raises:
            a runtime error if a specified split doesn't exist within the existing_dataset_dict.
        """
        if not self.queries:
            raise RuntimeError("Queries datasets have not been loaded!")
        return self._get_dataset_dict_from_split(self.queries, splits)

    def _get_dataset_dict_from_split(
        self, existing_dataset_dict: DatasetDict, splits: Union[str, List[str]] = None
    ) -> DatasetDict:
        """
        get the dataset from specified splits

        Parameters:
            existing_dataset_dict (DatasetDict): a dataset dict to get splits from.
            splits (Union[str, List[str]], optional): split names, can be a single name or a list of names. if none is provided, the entire DatasetDict object is returned.

        Returns:
            a DatasetDict object containing the requested splits.

        Raises:
            a runtime error if a specified split doesn't exist within the existing_dataset_dict.
        """
        if not splits:
            return existing_dataset_dict

        splits = self._check_all_split_names_exist(splits)
        if not isinstance(splits, list):
            raise RuntimeError(f"split {splits} doesn't exist for the current dataset!")
        dataset_splits = [
            (split_name, existing_dataset_dict[split_name]) for split_name in splits
        ]
        return DatasetDict(dataset_splits)

    def _check_all_split_names_exist(
        self, splits: Union[str, List[str]] = None
    ) -> Union[str, List[str]]:
        """
        check if the splits string or list exists within the corpus and queries Datasets. if any of the splits doesn't exist within either, the function returns the split name that doesn't exist. otherwise returns a list of splits

        Parameters:
            splits (Union[str, List[str]]) : a single split or a list of splits to check

        Returns:
             if a split doesn't exist, return the single string (split name that doesn't exist), or returns a list of validated split names.
        """
        if isinstance(splits, str):
            splits = [splits]
        for split_name in splits:
            if split_name not in self.corpus or split_name not in self.queries:
                return split_name
        return splits

    def get_corpus_header(self) -> Dict[str, List[str]]:
        """
        returns the header of this dataset's corpus

        Returns:
            a dictionary containing the headers of the corpus
        """
        if not self.corpus:
            raise RuntimeError("Corpus datasets have not been loaded!")
        return self.corpus.column_names

    def get_queries_header(self) -> Dict[str, List[str]]:
        """
        returns the header of this dataset's queries
        Returns:
            a dictionary containing the headers of the queries
        """
        if not self.queries:
            raise RuntimeError("Queries datasets have not been loaded!")
        return self.queries.column_names


