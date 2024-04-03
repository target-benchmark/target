from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
import csv
from dataset_loaders.utils import array_of_arrays_to_df, str_representation_to_array, str_representation_to_pandas_df
from typing import Iterable, Iterator

class AbsTargetDatasetLoader(ABC):
    '''
    The abstrack super class of target dataset loaders.
    This class contains implementations of utility functions shared by subclasses,
    but the loading functions are left as abstract so the subclasses will need to implement them separately. 
    '''
    def __init__(self,
                 dataset_name: str,
                 table_col_name: str = "table",
                 table_id_col_name: str = "table_id",
                 database_id_col_name: str = "database_id",
                 query_col_name: str = "query",
                 query_id_col_name: str = "query_id",
                 answer_col_name: str = "answer",
                 splits: str | list[str] = "test",
                 data_directory: str = None,
                 query_type: str = None,
                 **kwargs
                ):
        
        '''
        Constructor for the abstract data loader class.
        
        Parameters: 
            dataset_name (str): a string for the name of the dataset. required

            table_col_name (str, optional): name of the column that contains the nested array tables. defaults to "table", you can leave as default if working with any TARGET datasets. If user were to use custom datasets with different header names, this name can be adjusted accordingly.

            table_id_col_name (str, optional): name of the column that contains the unique identifiers of the tables. defaults to "table_id" across all TARGET datasets.
            
            database_id_col_name (str, optional): name of the column that contains the database id, this column exists in both queries and corpus datasets, and TARGET datasets loaders work on the assumption that the column name remains the same across the queries and corpus datasets. defaults to "database_id".

            query_col_name (str, optional): name of the column that contains the query strings. defaults to "query".

            query_id_col_name (str, optional): name of the column that contains the query ids. defaults to "query_id".
            
            answer_col_name (str, optional): name of the column that contains the answer str. defaults to "answer".

            splits (str | list[str], optional): splits of the data you want to load. defaults to test, since some models may use the train split of existing datasets for training, we opt to use test for our evaluation purposes.

            data_directory (str, optional): a directory where data files are stored. you don't have to provide one if you don't need to persist the file after loading it.
            
            query_type (str, optional): the type of queries that the dataset focuses on, for example fact verification, table QA, text to sql, etc. defaults to None.

        Instance Variables:
            aside from instance variables of the same name & purposes as the parameters, there's also:

            self.corpus: a huggingface DatasetDict object containing the corpus dataset, remains None until load corpus is complete. 
            self.queries: a huggingface DatasetDict object containing the queries dataset, remains None until load queries is complete. 
        '''

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
        self.query_type: str = query_type
        self.corpus: DatasetDict = None
        self.queries: DatasetDict = None

    @abstractmethod
    def load(self, split: str | list[str] = None) -> None:
        '''
        Load the dataset split.

        This method should be implemented by subclasses to load specific splits of a dataset, 
        such as 'train', 'test', or 'validation'. It can accept either a single split as a string or a list of splits.

        Parameters:
            split(str | list[str], optional): The dataset split or splits to load. If none are provided, splits specified in self.splits should be loaded. self.splits will also be updated to reflect the splits actually loaded

        Raises NotImplementedError: If the method is called on the abstract class directly.

        '''
        raise NotImplementedError("Base class do not use!")

    @abstractmethod
    def _load_corpus(self) -> None:
        pass
    @abstractmethod
    def _load_queries(self) -> None:
        pass



    def persist_corpus_to(self, format: str, path: str = None, splits: str | list[str] = None) -> None:
        """
        Saves the tables in the corpus to a specified location and format.

        Parameters:
            format (str): The format in which to save the corpus (e.g., 'csv', 'json').
            path (str, optional): The file system path where the corpus should be saved. creates directory if the directory doesn't exist. if no path is given, defaults to self.data_directory. if self.data_directory is also None, throw ValueError.
            splits (str | list[str]): split names to persist. if non is provided, all splits will be persisted.

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
                raise ValueError(f"split {splits} doesn't exist for the current dataset!")

        if not path:
            if not self.data_directory:
                raise ValueError(f"No path for persistence is specified!")
            path = self.data_directory

        path_to_write_to = Path(path)
        if path_to_write_to.suffix:
            raise ValueError(f"this path {path_to_write_to} looks like a path to a file.")
        if not path_to_write_to.exists():
            path_to_write_to.mkdir(parents=True, exist_ok=True)
        
        for split in splits:
            split_path = path_to_write_to / split
            if not split_path.exists():
                split_path.mkdir(parents=True, exist_ok=True)
            cur_split_dataset = self.corpus[split]
            for batch in cur_split_dataset.iter(1):
                table_name = Path(batch[self.table_id_col_name][0])
                nested_array = batch[self.table_col_name][0]
                if (format.lower() == "csv"):
                    if "csv" not in table_name.suffix:
                        table_name = table_name / ".csv"
                    table_path = split_path / table_name
                    with open(table_path, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows(nested_array)
                if (format.lower() == "json"):
                    if "json" not in table_name.suffix:
                        table_name = table_name / ".json"
                    table_path = split_path / table_name
                    # TODO: write JSON persistence logic
                    pass
                

    
    def convert_corpus_table_to(self, output_class: str = "nested array", splits: str | list[str] = None, batch_size: int = 64) -> Iterable[dict]:
        '''
        convert the corpus table to a specific format in memory. 
        
        Parameters:
        
            output_class (str): the output class name, can be nest_array, pandas, etc.
            splits (str | list[str]): split names to convert. if non is provided, all splits will be converted.
            batch_size (int): number of tables to be outputted at once

        Returns:
            a generator that contains a dictionary with the keys being the table_ids and the values being the corresponding table as an output_class object
        '''

        if not self.corpus:
            raise RuntimeError("Corpus has not been loaded!")
        
        if not splits:
            splits = self.splits
        else:
            splits = self._check_all_split_names_exist(splits)
            if not isinstance(splits, list):
                raise ValueError(f"split {splits} doesn't exist for the current dataset!")

        for split in splits:
            cur_split_dataset = self.corpus[split]
            for batch in cur_split_dataset.iter(batch_size):
                table_ids = batch[self.table_id_col_name]
                tables = [] 
                if "array" in output_class.lower():
                    tables = batch[self.table_col_name]
                elif "dataframe" in output_class.lower():
                    tables = map(array_of_arrays_to_df, batch[self.table_col_name])
                res_dict = {}
                for key, value in zip(table_ids, tables):
                    res_dict[key] = value
                yield res_dict


    def get_queries_for_task(self, splits: str | list[str] = None, batch_size: int =64) -> Iterable[list[dict[str, str]]]:
        if not self.queries:
            raise RuntimeError("Queries has not been loaded!")
        
        if not splits:
            splits = self.splits
        else:
            splits = self._check_all_split_names_exist(splits)
            if not isinstance(splits, list):
                raise ValueError(f"split {splits} doesn't exist for the current dataset!")
        for split in splits:
            cur_queries_split = self.queries[split]
            for batch in cur_queries_split.iter(batch_size):
                res_list = []
                query_ids = batch[self.query_id_col_name]
                queries = batch[self.query_col_name]
                database_ids = batch[self.database_id_col_name]
                table_ids = batch[self.table_id_col_name]
                answers = batch[self.answer_col_name]
                for query_id, query, database_id, table_id, answer in zip(query_ids, queries, database_ids, table_ids, answers):
                    res_list.append({
                        'query_id': query_id,
                        'query': query,
                        'database_id': database_id,
                        'table_id': table_id,
                        'answer': answer
                    })
                yield res_list

    def get_dataset_name(self) -> str:
        '''
        returns the name of the dataset

        Returns:
            a string which is the name of the dataset
        '''
        return self.dataset_name

    def get_corpus(self, splits: str | list[str] = None) -> DatasetDict:
        '''
        get the corpus of the loaded dataset. if the dataset has not been loaded, raise an error.

        Parameters:
            splits(str | list[str], optional): optional, either a string or a list of strings, each string is a split name. if none is provided, the entire DatasetDict object is returned.

        Returns: 
            a DatasetDict object containing the corresponding splits from the dataset's corpus

        Raises:
            a runtime error if a specified split doesn't exist within the existing_dataset_dict.

        '''
        if not self.corpus:
            raise RuntimeError("Corpus datasets have not been loaded!")
        return self._get_dataset_dict_from_split(self.corpus, splits)
    
    def get_queries(self, splits: str | list[str] = None) -> DatasetDict:
        '''
        get the queries of the loaded dataset. if the dataset has not been loaded, raise an error.
        
        Parameters:
            splits(str | list[str], optional): optional, either a string or a list of strings, each string is a split name. if none is provided, the entire DatasetDict object is returned.
        
        Returns: 
            a DatasetDict object containing the corresponding splits from the dataset's queries.
        
        Raises:
            a runtime error if a specified split doesn't exist within the existing_dataset_dict.
        '''
        if not self.queries:
            raise RuntimeError("Queries datasets have not been loaded!")
        return self._get_dataset_dict_from_split(self.queries, splits)
    
    def _get_dataset_dict_from_split(self, existing_dataset_dict: DatasetDict, splits: str | list[str] = None) -> DatasetDict:
        '''
        get the dataset from specified splits

        Parameters:
            existing_dataset_dict (DatasetDict): a dataset dict to get splits from.
            splits (str | list[str], optional): split names, can be a single name or a list of names. if none is provided, the entire DatasetDict object is returned.
        
        Returns:
            a DatasetDict object containing the requested splits.
        
        Raises:
            a runtime error if a specified split doesn't exist within the existing_dataset_dict.
        '''
        if not splits:
            return existing_dataset_dict

        splits = self._check_all_split_names_exist(splits)
        if not isinstance(splits, list):
            raise RuntimeError(f"split {splits} doesn't exist for the current dataset!")
        dataset_splits = [(split_name, existing_dataset_dict[split_name]) for split_name in splits]
        return DatasetDict(dataset_splits)

    def _check_all_split_names_exist(self, splits: str | list[str] = None) -> list[str] | str:
        '''
        check if the splits string or list exists within the corpus and queries Datasets. if any of the splits doesn't exist within either, the function returns the split name that doesn't exist. otherwise returns a list of splits

        Parameters:
            splits (str | list[str]) : a single split or a list of splits to check
        
        Returns:
             if a split doesn't exist, return the single string (split name that doesn't exist), or returns a list of validated split names.
        '''
        if isinstance(splits, str):
            splits = [splits]
        for split_name in splits:
            if split_name not in self.corpus or split_name not in self.queries:
                return split_name
        return splits
    
    def get_corpus_header(self) -> dict[str, list[str]]:
        '''
        returns the header of this dataset's corpus

        Returns:
            a dictionary containing the headers of the corpus
        '''
        if not self.corpus:
            raise RuntimeError("Corpus datasets have not been loaded!")
        return self.corpus.column_names
    
    def get_queries_header(self) -> dict[str, list[str]]:
        '''
        returns the header of this dataset's queries
        Returns:
            a dictionary containing the headers of the queries
        '''
        if not self.queries:
            raise RuntimeError("Queries datasets have not been loaded!")
        return self.queries.column_names



class QueryType(Enum):
    TEXT_2_SQL = "Text-2-Sql"
    FACT_VERIFICATION = "Fact Verification"
    TABLE_QA = "Table question answering"
    OTHER = "Other"