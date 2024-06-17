from dictionary_keys import *
from typing import Dict, Literal, Optional, Union, List, Any
import pandas as pd
from pydantic import BaseModel, Field


class QueryForTasksDataModel(BaseModel):
    table_id: str
    database_id: int
    query_id: int
    query: str
    answer: str

class CorpusForRetrieversDataModel(BaseModel):
    database_id: Union[int, List[int]] = Field(description="a single or a list of database ids")
    table_id: Union[str, List[str]] = Field(description="a single or a list of table ids")
    table: Union[Union[List[List], pd.DataFrame, Dict], Union[List[List[List]], List[pd.DataFrame], List[Dict]]] = Field(
        description="A single or a list of tables. the tables can be expected to be in the retriever's specified `expected_corpus_format`."
    )
    context: Union[Dict, List[Dict]] = Field(description="a single or a list of context dictionaries.")

class DatasetConfigDataModel(BaseModel):
    """
    A base data model for dataset configs. DO NOT USE DIRECTLY. use `GenericDatasetConfigDataModel` or `HFDatasetConfigDataModel`.
    Target's dataset configs are written in `TargetDatasetConfig.py`.
    """

    dataset_name: str = Field(
        description="Name of the dataset. This must be filled out."
    )

    table_col_name: str = Field(
        default=TABLE_COL_NAME,
        description="Name of the column that contains the tables in nested array format.",
    )

    table_id_col_name: str = Field(
        default=TABLE_ID_COL_NAME,
        description="Name of the column that contains the table ids.",
    )

    database_id_col_name: str = Field(
        default=DATABASE_ID_COL_NAME,
        description="Name of the column that contains the dataset ids.",
    )

    context_col_name: str = Field(
        default=CONTEXT_COL_NAME,
        description="Name of the column that contains the context(metadata, for example foreign keys in a text2sql dataset).",        
    )

    query_id_col_name: str = Field(
        default=QUERY_ID_COL_NAME,
        description="Name of the column that contains the query ids.",
    )

    query_col_name: str = Field(
        default=QUERY_COL_NAME,
        description="Name of the column that contains the queries.",
    )

    answer_col_name: str = Field(
        default=ANSWER_COL_NAME,
        description="Name of the column that contains the downstream task answers.",
    )
    split: Literal["test", "train", "validation"] = Field(
        default="test",
        description="Split to include. Defaults to only the test split.",
    )
    data_directory: str = Field(
        default=None,
        description="directory for where to persist the data to. defaults to None.",
    )
    query_type: Optional[
        Literal["Fact Verification", "Text to SQL", "Table Question Answering", "Other"]
    ] = Field(default=None, description="Type of query in this dataset.")

    aux: Optional[Dict] = Field(
        default=None, description="Any additional information related to the dataset."
    )


class GenericDatasetConfigDataModel(DatasetConfigDataModel):
    """
    Data model for local datasets.
    """

    dataset_path: str = Field(description="Path to the local dataset directory.")
    datafile_ext: str = (
        Field(default=None, description="File type of the dataset. csv, tsv, etc."),
    )


class HFDatasetConfigDataModel(DatasetConfigDataModel):
    """
    Data model for huggingface datasets.
    """

    hf_corpus_dataset_path: str = Field(
        description="A huggingface dataset path to the query dataset. It will look something like target-benchmark/fetaqa-corpus (namespace/corpus-dataset-name)"
    )

    hf_queries_dataset_path: str = Field(
        description="A huggingface dataset path to the query dataset. It will look something like target-benchmark/fetaqa-queries (namespace/queries-dataset-name)"
    )
