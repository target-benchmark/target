import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Literal

import pandas as pd

from target_benchmark.dataset_loaders.DatasetLoaderEnums import (
    InMemoryDataFormat,
    PersistenceDataFormat,
    QueryType,
)
from target_benchmark.dictionary_keys import (
    CONTEXT_COL_NAME,
    DATABASE_ID_COL_NAME,
    TABLE_COL_NAME,
    TABLE_ID_COL_NAME,
)


def set_query_type(string_rep: str) -> QueryType:
    string_rep = string_rep.lower()
    if string_rep in QueryType.FACT_VERIFICATION.value.lower():
        return QueryType.FACT_VERIFICATION
    elif string_rep in QueryType.TABLE_QA.value.lower():
        return QueryType.TABLE_QA
    elif string_rep in QueryType.TEXT_2_SQL.value.lower():
        return QueryType.TEXT_2_SQL
    elif string_rep in QueryType.NIH.value.lower():
        return QueryType.NIH
    else:
        return QueryType.OTHER


def get_random_tables(converted_corpus: Dict[str, Any], num_tables: int) -> Dict:
    indices = random.sample(range(len(converted_corpus[TABLE_COL_NAME])), num_tables)
    for col_name in [TABLE_COL_NAME, DATABASE_ID_COL_NAME, TABLE_ID_COL_NAME, CONTEXT_COL_NAME]:
        converted_corpus[col_name] = [converted_corpus[col_name][i] for i in indices]
    return converted_corpus


def set_persistence_data_format(string_rep: str) -> PersistenceDataFormat:
    cleaned = string_rep.lower().strip()
    if PersistenceDataFormat.JSON.value in cleaned:
        return PersistenceDataFormat.JSON
    elif PersistenceDataFormat.CSV.value in cleaned:
        return PersistenceDataFormat.CSV
    raise ValueError(
        f"the input formate {string_rep} did not match any available formats! try 'array', 'dataframe', or 'json'."
    )


def write_table_to_path(
    format: Literal["csv", "json"],
    table_name: Path,
    split_path: Path,
    nested_array: List[List],
) -> None:
    persistence_format = set_persistence_data_format(format)
    if persistence_format == PersistenceDataFormat.CSV:
        if "csv" not in table_name.suffix:
            table_name = table_name / ".csv"
        table_path = split_path / table_name
        with open(table_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(nested_array)
    elif persistence_format == PersistenceDataFormat.JSON:
        if "json" not in table_name.suffix:
            table_name = table_name / ".json"
        table_path = split_path / table_name
        headers = nested_array[0]
        rows = nested_array[1:]
        # Create a list of dictionaries, each representing a row
        dict_list = [dict(zip(headers, row)) for row in rows]

        with open(table_path, "w") as file:
            json.dump(dict_list, file, indent=4)  # 'indent=4' for pretty-printing


def set_in_memory_data_format(string_rep: str) -> InMemoryDataFormat:
    cleaned = string_rep.lower().strip()
    if InMemoryDataFormat.ARRAY.value in cleaned:
        return InMemoryDataFormat.ARRAY
    elif InMemoryDataFormat.DF.value in cleaned or "pandas" in cleaned:
        return InMemoryDataFormat.DF
    elif InMemoryDataFormat.DICTIONARY.value in cleaned:
        return InMemoryDataFormat.DICTIONARY
    raise ValueError(
        f"the input formate {string_rep} did not match any available formats! try 'array', 'dataframe', or 'json'."
    )


def enforce_split_literal(string_rep: str):
    splits = ("test", "train", "validation")
    if string_rep in splits:
        return string_rep
    raise ValueError(f"Split name {string_rep} is not a valid split name! Please use one of test, train, or validation")


def str_representation_to_pandas_df(array_repr: str) -> pd.DataFrame:
    """
    Attempts to convert a string representation of an array into an array object

    Parameters:
        array_repr (str): the string representation of the array

    Returns:
        an pandas DF, converted from the string

    Raises:
        ValueError if the representation doesn't convert to an array correctly
    """
    array = str_representation_to_array(array_repr)
    return array_of_arrays_to_df(array)


def array_of_arrays_to_df(array: List[List]) -> pd.DataFrame:
    return pd.DataFrame(data=array[1:], columns=array[0])


def array_of_arrays_to_dict(array: List[List]) -> Dict:
    headers = array[0]
    rows = array[1:]
    # Create a list of dictionaries, each representing a row
    return [dict(zip(headers, row)) for row in rows]


def convert_corpus_entry_to_df(col_name: str, entry: Dict) -> Dict:
    entry[col_name] = array_of_arrays_to_df(entry[col_name])
    return entry


def convert_corpus_entry_to_dict(col_name: str, entry: Dict) -> Dict:
    entry[col_name] = array_of_arrays_to_dict(entry[col_name])
    return entry


def str_representation_to_array(array_repr: str) -> List:
    """
    Attempts to convert a string representation of an array into an array object

    Parameters:
        array_repr (str): the string representation of the array

    Returns:
        an array, converted from the string

    Raises:
        ValueError if the representation doesn't convert to an array correctly
    """
    # array = literal_eval(array_repr)
    # if not isinstance(array, list) or not isinstance(array[0], list):
    #     raise ValueError("the input array is not a valid representation of a list!")
    array = json.loads(array_repr.replace("'", '"'))
    expected_len = len(array[0])
    for row in array:
        if len(row) != expected_len:
            raise ValueError(f"row {row} has unmatched number of items!")
    return array


def convert_nested_list_to(
    nested_list: List[List],
    output_format: Literal["array", "nested array", "pandas", "dataframe"],
):
    output_format = output_format.lower()
    if "array" in output_format:
        return nested_list
    elif "pandas" in output_format or "dataframe" in output_format:
        return array_of_arrays_to_df(nested_list)


def get_dummy_table_of_format(expected_format: Literal["array", "nested array", "pandas", "dataframe"] = "nested array"):
    dummy_table = [["header"], ["content"]]
    expected_format = expected_format.lower()
    if "array" in expected_format:
        return dummy_table
    elif "dataframe" in expected_format or "pandas" in expected_format:
        return array_of_arrays_to_df(dummy_table)
    else:
        return dummy_table
