import json
from typing import Any, List, Literal, Dict
import pandas as pd


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


def array_of_arrays_to_df(array: List[List]):
    return pd.DataFrame(data=array[1:], columns=array[0])


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
    nested_list: List[List], output_format: Literal["array", "nested array", "pandas", "dataframe"]
):
    output_format = output_format.lower()
    if "array" in output_format:
        return nested_list
    elif "pandas" in output_format or "dataframe" in output_format:
        return array_of_arrays_to_df(nested_list)


def markdown_table_with_headers(nested_array: List[List]):
    # the first row of the array is the header
    headers = nested_array[0]
    # The rest of the array are the data rows
    data_rows = nested_array[1:]

    # Start building the Markdown table
    markdown = "| " + " | ".join(str(header) for header in headers) + " |\n"

    # Add separator
    markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    # Add data rows
    for row in data_rows:
        markdown += "| " + " | ".join(str(item) for item in row) + " |\n"
    return markdown


def get_dummy_table_of_format(expected_format: Literal["array", "nested array", "pandas", "dataframe"] = "nested array"):
    dummy_table = [["header"], ["content"]]
    expected_format = expected_format.lower()
    if "array" in expected_format:
        return dummy_table
    elif "dataframe" in expected_format or "pandas" in expected_format:
        return array_of_arrays_to_df(dummy_table)
    else:
        return dummy_table


def check_col(entry, col):
    cur_type = None
    types = []
    for row in entry["table"][1:]:
        cell = row[col]
        try:
            int(cell)
            types.append(int)
            continue
        except:
            pass
        try:
            float(cell)
            types.append(float)
            continue
        except:
            pass
        return None
    
    if all([x == types[0] for x in types]):
        return types[0]
    else:
        return None

def interpret_numbers(entry: Dict[str, Any], table_col_name: str) -> Dict[str, Any]:
    table_list = entry[table_col_name]
    if len(table_list) > 1:
        conv_indices = {}
        for i in range(len(table_list[1])):
            res = check_col(entry, i)
            if res:
                conv_indices[i] = res
            
        for row in table_list[1:]:
            for conv_idx, conv_type in conv_indices.items():
                try:
                    row[conv_idx] = conv_type(row[conv_idx])
                except:
                    pass
    
    return entry