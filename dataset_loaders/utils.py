from ast import literal_eval
import pandas as pd
import json


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


def array_of_arrays_to_df(array: list[list]):
    return pd.DataFrame(data=array[1:], columns=array[0])


def str_representation_to_array(array_repr: str) -> list:
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
