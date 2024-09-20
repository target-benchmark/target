import json
from typing import List, Union

import hnswlib
import numpy as np


def construct_embedding_index(table_embeddings: List[List]):
    # Constructing index
    corpus_index = hnswlib.Index(
        space="cosine", dim=len(table_embeddings[0])
    )  # possible options are l2, cosine or ip

    # Initializing index - the maximum number of elements should be known beforehand
    corpus_index.init_index(
        max_elements=len(table_embeddings), ef_construction=200, M=16
    )

    # Element insertion (can be called several times):
    corpus_index.add_items(
        np.asarray(table_embeddings),
        list(range(0, len(table_embeddings))),
    )

    # Controlling the recall by setting ef:
    corpus_index.set_ef(50)  # ef should always be > k

    return corpus_index


def json_table_str(table_array: List[List], num_rows: Union[int, None] = None):
    # the first row of the array is the header
    headers = table_array[0]
    # The rest of the array are the data rows
    data_rows = table_array[1:]
    if num_rows:
        data_rows = table_array[1 : 1 + num_rows]

    table_dict = {}
    for i, row in enumerate(data_rows):
        # Builds dict with header as keys and row as values
        # {col1: value1, col2: value2}
        table_dict[i] = dict(zip(headers, row))

    # return string representation of json table
    return json.dumps(table_dict)


def markdown_table_str(
    table_array: List[List], num_rows: Union[int, None] = None
) -> str:
    # TODO: evaluate different formatting: markdown vs json
    if not table_array:
        return table_array

    # the first row of the array is the header
    headers = table_array[0]
    # The rest of the array are the data rows
    data_rows = []
    if num_rows is not None:
        data_rows = table_array[1 : 1 + num_rows]
    else:
        data_rows = table_array[1:]

    # Start building the Markdown table
    markdown = "| " + " | ".join(str(header) for header in headers) + " |\n"

    if num_rows and num_rows > 0:
        # Add separator
        markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    # Add data rows
    for row in data_rows:
        markdown += "| " + " | ".join(str(item) for item in row) + " |\n"
    return markdown
