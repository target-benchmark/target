import json
from typing import Union
from pydantic import BaseModel, Field
from openai import OpenAI

from pathlib import Path
import pandas as pd

import re
from sqlalchemy import (
    Engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
)


class TableInfo(BaseModel):
    """Information regarding a structured table."""

    table_name: str = Field(
        ..., description="table name (must be underscores and NO spaces)"
    )
    table_summary: str = Field(
        ..., description="short, concise summary/caption of the table"
    )


prompt_str = """\
Give me a summary of the table with the following JSON format.

- The table name must be unique to the table and describe it while being concise.
- Do NOT output a generic table name (e.g. table, my_table).

Table:
{table_str}

Summary: """

client = OpenAI()


def _get_table_info_with_index(
    table_info_dir: str, table_name: str
) -> Union[TableInfo, None]:
    results_gen = Path(table_info_dir).glob(f"{table_name}_*")
    results_list = list(results_gen)
    if len(results_list) == 0:
        return None
    elif len(results_list) == 1:
        path = results_list[0]
        return TableInfo.parse_file(path)
    else:
        raise ValueError(f"More than one file matching index: {list(results_gen)}")


def construct_table_info(
    table_info_dir: str, df: pd.DataFrame, database_id: str, table_name: str
) -> TableInfo:
    table_info = _get_table_info_with_index(table_info_dir, table_name)
    if table_info:
        return table_info

    df_str = df.head(10).to_csv()
    table_info_completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt_str.format(table_str=df_str)},
        ],
        response_format=TableInfo,
    )
    message = table_info_completion.choices[0].message
    if not message.parsed:
        raise json.JSONDecodeError
    table_info = TableInfo(
        table_name=message.parsed.table_name,
        table_summary=message.parsed.table_summary,
    )

    table_info.table_name = f"{database_id}:{table_name}:{table_info.table_name}"  # forcefully prepend the official table name
    out_file_path = f"{table_info_dir}/{database_id}_{table_name}.json"
    with open(out_file_path, "w") as file:
        json.dump(table_info.model_dump(), file)
    return table_info


# Function to create a sanitized column name
def sanitize_column_name(col_name):
    # Remove special characters and replace spaces with underscores
    return re.sub(r"\W+", "_", col_name)


# Function to create a table from a DataFrame using SQLAlchemy
def create_table_from_dataframe(
    df: pd.DataFrame, table_name: str, engine: Engine, metadata_obj: MetaData
):
    # Sanitize column names
    sanitized_columns = {col: sanitize_column_name(col) for col in df.columns}
    df = df.rename(columns=sanitized_columns)

    # Dynamically create columns based on DataFrame columns and data types
    columns = [
        Column(col, String if dtype == "object" else Integer)
        for col, dtype in zip(df.columns, df.dtypes)
    ]

    # Create a table with the defined columns
    table = Table(table_name, metadata_obj, *columns)

    # Create the table in the database
    metadata_obj.create_all(engine)

    # Insert data from DataFrame into the table
    with engine.connect() as conn:
        for _, row in df.iterrows():
            insert_stmt = table.insert().values(**row.to_dict())
            conn.execute(insert_stmt)
        conn.commit()
