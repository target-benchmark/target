import json
import re
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from sqlalchemy import Column, Engine, MetaData, String, Table, inspect
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


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
- You are allowed to output a name like <meaningful_table_name>_2, since some big tables are broken up into many smaller tables.

Existing Table Names:
{names_tried}

Table:
{table_str}

Summary: """


def get_table_info_with_index(
    table_info_dir: str, db_id: str, table_name: str
) -> Union[TableInfo, None]:
    results_gen = Path(table_info_dir).glob(f"{db_id}_{table_name}.json")
    results_list = list(results_gen)
    if len(results_list) == 0:
        return None
    elif len(results_list) == 1:
        path = results_list[0]
        return TableInfo.parse_file(path)
    else:
        raise ValueError(f"More than one file matching index: {list(results_gen)}")


class DuplicateTableNameError(Exception):
    pass


@retry(
    reraise=True,
    retry=retry_if_exception_type(exception_types=(json.JSONDecodeError, DuplicateTableNameError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=32),
)
def get_table_info_from_lm(df_str: str, names_tried: set, existing_names: Dict):
    client = OpenAI()
    table_info_completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {
                "role": "user",
                "content": prompt_str.format(table_str=df_str, names_tried=names_tried),
            },
        ],
        response_format=TableInfo,
    )
    message = table_info_completion.choices[0].message
    if not message.parsed:
        raise json.JSONDecodeError
    if message.parsed.table_name in existing_names:
        # duplicate table names, try again
        names_tried.add(message.parsed.table_name)
        raise DuplicateTableNameError(
            f"Table name {message.parsed.table_name} is a duplicate."
        )
    return message


def construct_table_info(
    table_info_dir: str,
    df: pd.DataFrame,
    database_id: str,
    table_name: str,
    existing_names: Dict,
) -> Union[TableInfo, None]:
    cleaned_table_name = re.sub(r"[^a-zA-Z0-9_]", "_", table_name)
    if isinstance(database_id, int):
        database_id = str(database_id)
    cleaned_db_id = re.sub(r"[^a-zA-Z0-9_]", "_", database_id)
    table_info = get_table_info_with_index(
        table_info_dir, database_id, cleaned_table_name
    )

    # check if
    # - table info has been constructed already
    # - and not appeared in other tables
    if table_info and table_info.table_name not in existing_names:
        return table_info

    df_str = df.head(10).to_csv()
    names_tried = set()

    message = get_table_info_from_lm(df_str, names_tried, existing_names)
    table_info = TableInfo(
        table_name=message.parsed.table_name,
        table_summary=message.parsed.table_summary,
    )
    out_file_path = Path(table_info_dir) / f"{cleaned_db_id}_{cleaned_table_name}.json"
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
) -> None:
    # Use inspector to check if the table already exists
    inspector = inspect(engine)
    if inspector.has_table(table_name):
        return  # Exit the function to avoid recreating the table

    # Sanitize column names
    sanitized_columns = {col: sanitize_column_name(col) for col in df.columns}
    df = df.rename(columns=sanitized_columns)

    # Dynamically create columns based on DataFrame columns and data types
    # columns = [
    #     Column(col, String if dtype == "object" else Integer)
    #     for col, dtype in zip(df.columns, df.dtypes)
    # ]
    columns = [Column("Column", String)]
    # Column(
    #     df.columns.values[0], String if df.dtypes.values[0] == "object" else Integer
    # )
    # ]

    # Create a table with the defined columns
    Table(table_name, metadata_obj, *columns)

    # Create the table in the database
    metadata_obj.create_all(engine)

    # Insert data from DataFrame into the table
    # with engine.connect() as conn:
    #     for _, row in df.iterrows():
    #         insert_stmt = table.insert().values(row.to_dict()[df.columns[0]])
    #         conn.execute(insert_stmt)
    #         break
    #     conn.commit()
