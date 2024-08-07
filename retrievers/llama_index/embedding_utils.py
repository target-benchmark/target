import json
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.objects import SQLTableNodeMapping, SQLTableSchema
from llama_index.legacy.bridge.pydantic import BaseModel, Field
from llama_index.llms.openai import OpenAI
from pathlib import Path
import pandas as pd

import re
from sqlalchemy import (
    create_engine,
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

program = LLMTextCompletionProgram.from_defaults(
    output_cls=TableInfo,
    llm=OpenAI(model="gpt-3.5-turbo"),
    prompt_template_str=prompt_str,
)


def _get_table_info_with_index(
    table_info_dir: str, table_name: str
) -> TableInfo | None:
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
    table_info: TableInfo = program(
        table_str=df_str,
    )
    table_info.table_name = f"{database_id}:{table_name}:{table_info.table_name}"  # forcefully prepend the official table name
    out_file_path = f"{table_info_dir}/{database_id}_{table_name}.json"
    with open(out_file_path, "w") as file:
        json.dump(table_info.dict(), file)
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
