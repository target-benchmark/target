from llama_index.core.program import LLMTextCompletionProgram
from llama_index.legacy.bridge.pydantic import BaseModel, Field
from llama_index.llms.openai import OpenAI

from pathlib import Path
import pandas as pd


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

Do NOT make the table name one of the following: {exclude_table_name_list}

Table:
{table_str}

Summary: """

program = LLMTextCompletionProgram.from_defaults(
    output_cls=TableInfo,
    llm=OpenAI(model="gpt-3.5-turbo"),
    prompt_template_str=prompt_str,
)


def _get_table_info_with_index(table_info_dir: str, table_name: str) -> str:
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
    table_info_dir: str, df: pd.DataFrame, table_name: str
) -> TableInfo:
    table_info = _get_table_info_with_index(table_info_dir, table_name)
    if table_info:
        return table_info

    df_str = df.head(10).to_csv()
    table_info: TableInfo = program(
        table_str=df_str,
        exclude_table_name_list=str(list(existing_table_names)),
    )
    table_info.table_name = table_name  # forcefully overwrite table name
    return table_info
