from typing import List, Tuple, Union

from pydantic import BaseModel


class DownstreamGeneratedResultDataModel(BaseModel):
    dataset_name: str
    query_id: Union[int, str]
    generated_results: Union[str, Tuple[str, str], List[str]]
