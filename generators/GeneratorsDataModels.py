from pydantic import BaseModel
from typing import Union

class DownstreamGeneratedResultDataModel(BaseModel):
    dataset_name: str
    query_id: Union[int, str]
    generated_results: str
