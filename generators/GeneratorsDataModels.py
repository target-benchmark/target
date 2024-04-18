from pydantic import BaseModel


class DownstreamGeneratedResultDataModel(BaseModel):
    dataset_name: str
    query_id: int
    generated_results: str
