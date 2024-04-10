from pydantic import BaseModel

class RetrievalResultDataModel(BaseModel):
    dataset_name: str
    query_id: int
    retrieval_results: list[str]