from pydantic import BaseModel, Field


class RetrievalResultDataModel(BaseModel):
    dataset_name: str
    query_id: int
    retrieval_results: list[str]
    retrieved_tables: list[str] = Field(
        default=[],
        description="a list of the string representation of the tables retrieved",
    )
