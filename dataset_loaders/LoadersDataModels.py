from pydantic import BaseModel

class QueryForTasksDataModel(BaseModel):
    table_id: str
    database_id: int
    query_id: int
    query: str
    answer: str