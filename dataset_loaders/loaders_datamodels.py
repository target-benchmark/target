from pydantic import BaseModel

class QueryDataForTasks(BaseModel):
    table_id: str
    database_id: int
    query: str
    query_id: int
    answer: str