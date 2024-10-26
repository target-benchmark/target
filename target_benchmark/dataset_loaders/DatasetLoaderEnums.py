from enum import Enum


class QueryType(Enum):
    TEXT_2_SQL = "Text to SQL"
    FACT_VERIFICATION = "Fact Verification"
    TABLE_QA = "Table Question Answering"
    NIH = "Needle in Haystack"
    OTHER = "Other"


class PersistenceDataFormat(Enum):
    JSON = "json"
    CSV = "csv"


class InMemoryDataFormat(Enum):
    ARRAY = "array"
    DF = "dataframe"
    DICTIONARY = "dictionary"
