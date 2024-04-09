'''
We use a lot of dictionaries for data. Subsequently, we are working with a set of dictionary key names. This file should contain all of them. When declaring dictionaries, please import this file and use the variable names in here.
'''


HF_DATASET_CONFIG_CORPUS_FIELD: str = 'hf_corpus_path'
HF_DATASET_CONFIG_QUERIES_FIELD: str = 'hf_queries_path'
GENERIC_DATASET_CONFIG_FIELD: str = 'dataset_path'

''' 
Default names for headers in queries & corpus datasets. These default names will be used by Target when referring to the corresponding columns.
'''
TABLE_COL_NAME: str = "table"
TABLE_ID_COL_NAME: str = "table_id"
DATABASE_ID_COL_NAME: str = "database_id"
QUERY_COL_NAME: str = "query"
QUERY_ID_COL_NAME: str = "query_id"
ANSWER_COL_NAME: str = "answer"