DEFAULT_QA_SYSTEM_MESSAGE = "You are a data analyst who reads tables to answer questions."
DEFAULT_FACT_VER_SYSTEM_MESSAGE = "Given the following evidence which may take the form of sentences or a data table, determine whether the evidence supports or refutes the following statement, or does not contain enough information. Assign the statement one of three labels: True, False, Not Enough Information. Do not include anything else in your answer."

DEFAULT_QA_USER_PROMPT = "Please use the following table(s) to answer the query. Tables: {table_str}\nQuery: {query_str}"