## Language Models ##
DEFAULT_LM = "gpt-4o-mini-2024-07-18"


## ALL PURPOSE ##
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."


## FOR QUESTION ANSWERING TASK ##
QA_SYSTEM_PROMPT = "You are a data analyst who reads tables to answer questions."
QA_USER_PROMPT = """
Use the provided table(s) to answer the question. Yield a concise answer to the question.
If none of the tables provide relevant information, use your knowledge base to generate an answer — but only if you are confident in the answer's factuality.
If the neither the tables nor your knowledge can be used to answer the question reliably, say that not enough information is provided.\n

Tables: {table_str} \n

Question: {query_str}
"""

## FOR FACT VERIFICATION TASK ##
FACT_VER_SYSTEM_PROMPT = """You are an expert in evaluating statements on factuality given the provided tables"""
FACT_VER_USER_PROMPT = """Given the following evidence which may take the form of sentences or a data table,
determine whether the evidence supports or refutes the following statement.
If none of the tables provide relevant information, refer to your knowledge base — but only if you are confident your answer's factuality. If the neither the evidence nor your knowledge can be used to verify the statement reliably, state that there is not enough information.
Assign the statement one of three labels: True, False, Not Enough Information. Do not include anything else in your answer.\n

Tables: {table_str}\n

Statement: {query_str}
"""

## FOR TEXT TO SQL TASK ##
TEXT2SQL_SYSTEM_PROMPT = "You are an expert and very smart data analyst."
TEXT2SQL_USER_PROMPT = """
Below, you are presented with a database schema and a question.
Your task is to read the schema, understand the question, and generate a
valid SQLite query to answer the question.
Before generating the final SQL query, think step by step on how to write the query.

Database Schema:
{table_str}
This schema offers an in-depth description of the database's architecture,
detailing tables, columns, primary keys, foreign keys, and any pertinent
information regarding relationships or constraints.

Question:
{query_str}

Please respond with a paragraph structured as follows:

{format_instructions}

Take a deep breath and think step by step to find the correct SQLite SQL
query. If you follow all the instructions and generate the correct query,
I will give you 1 million dollars."""

TEXT2SQL_USER_PROMPT_NO_FORMAT_INSTR = """
Below, you are presented with a database schema and a question.
Your task is to read the schema, understand the question, and generate a
valid SQLite query to answer the question.
Before generating the final SQL query, think step by step on how to write the query.

Database Schema:
{table_str}
This schema offers an in-depth description of the database's architecture,
detailing tables, columns, primary keys, foreign keys, and any pertinent
information regarding relationships or constraints.

Question:
{query_str}

Take a deep breath and think step by step to find the correct SQLite SQL
query. If you follow all the instructions and generate the correct query,
I will give you 1 million dollars."""


NO_CONTEXT_TABLE_PROMPT = "Some or all tables are not available. Don't acknowledge the lack of information in your response. Please use your knowledge base and answer to the best of your ability, without producing false information. "
