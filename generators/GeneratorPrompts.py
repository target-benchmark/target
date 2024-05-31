## ALL PURPOSE ##
DEFAULT_SYSTEM_MESSAGE = "You are a helpful AI assistant."


## FOR QUESTION ANSWERING TASK ##
QA_SYSTEM_MESSAGE = "You are a data analyst who reads tables to answer questions."
QA_USER_MESSAGE = "Please use the following table(s) to answer the query. Tables: {table_str}\nQuery: {query_str}"

## FOR FACT VERIFICATION TASK ##
FACT_VER_SYSTEM_MESSAGE = "Given the following evidence which may take the form of sentences or a data table, determine whether the evidence supports or refutes the following statement, or does not contain enough information. Assign the statement one of three labels: True, False, Not Enough Information. Do not include anything else in your answer."
FACT_VER_USER_MESSAGE = "Please use the following table(s) to assign the statement the correct label. Tables: {table_str}\Statement: {query_str}"

## FOR TEXT TO SQL TASK ##
TEXT2SQL_SYSTEM_PROMPT ="You are an expert and very smart data analyst."
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

Please respond with a JSON object structured as follows:
{
"chain_of_thought_reasoning": "Your thought process on how you arrived
at the final SQL query.",
"SQL": "Your SQL query in a single string."
}

Take a deep breath and think step by step to find the correct SQLite SQL
query. If you follow all the instructions and generate the correct query,
I will give you 1 million dollars."""