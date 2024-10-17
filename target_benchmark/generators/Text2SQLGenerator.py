from typing import Dict

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from target_benchmark.generators.DefaultGenerator import DefaultGenerator
from target_benchmark.generators.GeneratorPrompts import (
    TEXT2SQL_SYSTEM_PROMPT,
    TEXT2SQL_USER_PROMPT,
)


class Text2SQLGenerator(DefaultGenerator):
    def __init__(
        self,
        system_message: str = TEXT2SQL_SYSTEM_PROMPT,
        user_message: str = TEXT2SQL_USER_PROMPT,
    ):
        super().__init__(system_message=system_message, user_message=user_message)

        response_schemas = [
            ResponseSchema(
                name="chain_of_thought_reasoning",
                description="Your thought process on how you arrived at the final SQL query.",
            ),
            ResponseSchema(name="sql_query", description="the sql query you write."),
            ResponseSchema(
                name="database_id",
                description="the database id of the database you chose to query from.",
            ),
        ]

        self.output_parser = StructuredOutputParser.from_response_schemas(
            response_schemas
        )

        self.chat_template = ChatPromptTemplate(
            messages=[
                SystemMessage(content=(system_message)),
                HumanMessagePromptTemplate.from_template(user_message),
            ],
            input_variables=["table_str", "query_str"],
            partial_variables={
                "format_instructions": self.output_parser.get_format_instructions()
            },
        )
        self.chain = self.chat_template | self.language_model | self.output_parser

    def generate(self, table_str: str, query: str) -> Dict:
        # Note: currently text 2 sql generator takes in a database schema string
        # in the `table_str` input. In order to modify this, you can change the
        # `_get_downstream_task_results` method of the text 2 sql task. More
        # details can be found in the docs.

        return self._invoke_chain(table_str, query)
