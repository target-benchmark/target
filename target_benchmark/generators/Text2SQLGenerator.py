from json import JSONDecodeError
from typing import Dict

from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from target_benchmark.generators.DefaultGenerator import DefaultGenerator
from target_benchmark.generators.GeneratorPrompts import (
    TEXT2SQL_SYSTEM_PROMPT,
    TEXT2SQL_USER_PROMPT_NO_FORMAT_INSTR,
)


class Text2SQLResponse(BaseModel):
    """Information regarding a structured table."""

    chain_of_thought_reasoning: str = Field(..., description="Your thought process on how you arrived at the final SQL query.")
    sql_query: str = Field(..., description="the sql query you write.")
    database_id: str = Field(..., description="the database id of the database you chose to query from.")


class Text2SQLGenerator(DefaultGenerator):
    def __init__(
        self,
        system_message: str = TEXT2SQL_SYSTEM_PROMPT,
        user_message: str = TEXT2SQL_USER_PROMPT_NO_FORMAT_INSTR,
    ):
        super().__init__(system_message=system_message, user_message=user_message)
        self.user_message = user_message
        # response_schemas = [
        #     ResponseSchema(
        #         name="chain_of_thought_reasoning",
        #         description="Your thought process on how you arrived at the final SQL query.",
        #     ),
        #     ResponseSchema(name="sql_query", description="the sql query you write."),
        #     ResponseSchema(
        #         name="database_id",
        #         description="the database id of the database you chose to query from.",
        #     ),
        # ]

        # self.output_parser = StructuredOutputParser.from_response_schemas(
        #     response_schemas
        # )

        # self.chat_template = ChatPromptTemplate(
        #     messages=[
        #         SystemMessage(content=(system_message)),
        #         HumanMessagePromptTemplate.from_template(user_message),
        #     ],
        #     input_variables=["table_str", "query_str"],
        #     partial_variables={
        #         "format_instructions": self.output_parser.get_format_instructions()
        #     },
        # )
        # self.chain = self.chat_template | self.language_model | self.output_parser

    @retry(
        reraise=True,
        retry=retry_if_exception_type(exception_types=JSONDecodeError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=32),
    )
    def generate(self, table_str: str, query: str) -> Dict:
        # Note: currently text 2 sql generator takes in a database schema string
        # in the `table_str` input. In order to modify this, you can change the
        # `_get_downstream_task_results` method of the text 2 sql task. More
        # details can be found in the docs.

        # return self._invoke_chain(table_str, query)
        client = OpenAI()
        table_info_completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {
                    "role": "user",
                    "content": self.user_message.format(table_str=table_str, query_str=query),
                },
            ],
            response_format=Text2SQLResponse,
        )
        message = table_info_completion.choices[0].message
        if not message.parsed:
            raise JSONDecodeError

        return message.parsed.model_dump()
