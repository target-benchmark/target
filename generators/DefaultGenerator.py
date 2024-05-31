from generators.AbsGenerator import AbsGenerator
from generators.GeneratorPrompts import (
    DEFAULT_SYSTEM_MESSAGE,
    DEFAULT_QA_USER_MESSAGE,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI

import os


class DefaultGenerator(AbsGenerator):
    def __init__(self, system_message: str = DEFAULT_SYSTEM_MESSAGE, user_message: str = DEFAULT_QA_USER_MESSAGE):
        super().__init__()
        self.language_model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
        )
        self.chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=(system_message)),
                HumanMessagePromptTemplate.from_template(user_message),
            ]
        )
        self.chain = self.chat_template | self.language_model

    def generate(self, table_str: str, query: str) -> str:
        return self.chain.invoke({"table_str": table_str, "query_str": query}).content
