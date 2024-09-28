from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI

from target_benchmark.generators.AbsGenerator import AbsGenerator
from target_benchmark.generators.GeneratorPrompts import (
    DEFAULT_SYSTEM_PROMPT,
    QA_USER_PROMPT,
)


class DefaultGenerator(AbsGenerator):
    def __init__(
        self,
        system_message: str = DEFAULT_SYSTEM_PROMPT,
        user_message: str = QA_USER_PROMPT,
    ):
        super().__init__()
        self.language_model = ChatOpenAI(
            model="gpt-4o-mini-2024-07-18",
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
