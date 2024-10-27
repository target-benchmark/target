from typing import Dict

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from target_benchmark.generators.AbsGenerator import AbsGenerator
from target_benchmark.generators.GeneratorPrompts import (
    DEFAULT_LM,
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
        self.language_model = ChatOpenAI(model=DEFAULT_LM, temperature=0.0)
        self.chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=(system_message)),
                HumanMessagePromptTemplate.from_template(user_message),
            ]
        )
        self.chain = self.chat_template | self.language_model

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=32),
    )
    def _invoke_chain(self, table_str: str, query: str):
        return self.chain.invoke({"table_str": table_str, "query_str": query})

    def generate(self, table_str: str, query: str) -> Dict:
        return {"content": self._invoke_chain(table_str, query).content}
