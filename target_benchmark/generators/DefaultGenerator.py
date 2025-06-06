from typing import Dict

from openai import RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from target_benchmark.dictionary_keys import CONTENT_KEY_NAME
from target_benchmark.generators.AbsGenerator import AbsGenerator
from target_benchmark.generators.GeneratorPrompts import (
    DEFAULT_LM,
    DEFAULT_SYSTEM_PROMPT,
    QA_USER_PROMPT,
)


class DefaultGenerator(AbsGenerator):
    def __init__(
        self, system_message: str = DEFAULT_SYSTEM_PROMPT, user_message: str = QA_USER_PROMPT, lm_model_name: str = DEFAULT_LM
    ):
        from langchain_core.messages import SystemMessage
        from langchain_core.prompts import (
            ChatPromptTemplate,
            HumanMessagePromptTemplate,
        )
        from langchain_openai import ChatOpenAI

        super().__init__()
        self.language_model = ChatOpenAI(model=lm_model_name, temperature=0.0)
        self.chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=(system_message)),
                HumanMessagePromptTemplate.from_template(user_message),
            ]
        )
        self.chain = self.chat_template | self.language_model

    @retry(
        reraise=True,
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(10),
        wait=wait_fixed(5),
    )
    def _invoke_chain(self, table_str: str, query: str):
        return self.chain.invoke({"table_str": table_str, "query_str": query})

    def generate(self, table_str: str, query: str) -> Dict:
        return {CONTENT_KEY_NAME: self._invoke_chain(table_str, query).content}
