from abc import ABC, abstractmethod


class AbsGenerator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, table_str: str, query: str) -> str:
        """
        Generate a response with the generator.

        Parameters:
            table_str (str): the string representation of the table/information related to the query
            query (str): the query string

        Returns:
            a string, the generated answer.
        """
        pass
