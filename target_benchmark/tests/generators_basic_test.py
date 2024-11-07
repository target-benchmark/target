import unittest
from target_benchmark.generators.DefaultGenerator import DefaultGenerator
from target_benchmark.generators.Text2SQLGenerator import Text2SQLGenerator


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.dg = DefaultGenerator()

    def test_chain_runs(self):
        table_str = """
| Employee ID | Name       | Department  | Salary |
|-------------|------------|-------------|--------|
| 1           | Alice Smith| Marketing   | $60,000|
| 2           | Bob Johnson| Sales       | $55,000|
| 3           | Carol Lee  | IT          | $75,000|
| 4           | Dave Brown | HR          | $50,000|
        """
        query = "What's the salary of Carol Lee?"
        answer = self.dg.generate(table_str=table_str, query=query)
        print(answer)

    def test_text2sql_generator(self):
        table_str = """Table Name: perpetrator
 Schema:
CREATE TABLE "perpetrator" (
"Perpetrator_ID" int,
"People_ID" int,
"Date" text,
"Year" real,
"Location" text,
"Country" text,
"Killed" int,
"Injured" int,
PRIMARY KEY ("Perpetrator_ID"),
FOREIGN KEY ("People_ID") REFERENCES "people"("People_ID")
)
Table Name: people
 Schema:
CREATE TABLE "people" (
"People_ID" int,
"Name" text,
"Height" real,
"Weight" real,
"Home Town" text,
PRIMARY KEY ("People_ID")
)
"""
        query = "what is the name of the most recently recorded perpetrator?"
        generator = Text2SQLGenerator()
        print(generator.generate(table_str, query))


if __name__ == "__main__":
    unittest.main()
