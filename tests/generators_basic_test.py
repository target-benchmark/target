import unittest
from generators.DefaultGenerator import DefaultGenerator


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


if __name__ == "__main__":
    unittest.main()
