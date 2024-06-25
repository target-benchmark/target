import unittest
from tasks.Text2SQLTask import Text2SQLTask


class T2SDataloadersTest(unittest.TestCase):

    def test_text2sql_get_schema(self):
        text2sql_task = Text2SQLTask()

        expected_schema = """Table Name: perpetrator
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
        self.assertEqual(
            expected_schema, text2sql_task._get_schema("spider", "perpetrator")
        )

    def test_text2sql_calculate_metric(self):
        self.maxDiff = None
        text2sql_task = Text2SQLTask()
        text2sql_task.pred_sql = [
            (
                "SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 2",
                "california_schools",
            ),
            (
                "SELECT T2.A2, T2.A3 FROM account AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id INNER JOIN loan AS T3 ON T1.account_id = T3.account_id WHERE T3.loan_id = 4990",
                "financial",
            ),
        ]
        text2sql_task.ref_sql = [
            (
                "SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1",
                "california_schools",
            ),
            (
                "SELECT T2.A2, T2.A3 FROM account AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id INNER JOIN loan AS T3 ON T1.account_id = T3.account_id WHERE T3.loan_id = 4990",
                "financial",
            ),
        ]
        text2sql_task.difficulties = ["easy", "hard"]
        text2sql_task.current_dataset = "bird"
        self.assertDictEqual(
            text2sql_task._calculate_downstream_task_performance().model_dump(),
            {
                "task_name": "Text to SQL Task",
                "scores": {
                    "easy": {"accuracy": 0, "num_sqls": 1},
                    "hard": {"accuracy": 1.0, "num_sqls": 1},
                    "all": {"accuracy": 0.5, "num_sqls": 2},
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
