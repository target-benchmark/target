import unittest
from tasks.Text2SQLTask import Text2SQLTask


class T2SDataloadersTest(unittest.TestCase):
    def test_text2sql(self):
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
)"""
        print(text2sql_task._get_schema("spider", "perpetrator"))
        print(text2sql_task._get_schema("bird", "california_schools"))


if __name__ == "__main__":
    unittest.main()
