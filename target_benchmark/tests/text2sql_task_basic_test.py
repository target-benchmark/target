import unittest
from target_benchmark.retrievers.RetrieversDataModels import RetrievalResultDataModel
from target_benchmark.tasks.Text2SQLTask import Text2SQLTask
from target_benchmark.dataset_loaders import Text2SQLDatasetLoader
from target_benchmark.dataset_loaders.TargetDatasetConfig import DEFAULT_SPIDER_TEST_DATASET_CONFIG, DEFAULT_BIRD_VALIDATION_DATASET_CONFIG


class T2SDataloadersTest(unittest.TestCase):

    def test_with_dataset_loader(self):
        spider_loader = Text2SQLDatasetLoader(
            **DEFAULT_SPIDER_TEST_DATASET_CONFIG.model_dump()
        )
        spider_loader.load()
        table_id_to_table = spider_loader.get_table_id_to_table()
        text2sql_task = Text2SQLTask(metrics=["execution_accuracy", "execution_ves"])
        text2sql_task.setup_database_dirs({"spider": spider_loader})
        text2sql_task.current_dataset = "spider"
        #         expected_schema = """Table Name: perpetrator
        #  Schema:
        # CREATE TABLE "perpetrator" (
        # "Perpetrator_ID" int,
        # "People_ID" int,
        # "Date" text,
        # "Year" real,
        # "Location" text,
        # "Country" text,
        # "Killed" int,
        # "Injured" int,
        # PRIMARY KEY ("Perpetrator_ID"),
        # FOREIGN KEY ("People_ID") REFERENCES "people"("People_ID")
        # )
        # Table Name: people
        #  Schema:
        # CREATE TABLE "people" (
        # "People_ID" int,
        # "Name" text,
        # "Height" real,
        # "Weight" real,
        # "Home Town" text,
        # PRIMARY KEY ("People_ID")
        # )
        # """
        #         self.assertEqual(
        #             expected_schema, text2sql_task._get_schema("spider", "perpetrator")
        #         )
        query_batch = {
            "query_id": [0],
            "database_id": ["soccer_3"],
            "table_id": {"N/A"},
            "query": ["How many clubs are there?"],
            "answer": ["SELECT count(*) FROM club"],
            "difficulty": ["easy"],
        }
        retrieval_results = [
            RetrievalResultDataModel(
                dataset_name="spider",
                query_id=0,
                retrieval_results=[("soccer_3", "N/A")],
            )
        ]
        dataset_name = "spider"
        results = text2sql_task._get_downstream_task_results(
            query_batch, retrieval_results, dataset_name, table_id_to_table
        )
        res_dict = results[0].model_dump()

        self.assertEqual(res_dict["dataset_name"], "spider")
        self.assertEqual(res_dict["query_id"], 0)
        text2sql_task._update_downstream_task_metrics(query_batch, results)
        self.assertListEqual(
            [("SELECT count(*) FROM club", "soccer_3")], text2sql_task.ref_sql
        )
        self.assertListEqual(["easy"], text2sql_task.difficulties)
        print(text2sql_task._calculate_downstream_task_performance())

    def test_text2sql_calculate_metric(self):
        self.maxDiff = None
        bird_loader = Text2SQLDatasetLoader(**DEFAULT_BIRD_VALIDATION_DATASET_CONFIG.model_dump())
        bird_loader.load()
        text2sql_task = Text2SQLTask()
        text2sql_task.current_dataset = "bird"
        text2sql_task.setup_database_dirs({"bird": bird_loader})
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
