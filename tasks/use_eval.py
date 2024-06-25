from tasks.utils import evaluate_sql_execution
from dataset_loaders.Text2SQLDatasetLoader import (
    default_spider_database_path,
    default_bird_database_path,
)
# import json
# kmap = build_foreign_key_map_from_json("../.text_2_sql_datasets/spider/tables.json")
# with open("out.json", "w") as file:
#     json.dump(kmap, file, indent=4)

# scores = text2sql_evaluate("../.text_2_sql_datasets/spider/dev_gold.sql", "../.text_2_sql_datasets/spider/dev_gold.sql", "../.text_2_sql_datasets/spider/test_database", "match", "../.text_2_sql_datasets/spider/tables.json", False, False, False)
predicted_sqls = [("SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1", "california_schools"), ("SELECT T2.A2, T2.A3 FROM account AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id INNER JOIN loan AS T3 ON T1.account_id = T3.account_id WHERE T3.loan_id = 4990", "financial")]
gold_sqls = [("SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1", "california_schools"), ("SELECT T2.A2, T2.A3 FROM account AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id INNER JOIN loan AS T3 ON T1.account_id = T3.account_id WHERE T3.loan_id = 4990", "financial")]
difficulties = ["easy", "hard"]
db_path = default_bird_database_path
scores = evaluate_sql_execution(
    predicted_sqls,
    gold_sqls,
    difficulties,
    db_path,
    3,
    60,
    True
)

print(scores)

# bird_run_eval("../.text_2_sql_datasets/bird/", "../.text_2_sql_datasets/bird/", "../.text_2_sql_datasets/bird/dev_databases", )
