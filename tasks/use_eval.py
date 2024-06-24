from utils import text2sql_evaluate, build_foreign_key_map_from_json

# import json
# kmap = build_foreign_key_map_from_json("../.text_2_sql_datasets/spider/tables.json")
# with open("out.json", "w") as file:
#     json.dump(kmap, file, indent=4)

# scores = text2sql_evaluate("../.text_2_sql_datasets/spider/dev_gold.sql", "../.text_2_sql_datasets/spider/dev_gold.sql", "../.text_2_sql_datasets/spider/test_database", "match", "../.text_2_sql_datasets/spider/tables.json", False, False, False)


scores = text2sql_evaluate(
    "../.text_2_sql_datasets/bird/dev.sql",
    "../.text_2_sql_datasets/bird/dev.sql",
    "../.text_2_sql_datasets/bird/dev_databases",
    "exec",
    "../.text_2_sql_datasets/bird/dev_tables.json",
    False,
    False,
    False,
)

print(scores)

# bird_run_eval("../.text_2_sql_datasets/bird/", "../.text_2_sql_datasets/bird/", "../.text_2_sql_datasets/bird/dev_databases", )
