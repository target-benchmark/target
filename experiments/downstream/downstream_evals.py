import argparse
from pathlib import Path

from target_benchmark.evaluators import TARGET

# Create the parser
parser = argparse.ArgumentParser(description="Run downstream evals.")
parser.add_argument("retriever_name", type=str, help="name of the retriever")
args = parser.parse_args()


target_fetaqa = TARGET(("Table Question Answering Task", "fetaqa"))
target_ottqa = TARGET(("Table Question Answering Task", "ottqa"))
target_tabfact = TARGET(("Fact Verification Task", "tabfact"))
target_spider = TARGET(("Text to SQL Task", "spider-test"))
target_bird = TARGET(("Text to SQL Task", "bird-validation"))

retriever_name = args.retriever_name

retrieval_result_files_dir = (
    Path("/home/jixy2012/carl/target/experiments/retrieval") / retriever_name
)


fetaqa_test_retrieval_result = retrieval_result_files_dir / "fetaqa_10.jsonl"
results_fetaqa_test = target_fetaqa.evaluate_downstream_task(
    str(fetaqa_test_retrieval_result),
    "Table Question Answering Task",
    "test",
    f"./{retriever_name}_fetaqa_downstream_results.jsonl",
)
print(results_fetaqa_test)

with open("./fetaqa_performances.jsonl", "w") as file:
    for result in results_fetaqa_test.values():
        file.write(result.model_dump_json() + "\n")

ottqa_val_retrieval_result = retrieval_result_files_dir / "ottqa_10.jsonl"
results_ottqa_val = target_ottqa.evaluate_downstream_task(
    str(ottqa_val_retrieval_result),
    "Table Question Answering Task",
    "validation",
    f"./{retriever_name}_ottqa_downstream_results.jsonl",
)
print(results_ottqa_val)

with open("./ottqa_performances.jsonl", "w") as file:
    for result in results_ottqa_val.values():
        file.write(result.model_dump_json() + "\n")


tabfact_test_retrieval_result = retrieval_result_files_dir / "tabfact_10.jsonl"
results_tabfact_test = target_tabfact.evaluate_downstream_task(
    str(tabfact_test_retrieval_result),
    "Fact Verification Task",
    "test",
    f"./{retriever_name}_tabfact_downstream_results.jsonl",
)
print(results_tabfact_test)

with open("./tabfact_performances.jsonl", "w") as file:
    for result in results_tabfact_test.values():
        file.write(result.model_dump_json() + "\n")

spider_test_retrieval_result = retrieval_result_files_dir / "spider_1.jsonl"
results_spider_test = target_spider.evaluate_downstream_task(
    str(spider_test_retrieval_result),
    "Text to SQL Task",
    "test",
    f"./{retriever_name}_spider_downstream_results.jsonl",
)

with open("./spider_performances.jsonl", "w") as file:
    for result in results_spider_test.values():
        file.write(result.model_dump_json() + "\n")

bird_val_retrieval_result = retrieval_result_files_dir / "bird_1.jsonl"
results_bird_val = target_bird.evaluate_downstream_task(
    str(bird_val_retrieval_result),
    "Text to SQL Task",
    "validation",
    f"./{retriever_name}_bird_downstream_results.jsonl",
)

with open("./bird_performances.jsonl", "w") as file:
    for result in results_bird_val.values():
        file.write(result.model_dump_json() + "\n")
