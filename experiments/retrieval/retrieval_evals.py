import argparse

from target_benchmark.evaluators import TARGET
from target_benchmark.retrievers import (
    HNSWOpenAIEmbeddingRetriever,
    LlamaIndexRetriever,
)

parser = argparse.ArgumentParser(description="Run downstream evals.")
parser.add_argument("--retriever_name", type=str, help="name of the retriever")
parser.add_argument(
    "--num_rows", type=int, default=100, help="num rows to include for hnsw"
)
parser.add_argument(
    "--persist",
    action="store_true",
    help="Whether to persist the data. Defaults to False.",
)
args = parser.parse_args()
retriever_name = args.retriever_name
num_rows = args.num_rows
persist = args.persist
print(f"persist: {persist}")

target_fetaqa = TARGET(("Table Retrieval Task", "fetaqa"))
target_ottqa = TARGET(("Table Retrieval Task", "ottqa"))
target_tabfact = TARGET(("Table Retrieval Task", "tabfact"))
target_spider = TARGET(("Table Retrieval Task", "spider-test"))
target_bird = TARGET(("Table Retrieval Task", "bird-validation"))
target_infiagentda = TARGET(("Table Retrieval Task", "infiagentda"))

top_ks = [1, 5, 10, 25, 50]


def run_eval_for_top_ks(
    retriever,
    retriever_name: str,
    top_ks: list[int],
    target: TARGET,
    dataset_name: str,
    split: str,
    persist: bool = False,
):
    results = []
    persist_path = None

    for top_k in top_ks:
        if persist:
            persist_path = f"./{dataset_name}_{top_k}.jsonl"
        else:
            persist_path = None
        performance = target.run(
            retriever=retriever,
            split=split,
            batch_size=100,
            top_k=top_k,
            retrieval_results_file=persist_path,
        )
        results.append(performance)
        print(performance)
    return results


def write_performances(results, dataset_name):
    with open(f"./{dataset_name}_performances.jsonl", "w") as file:
        for result in results:
            file.write(str(result) + "\n")


retriever = None
if retriever_name == "llamaindex":
    retriever = LlamaIndexRetriever()
elif "hnsw_openai" in retriever_name:
    print(num_rows)
    retriever = HNSWOpenAIEmbeddingRetriever(num_rows=num_rows)

# fetaqa test
results_fetaqa_test = run_eval_for_top_ks(
    retriever, retriever_name, top_ks, target_fetaqa, "fetaqa", "test", persist
)
write_performances(results=results_fetaqa_test, dataset_name="fetaqa")

# ottqa
results_ottqa_val = run_eval_for_top_ks(
    retriever, retriever_name, top_ks, target_ottqa, "ottqa", "validation", persist
)
write_performances(results=results_ottqa_val, dataset_name="ottqa")

# tabfact
results_tabfact_test = run_eval_for_top_ks(
    retriever, retriever_name, top_ks, target_tabfact, "tabfact", "test", persist
)
write_performances(results=results_tabfact_test, dataset_name="tabfact")

# spider
results_spider_test = run_eval_for_top_ks(
    retriever, retriever_name, top_ks, target_spider, "spider", "test", persist
)
write_performances(results=results_spider_test, dataset_name="spider")

# bird
results_bird_validation = run_eval_for_top_ks(
    retriever, retriever_name, top_ks, target_bird, "bird", "validation", persist
)
write_performances(results=results_bird_validation, dataset_name="bird")
