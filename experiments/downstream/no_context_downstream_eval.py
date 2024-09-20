from target_benchmark.evaluators import TARGET
from target_benchmark.retrievers import NoContextRetriever

target_fetaqa = TARGET(("Table Question Answering Task", "fetaqa"))
target_ottqa = TARGET(("Table Question Answering Task", "ottqa"))
target_tabfact = TARGET(("Fact Verification Task", "tabfact"))
target_spider = TARGET(("Text to SQL Task", "spider-test"))
target_bird = TARGET(("Text to SQL Task", "bird-validation"))

top_ks = [1]


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
            downstream_results_file=f"./no_context_{dataset_name}_downstream_results.jsonl",
        )
        results.append(performance)
        print(performance)
    return results


def write_performances(results, dataset_name):
    with open(f"./{dataset_name}_performances.jsonl", "w") as file:
        for result in results:
            file.write(str(result) + "\n")


retriever = NoContextRetriever()
retriever_name = "no_context"
persist = False
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
