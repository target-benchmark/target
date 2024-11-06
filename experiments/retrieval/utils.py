import argparse
from pathlib import Path

from target_benchmark.evaluators import TARGET
from target_benchmark.retrievers import (
    HNSWOpenAIEmbeddingRetriever,
    LlamaIndexRetriever,
    OTTQARetriever,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run downstream evals.")
    parser.add_argument("--retriever_name", type=str, help="name of the retriever")
    parser.add_argument("--num_rows", type=int, default=100, help="num rows to include for hnsw")
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Whether to persist the data. Defaults to False.",
    )
    parser.add_argument(
        "--top_ks",
        type=str,
        default="1 5 10 25 50",
        help="Space separated list of top ks. for example '1 3 5 10'.",
    )
    parser.add_argument("--num_nih_tables", type=int, default=None, help="number of tables to include in the NIH dataset")
    args = parser.parse_args()
    args.top_ks = [int(k) for k in args.top_ks.split(" ")]
    return args


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
            path = Path("./") / retriever_name / dataset_name / f"{top_k}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            persist_path = str(path)
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


def write_performances(results, retriever_name: str, dataset_name: str):
    path = Path("./") / retriever_name / dataset_name / "performances.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        for result in results:
            file.write(str(result) + "\n")


def initialize_retriever(retriever_name: str, num_rows: int = None, out_dir_appendix=None):
    if retriever_name == "llamaindex":
        return LlamaIndexRetriever()
    elif "hnsw_openai" in retriever_name:
        if out_dir_appendix:
            out_dir = Path(HNSWOpenAIEmbeddingRetriever.get_default_out_dir()) / out_dir_appendix
        return HNSWOpenAIEmbeddingRetriever(num_rows=num_rows, out_dir=out_dir)
    elif "tfidf_no_title" in retriever_name:
        return OTTQARetriever(encoding="tfidf", withtitle=False)
    elif "tfidf_with_title" in retriever_name:
        return OTTQARetriever(encoding="tfidf", withtitle=True)
    elif "bm25_no_title" in retriever_name:
        return OTTQARetriever(encoding="bm25", withtitle=False)
    elif "bm25_with_title" in retriever_name:
        return OTTQARetriever(encoding="bm25", withtitle=True)
    else:
        raise ValueError(f"Passed in retriever {retriever_name} not yet supported")
