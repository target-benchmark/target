import argparse
from pathlib import Path
from typing import List, Tuple

from target_benchmark.evaluators import TARGET
from target_benchmark.retrievers import (
    HNSWOpenAIEmbeddingRetriever,
    LlamaIndexRetriever,
    OTTQARetriever,
    RowSerializationRetriever,
    StellaEmbeddingRetriever,
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
    parser.add_argument("--retrieval_results_dir", type=str, default=None, help="folder to persist retrieval results.")
    parser.add_argument("--downstream_results_dir", type=str, default=None, help="folder to persist downstream results.")
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
    retrieval_results_dir: str = None,
    downstream_results_dir: str = None,
):
    results = []
    print(f"dir for retrieval results: {retrieval_results_dir}")
    for top_k in top_ks:
        performance = target.run(
            retriever=retriever,
            split=split,
            batch_size=100,
            top_k=top_k,
            retrieval_results_dir=retrieval_results_dir,
            downstream_results_dir=downstream_results_dir,
            dataset_name_for_dir=dataset_name,
        )
        results.append(performance)
        print(performance)
    return results


def write_performances(results, dataset_name: str, retrieval_results_dir: str, downstream_results_dir: str):
    def write_to_file(dir: str):
        path = Path(dir) / dataset_name / "performances.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as file:
            for result in results:
                file.write(str(result) + "\n")

    if downstream_results_dir:
        write_to_file(downstream_results_dir)
    elif retrieval_results_dir:
        write_to_file(retrieval_results_dir)


def initialize_retriever(retriever_name: str, num_rows: int = None, out_dir_appendix=None):
    if retriever_name == "llamaindex":
        return LlamaIndexRetriever()
    elif "hnsw_openai" in retriever_name:
        out_dir = None
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
    elif "stella" in retriever_name:
        return StellaEmbeddingRetriever(num_rows=num_rows)
    elif "row_serial" in retriever_name:
        return RowSerializationRetriever()
    else:
        raise ValueError(f"Passed in retriever {retriever_name} not yet supported")


def test_main(evals: List[Tuple[str, TARGET, str]]):
    args = parse_arguments()
    retriever_name = args.retriever_name
    num_rows = args.num_rows
    top_ks = args.top_ks
    retrieval_results_dir = args.retrieval_results_dir
    downstream_results_dir = args.downstream_results_dir
    retriever = initialize_retriever(retriever_name, num_rows)

    for dataset_name, target_eval, split in evals:
        results = run_eval_for_top_ks(
            retriever, retriever_name, top_ks, target_eval, dataset_name, split, retrieval_results_dir, downstream_results_dir
        )
        write_performances(results, dataset_name, retrieval_results_dir, downstream_results_dir)
