from utils import (
    initialize_retriever,
    parse_arguments,
    run_eval_for_top_ks,
    write_performances,
)

from target_benchmark.evaluators import TARGET


def main():
    args = parse_arguments()
    retriever_name = args.retriever_name
    num_rows = args.num_rows
    persist = args.persist
    top_ks = args.top_ks

    retriever = initialize_retriever(retriever_name, num_rows)

    evals = [
        ("fetaqa", TARGET(("Table Retrieval Task", "fetaqa")), "test"),
        ("ottqa", TARGET(("Table Retrieval Task", "ottqa")), "validation"),
        ("tabfact", TARGET(("Table Retrieval Task", "tabfact")), "test"),
        ("spider", TARGET(("Table Retrieval Task", "spider-test")), "test"),
        ("bird", TARGET(("Table Retrieval Task", "bird-validation")), "validation"),
    ]
    for dataset_name, target_eval, split in evals:
        results = run_eval_for_top_ks(retriever, retriever_name, top_ks, target_eval, dataset_name, split, persist)
        write_performances(results, retriever_name, dataset_name)


if __name__ == "__main__":
    main()
