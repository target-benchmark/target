from utils import (
    initialize_retriever,
    parse_arguments,
    run_eval_for_top_ks,
    write_performances,
)

from target_benchmark.dataset_loaders.TargetDatasetConfig import (
    DEFAULT_FETAQA_DATASET_CONFIG,
    DEFAULT_GITTABLES_DATASET_CONFIG,
)
from target_benchmark.evaluators import TARGET
from target_benchmark.tasks import TableRetrievalTask


def main():
    args = parse_arguments()
    retriever_name = args.retriever_name
    num_rows = args.num_rows
    persist = args.persist
    top_ks = args.top_ks
    num_nih_tables = args.num_nih_tables

    retriever = initialize_retriever(retriever_name, num_rows, f"nih/{num_nih_tables}")
    gittables_config = DEFAULT_GITTABLES_DATASET_CONFIG.model_copy()
    gittables_config.num_tables = num_nih_tables
    table_retrieval_task = TableRetrievalTask(
        {
            "fetaqa": DEFAULT_FETAQA_DATASET_CONFIG,
            "gittables": gittables_config,
        }
    )
    evals = [
        (f"fetaqa_gittables_{num_nih_tables}", TARGET(table_retrieval_task), "test"),
        # ("ottqa_gittables", TARGET(("Table Retrieval Task", ["ottqa", "gittables"])), "validation"),
        # ("tabfact_gittables", TARGET(("Table Retrieval Task", ["tabfact", "gittables"])), "test"),
        # ("spider_gittables", TARGET(("Table Retrieval Task", ["spider-test", "gittables"])), "test"),
        # ("bird_gittables", TARGET(("Table Retrieval Task", ["bird-validation", "gittables"])), "validation"),
    ]
    for dataset_name, target_eval, split in evals:
        results = run_eval_for_top_ks(retriever, retriever_name, top_ks, target_eval, dataset_name, split, persist)
        write_performances(results, retriever_name, dataset_name)


if __name__ == "__main__":
    main()
