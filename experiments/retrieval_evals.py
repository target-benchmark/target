from utils import test_main

from target_benchmark.evaluators import TARGET


def main():
    evals = [
        # ("fetaqa", TARGET(("Table Retrieval Task", "fetaqa")), "test"),
        ("ottqa", TARGET(("Table Retrieval Task", "ottqa")), "validation"),
        # ("tabfact", TARGET(("Table Retrieval Task", "tabfact")), "test"),
        # ("spider", TARGET(("Table Retrieval Task", "spider-test")), "test"),
        # ("bird", TARGET(("Table Retrieval Task", "bird-validation")), "validation"),
    ]
    test_main(evals)


if __name__ == "__main__":
    main()
