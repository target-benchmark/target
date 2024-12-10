from utils import test_main

from target_benchmark.evaluators import TARGET


def main():
    evals = [
        # ("fetaqa", TARGET(("Table Question Answering Task", "fetaqa")), "test"),
        # ("ottqa", TARGET(("Table Question Answering Task", "ottqa")), "validation"),
        # ("tabfact", TARGET(("Fact Verification Task", "tabfact")), "test"),
        ("spider-test", TARGET(("Text to SQL Task", "spider-test")), "test"),
        ("bird-validation", TARGET(("Text to SQL Task", "bird-validation")), "validation"),
    ]
    test_main(evals)


if __name__ == "__main__":
    main()
