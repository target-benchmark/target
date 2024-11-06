from typing import Literal

from datasets import load_dataset

from target_benchmark.dataset_loaders.AbsDatasetLoader import AbsDatasetLoader


class HFDatasetLoader(AbsDatasetLoader):
    def __init__(
        self,
        dataset_name: str,
        hf_corpus_dataset_path: str,
        hf_queries_dataset_path: str,
        num_tables: int = None,
        split: Literal["test", "train", "validation"] = "test",
        data_directory: str = None,
        query_type: str = "",
        **kwargs
    ):
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            data_directory=data_directory,
            query_type=query_type,
            num_tables=num_tables,
            **kwargs,
        )
        """
        Constructor for a generic dataset loader that loads from a huggingface dataset.
        Parameters:
            hf_corpus_dataset_path (str): the path to your huggingface hub corpus dataset. it will look something like target-benchmark/fetaqa-corpus (namespace/dataset-name)
            hf_queries_dataset_path (str): the path to your huggingface hub queries dataset path.
        """

        self.hf_corpus_dataset_path = hf_corpus_dataset_path
        self.hf_queries_dataset_path = hf_queries_dataset_path

    def _load_corpus(self) -> None:
        if not self.corpus:
            self.corpus = load_dataset(path=self.hf_corpus_dataset_path, split=self.split)

    def _load_queries(self) -> None:
        if not self.queries:
            self.queries = load_dataset(path=self.hf_queries_dataset_path, split=self.split)
