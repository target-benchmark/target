from typing import Literal

from target_benchmark.dataset_loaders.HFDatasetLoader import HFDatasetLoader


class NeedleInHaystackDataLoader(HFDatasetLoader):
    def __init__(
        self,
        dataset_name: str,
        hf_corpus_dataset_path: str,
        hf_queries_dataset_path: str,
        num_tables: int = None,
        split: Literal["test", "train", "validation"] = "train",
        data_directory: str = None,
        **kwargs,
    ):
        super().__init__(
            dataset_name=dataset_name,
            hf_corpus_dataset_path=hf_corpus_dataset_path,
            hf_queries_dataset_path=hf_queries_dataset_path,
            num_tables=num_tables,
            split=split,
            data_directory=data_directory,
            query_type="Other",
            kwargs=kwargs,
        )

    def _load_queries(self) -> None:
        pass

    def _raise_no_queries_error(self):
        raise NotImplementedError("Needle-in-haystack datasets have no queries!")

    def get_queries_for_task(self, batch_size: int = 64):
        self._raise_no_queries_error()

    def get_queries(self):
        self._raise_no_queries_error()

    def get_queries_size(self):
        self._raise_no_queries_error()

    def get_queries_header(self):
        self._raise_no_queries_error()
