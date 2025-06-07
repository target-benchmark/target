import json
import random
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional

from huggingface_hub import snapshot_download

from target_benchmark.dataset_loaders import HFDatasetLoader
from target_benchmark.dataset_loaders.utils import (
    convert_tables_by_format,
    set_in_memory_data_format,
    write_table_to_path,
)
from target_benchmark.dictionary_keys import (
    CONTEXT_COL_NAME,
    DATABASE_ID_COL_NAME,
    TABLE_COL_NAME,
    TABLE_ID_COL_NAME,
)


class Text2SQLDatasetLoader(HFDatasetLoader):
    def __init__(
        self,
        dataset_name: str,
        hf_corpus_dataset_path: str,
        hf_queries_dataset_path: str,
        num_tables: int = None,
        split: Literal["test", "train", "validation"] = "test",
        data_directory: str = None,
        **kwargs,
    ):
        # NOTE: due to hf dataset typing constraints, we are unable to put text2sql
        # corpus into hf datasets. so expect a dictionary when getting `.corpus`

        super().__init__(
            dataset_name=dataset_name,
            hf_corpus_dataset_path=hf_corpus_dataset_path,
            hf_queries_dataset_path=hf_queries_dataset_path,
            num_tables=num_tables,
            split=split,
            data_directory=data_directory,
            query_type="Text to SQL",
            kwargs=kwargs,
        )
        self.path_to_database_dir: str = None
        self.corpus: Dict = None

    def _load_corpus(self) -> None:
        path_to_data_dir = snapshot_download(repo_id=self.hf_corpus_dataset_path, repo_type="dataset")
        time.sleep(0.5)
        path_to_context = Path(path_to_data_dir, f"corpus-{self.split}.json")
        with open(path_to_context, "r") as file:
            self.corpus = json.load(file)
        self.path_to_database_dir = Path(path_to_data_dir, f"{self.split}_database")

    def get_path_to_database(self) -> Path:
        return self.path_to_database_dir

    def persist_corpus_to(self, format: Literal["csv", "json"], path: str = None) -> None:
        """
        Saves the tables in the corpus to a specified location and format.

        Parameters:
            format (Literal["csv", "json"]): The format in which to save the corpus (e.g., 'csv', 'json').
            path (str, optional): The file system path where the corpus should be saved. creates directory if the directory doesn't exist. if no path is given, defaults to self.data_directory. if self.data_directory is also None, throw ValueError.

        Raises:
            Runtime error if the corpus has not been loaded.
        """
        if not self.corpus:
            raise RuntimeError("Corpus has not been loaded!")

        if not path:
            if not self.data_directory:
                raise ValueError("No path for persistence is specified!")
            path = self.data_directory

        path_to_write_to = Path(path)
        if path_to_write_to.suffix:
            raise ValueError(f"this path {path_to_write_to} looks like a path to a file.")
        if not path_to_write_to.exists():
            path_to_write_to.mkdir(parents=True, exist_ok=True)

        split_path = path_to_write_to / self.split
        if not split_path.exists():
            split_path.mkdir(parents=True, exist_ok=True)
        for i in range(len(self.corpus[TABLE_COL_NAME])):
            table_name = Path(self.corpus[TABLE_ID_COL_NAME][i])
            nested_array = self.corpus[TABLE_ID_COL_NAME][i]
            write_table_to_path(format, table_name, split_path, nested_array)

    def get_corpus_iter(
        self,
        output_format: str = "nested array",
        batch_size: int = 1,
        num_tables: Optional[int] = None,
        seed: Optional[int] = 42,
    ):
        if not self.corpus:
            raise RuntimeError("Corpus has not been loaded!")

        in_memory_format = set_in_memory_data_format(output_format)
        corpus = self.corpus.copy()
        if num_tables is not None:
            assert (
                num_tables <= self.get_corpus_size()
            ), f"number of tables ({num_tables}) to sample cannot exceed the number of tables in the corpus ({self.get_corpus_size()})"
            random.seed(seed)
            idx = random.choices([i for i in range(self.get_corpus_size())], k=num_tables)
            for key, value in corpus.items():
                corpus[key] = [value[i] for i in idx]
        for i in range(0, len(self.corpus[TABLE_COL_NAME]), batch_size):
            batch = {}
            # Use list comprehensions to extract each column
            batch[TABLE_COL_NAME] = convert_tables_by_format(
                tables=corpus[TABLE_COL_NAME][i : i + batch_size],
                format=in_memory_format,
            )
            batch[DATABASE_ID_COL_NAME] = corpus[DATABASE_ID_COL_NAME][i : i + batch_size]
            batch[TABLE_ID_COL_NAME] = corpus[TABLE_ID_COL_NAME][i : i + batch_size]
            batch[CONTEXT_COL_NAME] = corpus[CONTEXT_COL_NAME][i : i + batch_size]
            yield batch

    def get_corpus(self) -> List[Dict]:
        """
        get the corpus of the loaded dataset. if the dataset has not been loaded, raise an error.

        Returns:
            a Dictionary object

        Raises:
            a runtime error if corpus has not been loaded yet.

        """
        if not self.corpus:
            raise RuntimeError("Corpus datasets have not been loaded!")
        return self.corpus

    def get_corpus_size(self) -> int:
        """
        Get the number of tables in the corpus.

        Returns:
            number of tables.

        Raises:
            a runtime error if corpus has not been loaded yet.
        """
        if not self.corpus:
            raise RuntimeError("Corpus datasets have not been loaded!")
        return len(self.corpus[TABLE_COL_NAME])

    def get_corpus_header(self) -> List[str]:
        """
        returns the header of this dataset's corpus

        Returns:
            a dictionary containing the headers of the corpus
        """
        if not self.corpus:
            raise RuntimeError("Corpus datasets have not been loaded!")
        return list(self.corpus.keys())
