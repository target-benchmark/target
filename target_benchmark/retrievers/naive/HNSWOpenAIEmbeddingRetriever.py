import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import tiktoken
import tqdm
from dotenv import load_dotenv
from openai import BadRequestError, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from target_benchmark.dictionary_keys import (
    DATABASE_ID_COL_NAME,
    TABLE_COL_NAME,
    TABLE_ID_COL_NAME,
)
from target_benchmark.retrievers import AbsCustomEmbeddingRetriever
from target_benchmark.retrievers.utils import (
    construct_embedding_index,
    markdown_table_str,
)

file_dir = os.path.dirname(os.path.realpath(__file__))
default_out_dir = os.path.join(file_dir, "retrieval_files", "openai")
default_embedding_model_id = "text-embedding-3-small"
default_max_tokens = 8192
default_tokenizer_id = "cl100k_base"


class HNSWOpenAIEmbeddingRetriever(AbsCustomEmbeddingRetriever):
    """
    The difference with the DefaultOpenAIEmbeddingRetriever is that it uses the
    HNSW index for consistent evaluation with the HySE retriever.
    """

    def __init__(
        self,
        out_dir: str = None,
        embedding_model_id: str = default_embedding_model_id,
        tokenizer_id: str = default_tokenizer_id,
        expected_corpus_format: str = "nested array",
        num_rows: Union[int, None] = None,
    ):
        super().__init__(expected_corpus_format=expected_corpus_format)

        load_dotenv()

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        if not out_dir:
            self.out_dir = Path(default_out_dir)
        else:
            self.out_dir = Path(out_dir)
        self.corpus_identifier = ""
        self.embedding_model_id = embedding_model_id
        # TODO: need to get this dynamically according to model id
        self.embedding_model_encoding = tiktoken.get_encoding(default_tokenizer_id)
        self.num_rows = num_rows
        self.corpus_index = None
        self.db_table_ids = None

    @classmethod
    def get_default_out_dir(cls) -> str:
        return str(default_out_dir)

    def retrieve(
        self,
        query: str,
        dataset_name: str,
        top_k: int,
        **kwargs,
    ):
        self._load_hnsw(self._get_corpus_identifier(dataset_name))
        query_embedding = self.embed_query(query)

        # Query dataset
        retrieved_ids, distances = self.corpus_index.knn_query(np.array(query_embedding), k=top_k)

        # Get original table_ids (table names) from the retrieved integer identifiers for each query
        retrieved_full_ids = [self.db_table_ids[id] for id in retrieved_ids[0]]

        return retrieved_full_ids

    def embed_corpus(self, dataset_name: str, corpus: Iterable[Dict]):
        """
        Function to embed the given corpus. This will be called in the evaluation pipeline before any retrieval.

        Parameters:
            dataset_name (str): the name of the corpus dataset.
            corpus (Iterable[Dict[str, List]]): an iterable of dicts, each being a batch of entries in the corpus dataset, containing database id, table id, the table contents (which the user can assume is in the format of self.expected_corpus_format), and context metadata (in these exact keys).
        Returns:
            nothing. the indexed embeddings are stored in a file.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)

        corpus_identifier = self._get_corpus_identifier(dataset_name)
        # get the paths to the persistence files
        idx_path, db_table_ids_path = self._construct_persistence_paths(corpus_identifier)

        if idx_path.exists() and db_table_ids_path.exists():
            print("using previously constructed index")
            return

        embedded_corpus = self._embed_corpus_parallel(corpus)
        corpus_index = construct_embedding_index(list(embedded_corpus.values()))

        # Store table embedding index and table ids in distinct files
        with open(idx_path, "wb") as f:
            pickle.dump(corpus_index, f)

        with open(db_table_ids_path, "wb") as f:
            pickle.dump(list(embedded_corpus.keys()), f)

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=32),
    )
    def embed_query(self, query: str) -> np.ndarray:
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model_id,
                input=query,
            )
            return np.array(response.data[0].embedding)
        except BadRequestError as e:
            print(type(query), len(query))
            raise e

    def _process_table(self, db_id: str, table_id: str, table: List[List[str]]) -> Tuple[Tuple[str, str], str]:
        tup_id = (db_id, table_id)
        num_rows_to_include = len(table)
        if self.num_rows:
            num_rows_to_include = self.num_rows
        while num_rows_to_include >= 0:
            table_str = markdown_table_str(table, num_rows=num_rows_to_include)
            num_tokens = len(self.embedding_model_encoding.encode(table_str))
            if num_tokens < default_max_tokens:  # this is not great, need to remove hardcode in future
                break
            num_rows_to_include -= 10

        if self.num_rows and num_rows_to_include != self.num_rows:
            print(f"truncated input due to context length constraints, included {num_rows_to_include} rows")
        return tup_id, self._chunking_text(table_str)

    def _chunking_text(self, text: str, max_tokens: int = default_max_tokens, overlap: int = 0):
        """

        break text into chunks if exceeding the limit of embedding model

        """
        tokens = self.embedding_model_encoding.encode(text)
        if len(tokens) > max_tokens:
            chunk_embs = []
            for i in range(0, len(tokens), max_tokens - overlap):
                chunk = tokens[i : i + max_tokens]
                chunk_embs.append(self.embed_query(self.embedding_model_encoding.decode(chunk)))
            return np.average(np.stack(chunk_embs, axis=0), axis=0)
        else:
            return self.embed_query(text)

    def _embed_corpus_parallel(self, corpus: Iterable[Dict]) -> Dict:
        with ThreadPoolExecutor() as executor:
            future_to_tup_id = [
                executor.submit(self._process_table, db_id, table_id, table)
                for corpus_dict in corpus
                for db_id, table_id, table in zip(
                    corpus_dict[DATABASE_ID_COL_NAME],
                    corpus_dict[TABLE_ID_COL_NAME],
                    corpus_dict[TABLE_COL_NAME],
                )
            ]
            embedded_corpus = {}
            for future in tqdm.tqdm(as_completed(future_to_tup_id), total=len(future_to_tup_id)):
                tup_id, embedded_table = future.result()
                embedded_corpus[tup_id] = embedded_table

        return embedded_corpus

    def _construct_persistence_paths(self, corpus_identifier: str) -> Tuple[Path, Path]:
        idx_path = self.out_dir / f"corpus_index_{corpus_identifier}.pkl"
        db_table_ids_path = self.out_dir / f"db_table_ids_{corpus_identifier}.pkl"
        return idx_path, db_table_ids_path

    def _load_hnsw(self, corpus_identifier: str):
        # no need to reload if passed in id is the same as current id
        # and the index and mappings loaded
        if corpus_identifier == self.corpus_identifier and self.corpus_index and self.db_table_ids:
            return

        # get the paths to the persistence files
        idx_path, db_table_ids_path = self._construct_persistence_paths(corpus_identifier)

        # throw error if files don't exist
        if not idx_path.exists() or not db_table_ids_path.exists():
            raise RuntimeError("cannot find the relevant index files.")

        # load files
        with open(idx_path, "rb") as f:
            self.corpus_index = pickle.load(f)

        with open(db_table_ids_path, "rb") as f:
            # stored separately as hnsw only takes int indices
            self.db_table_ids = pickle.load(f)
        # update the corpus identifier
        self.corpus_identifier = corpus_identifier

    def _get_corpus_identifier(self, dataset_name: str) -> str:
        corpus_identifier = f"{dataset_name}_numrows_all"
        if self.num_rows is not None:
            corpus_identifier = f"{dataset_name}_numrows_{self.num_rows}"
        return corpus_identifier
