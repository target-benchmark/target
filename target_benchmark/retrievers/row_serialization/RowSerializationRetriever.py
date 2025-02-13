from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from target_benchmark.dictionary_keys import (
    DATABASE_ID_COL_NAME,
    METADATA_DB_ID_KEY_NAME,
    METADATA_TABLE_ID_KEY_NAME,
    TABLE_COL_NAME,
    TABLE_ID_COL_NAME,
)
from target_benchmark.retrievers import AbsCustomEmbeddingRetriever

file_dir = Path(__file__).parent / "storage"


class RowSerializationRetriever(AbsCustomEmbeddingRetriever):
    def __init__(self, embeddings_dir: str = str(file_dir)):
        super().__init__("nested array")
        self.model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()
        self.client = QdrantClient(Path(embeddings_dir))
        self.emb_dims = self.model.get_sentence_embedding_dimension()

    def retrieve(self, query: str, dataset_name: str, top_k: int, **kwargs) -> List[Tuple]:
        num_points = self.client.count(collection_name=dataset_name, exact=True)
        results_dict = {}
        search_limit = top_k
        idx = 0
        while len(results_dict) < top_k and search_limit <= num_points:
            result = self.client.search(
                collection_name=dataset_name,
                query_vector=self.embed_query(query, dataset_name),
                limit=search_limit,
                with_payload=True,
            )
            while len(results_dict) < top_k and idx < len(result):
                scored_point = result[idx]
                table_id = (
                    scored_point.payload[METADATA_DB_ID_KEY_NAME],
                    scored_point.payload[METADATA_TABLE_ID_KEY_NAME],
                )
                if table_id not in results_dict:
                    results_dict[table_id] = idx
                idx += 1
            search_limit *= 2
        return sorted(results_dict, key=results_dict.get)

    def embed_corpus(self, dataset_name: str, corpus: Iterable[Dict]):
        if self.client.collection_exists(dataset_name):
            return
        self.client.create_collection(
            collection_name=dataset_name,
            vectors_config=models.VectorParams(size=self.emb_dims, distance=models.Distance.COSINE),
        )
        metadata = []
        vectors = []
        for entry in corpus:
            entry = {key: value[0] for key, value in entry.items()}
            # serialize table into a list of strings
            serialized_table = self._serialize_table(entry[TABLE_COL_NAME])
            # encode the table
            table_embedding = self.model.encode(serialized_table)

            # check how many duplicate metadata entries are needed
            num_metadata = 1
            if table_embedding.ndim == 1:
                table_embedding = np.expand_dims(table_embedding, axis=0)
            else:
                num_metadata = table_embedding.shape[0]

            # shape of table_embedding: <num_vectors x dim of vector>
            vectors.append(table_embedding)
            # for all vectors corresponding to the same table,
            # you need the same metadata entry
            metadata.extend(
                [
                    {
                        METADATA_TABLE_ID_KEY_NAME: entry[TABLE_ID_COL_NAME],
                        METADATA_DB_ID_KEY_NAME: entry[DATABASE_ID_COL_NAME],
                    }
                ]
                * num_metadata
            )

        vectors = np.concatenate(vectors, axis=0)
        assert vectors.shape[0] == len(
            metadata
        ), f"Mismatch between vectors shape and the metadata entries! Shape: {vectors.shape[0]}, Metadata entries: {len(metadata)}"
        self.client.upload_collection(
            dataset_name,
            vectors=vectors,
            payload=metadata,
        )

    def _serialize_table(self, corpus_entry: Dict) -> List[str]:
        table = corpus_entry[TABLE_COL_NAME]  # get the table as a nested array
        serialized_rows = []  # keep an array, each element will be a serialized row
        headers = table[0]  # grab table column names
        for row in corpus_entry[TABLE_COL_NAME][1:]:
            assert len(headers) == len(row)
            row_str = ""  # build the table string
            for col_name, cell_value in zip(headers, row):
                # match each column name to the cell value.
                row_str += f"{col_name} is {cell_value}, "
            serialized_rows.append(row_str)
        return serialized_rows
