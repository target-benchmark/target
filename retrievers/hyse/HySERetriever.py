import json
import os

from retrievers.AbsCustomEmbeddingRetriever import AbsCustomEmbeddingRetriever

from openai import OpenAI
from typing import Dict, Iterable, Iterator, List


class HyseRetriever(AbsCustomEmbeddingRetriever):

    def __init__(
        self,
        script_dir: str,
        expected_corpus_format: str = "nested array",
    ):
        super().__init__(expected_corpus_format)

        load_dotenv()

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.out_dir = "hyse_files/"

    def retrieve(
        self,
        query: str,
        dataset_name: str,
        top_k: int,
        **kwargs,
    ) -> List[str]:
        """
        Directly retrieves the predicted relevant tables for the query.

        Parameters:
            query (str): the actual query string.

            dataset_name (str): identifier for the dataset that these queries come from.
            since retrieval evaluation can be done for multiple datasets, use this as a way of choosing which dataset's corpus to retrieve from.

            top_k (int): the top k tables to retrieve for each query

            any additional kwargs you'd like to include.

        Returns:
            List[str]: the list of table ids of the retrieved tables.
        """
        with open(
            os.path.join(self.out_dir, f"corpus_index_{dataset_name}.pkl"), "rb"
        ) as f:
            corpus_index = pickle.load(f)

        with open(
            os.path.join(self.out_dir, f"table_ids_{dataset_name}.pkl"), "rb"
        ) as f:
            table_ids = pickle.load(f)

        # Query dataset, k - number of the closest elements (returns 2 numpy arrays)
        queries_retrieved_tables, distances = corpus_index.knn_query(
            np.array([emb for emb in queries_hyse_df["embedding"]]),
            # retrieves 10*2 tables, 10 tables per schema
            k=int(top_k / 2),
        )

        retrieved_table_ids = [table_ids[id] for id in retrieved_ids[0]] + [
            table_ids[id] for id in retrieved_ids[1]
        ]

        return retrieved_table_ids

    def embed_corpus(self, dataset_name: str, corpus: Iterable[dict]):
        """
        Cunction to embed the given corpus. This will be called in the evaluation pipeline before any retrieval.

        Parameters:
            dataset_name (str): the name of the corpus dataset.

            corpus (Iterable[Dict[str, object]]): an iterable of dictionaries,
            each dictionary mapping the table id to the table object (which the user can assume is in the format of self.expected_corpus_format).

        Returns:
            nothing. the indexed embeddings are stored in a file.
        """
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        embedded_corpus = {}
        for corpus_dict in corpus:
            # key = table_id, value = table
            for key, value in corpus_dict.items():
                embedded_corpus[key] = _embed_table(key, value)

        corpus_embeddings_df = (
            pd.DataFrame.from_records(embedded_corpus)
            .transpose()
            .rename({0: "embedding"}, axis=1)
        )

        # Constructing index
        corpus_index = hnswlib.Index(
            space="cosine", dim=len(corpus_embeddings_df["embedding"][0])
        )  # possible options are l2, cosine or ip

        # Initializing index - the maximum number of elements should be known beforehand
        corpus_index.init_index(
            max_elements=corpus_embeddings_df.shape[0], ef_construction=200, M=16
        )

        # Element insertion (can be called several times):
        corpus_index.add_items(
            np.array([emb for emb in corpus_embeddings_df["embedding"]]),
            corpus_embeddings_df.reset_index().index.tonumpy(),
            corpus_embeddings_df,
        )

        # Controlling the recall by setting ef:
        corpus_index.set_ef(50)  # ef should always be > k

        # Store table embedding index and table ids in distinct files
        with open(
            os.path.join(self.out_dir, f"corpus_index_{dataset_name}.pkl"), "wb"
        ) as f:
            pickle.dump(corpus_index, f)

        with open(
            os.path.join(self.out_dir, f"table_ids_{dataset_name}.pkl"), "wb"
        ) as f:
            pickle.dump(corpus_embeddings_df.index.tolist(), f)

    def _embed_table(table_id: str, table: List[List]) -> List[List]:
        """Embed table using default openai embedding model, only using table header."""

        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                # todo: add table values
                input=" ".join(table[0]),
            )
            return response.data[0].embedding
        except:
            print("error on: ", table_id)

    def generate_hypothetical_schemas(query: str) -> List[List]:
        """Generate a hypothetical schema relevant to answer the query."""

        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {
                    "role": "system",
                    # todo: add table values
                    # todo: expand to multi-table
                    "content": """
                  Generate exactly 2 table headers, which are different in semantics and size,
                  of tables which could potentially be used to answer the given query.
                  Return a list in which each item is a list of table attributes (strings).
                  """,
                },
                {"role": "user", "content": f"{query}"},
            ],
            temperature=0,
        )

        hypothetical_schemas = eval(
            response.to_dict()["choices"][0]["message"]["content"]
        )

        return hypothetical_schemas
