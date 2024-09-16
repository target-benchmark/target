import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from llama_index.core import SQLDatabase, VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SQLTableNodeMapping, SQLTableSchema
from llama_index.core.query_pipeline import InputComponent
from llama_index.core.query_pipeline import QueryPipeline as QP
from sqlalchemy import MetaData, create_engine

from target_benchmark.dictionary_keys import (
    DATABASE_ID_COL_NAME,
    TABLE_COL_NAME,
    TABLE_ID_COL_NAME,
)
from target_benchmark.retrievers import AbsCustomEmbeddingRetriever

from .embedding_utils import construct_table_info, create_table_from_dataframe

cur_dir_path = Path(__file__).parent.resolve()
data_path = cur_dir_path / "data/"


class LlamaIndexRetriever(AbsCustomEmbeddingRetriever):
    def __init__(self):
        """
        Parameters:
            expected_corpus_format (str, optional): a string indicating what corpus format (ie nested array, dictionary, pandas df, etc.) the `embed_corpus` function expects from its input.
        """
        super().__init__(expected_corpus_format="dataframe")
        self.query_pipelines = {}
        self.object_indices = {}
        self.top_ks = {}

    def retrieve(
        self,
        query: str,
        dataset_name: str,
        top_k: int,
        **kwargs,
    ) -> List[Tuple]:
        if (
            dataset_name not in self.top_ks
            or top_k != self.top_ks[dataset_name]
            or dataset_name not in self.query_pipelines
        ):
            # check if we need to reconstruct new query pipeline
            # if the top k has changed, or a qp has never been constructed, construct the query pipeline with the new top k
            self.construct_query_pipeline(dataset_name, top_k)
        dataset_persistence_path = data_path / dataset_name
        mapping_path = dataset_persistence_path / "name_mapping.json"
        if not dataset_persistence_path.exists() or not mapping_path.exists():
            raise ValueError(
                "embedding data has not been created or persisted! aborting retrieval"
            )
        with open(mapping_path, "r") as mapping_file:
            # load the generated -> real table name mapping
            mapping = json.load(mapping_file)

        retrieved_tables = self.query_pipelines[dataset_name].run(query=query)
        answers = []
        for res in retrieved_tables:
            try:
                answers.append(tuple(mapping[res.table_name]))
            except KeyError:
                raise ValueError(
                    f"retrieved table name {res} does not map to a table in the dataset. Aborting retrieval!"
                )

        return answers

    def embed_corpus(self, dataset_name: str, corpus: Iterable[Dict]) -> None:
        table_infos = []
        metadata_obj = MetaData()

        dataset_persistence_path = data_path / dataset_name
        dataset_persistence_path.mkdir(parents=True, exist_ok=True)

        db_path = dataset_persistence_path / "database.db"
        mapping_path = dataset_persistence_path / "name_mapping.json"

        engine = create_engine(f"sqlite:///{db_path}")
        sql_database = SQLDatabase(engine)
        mapping = {}
        for entry_batch in corpus:
            # create table info object for the table in corpus
            for i in range(len(entry_batch[TABLE_COL_NAME])):
                table = entry_batch[TABLE_COL_NAME][i]
                db_id = entry_batch[DATABASE_ID_COL_NAME][i]
                table_id = entry_batch[TABLE_ID_COL_NAME][i]
                table_info = construct_table_info(
                    str(dataset_persistence_path), table, db_id, table_id, mapping
                )
                if not table_info:
                    raise ValueError("a valid table name cannot be generated!")

                # append the table info to list of all table infos
                table_infos.append(table_info)
                real_table_name = [str(db_id), table_id]
                mapping[table_info.table_name] = real_table_name
                # insert the table to the sql db.
                create_table_from_dataframe(
                    table, table_info.table_name, engine, metadata_obj
                )
        with open(mapping_path, "w") as mapping_file:
            json.dump(mapping, mapping_file)

        # construct retriever
        table_node_mapping = SQLTableNodeMapping(sql_database)
        table_schema_objs = [
            SQLTableSchema(table_name=t.table_name, context_str=t.table_summary)
            for t in table_infos
        ]  # add a SQLTableSchema for each table

        obj_index = ObjectIndex.from_objects(
            table_schema_objs,
            table_node_mapping,
            VectorStoreIndex,
        )
        self.object_indices[dataset_name] = obj_index

    def construct_query_pipeline(self, dataset_name: str, top_k: int):
        query_pipeline = QP(verbose=False)
        query_pipeline.add_modules(
            module_dict={
                "input": InputComponent(),
                "table_retriever": self.object_indices[dataset_name].as_retriever(
                    similarity_top_k=top_k
                ),
            }
        )
        query_pipeline.add_link("input", "table_retriever")
        self.query_pipelines[dataset_name] = query_pipeline
        self.top_ks[dataset_name] = top_k
