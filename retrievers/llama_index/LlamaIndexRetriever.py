from typing import Dict, Iterable, List, Tuple
from retrievers import AbsCustomEmbeddingRetriever
from .embedding_utils import create_table_from_dataframe, construct_table_info
from dictionary_keys import *
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    InputComponent,
)
from llama_index.core import SQLDatabase, VectorStoreIndex
from pathlib import Path
from sqlalchemy import create_engine, MetaData

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

    def retrieve(
        self,
        query: str,
        dataset_name: str,
        top_k: int,
        **kwargs,
    ) -> List[Tuple]:
        retrieved_tables = self.query_pipelines[dataset_name].run(query=query)
        answers = []
        for res in retrieved_tables:
            answers.append(tuple(res.table_name.split(":")[:2]))
        print(answers)
        return answers

    def embed_corpus(self, dataset_name: str, corpus: Iterable[Dict]) -> None:
        table_infos = []
        metadata_obj = MetaData()

        dataset_persistence_path = data_path / dataset_name
        dataset_persistence_path.mkdir(parents=True, exist_ok=True)

        db_path = dataset_persistence_path / "database.db"

        engine = create_engine(f"sqlite:///{db_path}")
        sql_database = SQLDatabase(engine)

        for entry_batch in corpus:
            # create table info object for the table in corpus
            for i in range(len(entry_batch[TABLE_COL_NAME])):
                table = entry_batch[TABLE_COL_NAME][i]
                db_id = entry_batch[DATABASE_ID_COL_NAME][i]
                table_id = entry_batch[TABLE_ID_COL_NAME][i]
            table_info = construct_table_info(str(data_path), table, db_id, table_id)

            # append the table info to list of all table infos
            table_infos.append(table_info)

            # insert the table to the sql db.
            create_table_from_dataframe(
                table, table_info.table_name, engine, metadata_obj
            )

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
        # self.obj_retrievers[dataset_name] = obj_index.as_retriever(similarity_top_k=3)

        qp = QP(verbose=True)
        qp.add_modules(
            module_dict={
                "input": InputComponent(),
                "table_retriever": obj_index.as_retriever(similarity_top_k=3),
            }
        )
        qp.add_link("input", "table_retriever")
        self.query_pipelines[dataset_name] = qp
