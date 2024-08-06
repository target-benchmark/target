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

    def retrieve(
        self,
        query: str,
        dataset_name: str,
        top_k: int,
        **kwargs,
    ) -> List[Tuple]:
        res = self.query_pipelines[dataset_name].run(query=query)
        return str(res)


    def embed_corpus(self, dataset_name: str, corpus: Iterable[Dict]) -> None:
        table_infos = [] # TODO: use table infos from utils
        metadata_obj = MetaData()

        db_path = data_path / dataset_name / "database.db"
        engine = create_engine(f'sqlite:///{db_path}')
        sql_database = SQLDatabase(engine)

        for entry in corpus:
            # create table info object for the table in corpus
            table_info = construct_table_info(str(data_path), entry[TABLE_COL_NAME], entry[DATABASE_ID_COL_NAME], entry[TABLE_ID_COL_NAME])

            # append the table info to list of all table infos
            table_infos.append(table_info)

            # insert the table to the sql db.
            create_table_from_dataframe(entry[TABLE_COL_NAME], table_info.table_name, engine, metadata_obj)

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
            module_dict= {
                "input": InputComponent(),
                "table_retriever": obj_index.as_retriever(similarity_top_k=3),
            }
        )
        qp.add_link("input", "table_retriever")
        self.query_pipelines[dataset_name] = qp
