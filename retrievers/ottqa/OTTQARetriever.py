#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive mode for the tfidf DrQA retriever module."""
import ast
import json
import os

from dotenv import load_dotenv
from typing import Dict, Iterable, List

from .drqa import retriever
from .utils import convert_table_representation, TFIDFBuilder
from ..AbsCustomEmbeddingRetriever import AbsCustomEmbeddingRetriever


class OTTQARetriever(AbsCustomEmbeddingRetriever):
    def __init__(
        self,
        encoding: str = "tfidf",
        expected_corpus_format: str = "nested array",
    ):
        super().__init__(expected_corpus_format)

        load_dotenv()

        self.rankers: Dict[str, Union[retriever.TfidfDocRanker, retriever.BM25DocRanker]] = {}
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.out_dir = os.path.join(file_dir, "title_sectitle_schema/")
        self.encoding = encoding

        assert encoding in ["tfidf", "bm25"], "encoding unknown, should be tfidf or bm25"

    def retrieve(
        self,
        query: str,
        dataset_name: str,
        top_k: int,
        **kwargs,
    ) -> List[str]:
        ranker = self.rankers[dataset_name]
        doc_names, doc_scores = ranker.closest_docs(query, top_k)
        return [ast.literal_eval(doc_name) for doc_name in doc_names]

    def embed_corpus(self, dataset_name: str, corpus: Iterable[Dict]):
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        converted_corpus = {}
        for entry in corpus:
            for db_id, table_id, table in zip(
                entry["database_id"], entry["table_id"], entry["table"]
            ):
                tup = (db_id, table_id)
                converted_corpus[str(tup)] = convert_table_representation(
                    db_id, table_id, table
                )
        file_name = "temp_data.json"

        # Write the dictionary to a file in JSON format
        with open(os.path.join(self.out_dir, file_name), "w") as f:
            json.dump(converted_corpus, f)
        
        # TODO: make configurable tfidf versus bm25
        builder = TFIDFBuilder()
        out_path = builder.build_tfidf(self.out_dir, converted_corpus)
        self.rankers[dataset_name] = retriever.get_class(self.encoding)(tfidf_path=out_path)
