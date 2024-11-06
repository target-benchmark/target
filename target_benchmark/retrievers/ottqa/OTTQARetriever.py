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
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Union

from dotenv import load_dotenv

from target_benchmark.retrievers.AbsCustomEmbeddingRetriever import (
    AbsCustomEmbeddingRetriever,
)
from target_benchmark.retrievers.ottqa.drqa import retriever
from target_benchmark.retrievers.ottqa.utils import (
    TFIDFBuilder,
    convert_table_representation,
    default_hash_size,
    default_tokenizer,
    get_filename,
)

file_dir = os.path.dirname(os.path.realpath(__file__))
default_out_dir = os.path.join(file_dir, "retrieval_files")


class OTTQARetriever(AbsCustomEmbeddingRetriever):
    def __init__(
        self,
        out_dir: str = default_out_dir,
        encoding: Literal["tfidf", "bm25"] = "tfidf",
        withtitle: bool = False,
        ngram: int = 2,
        hash_size: int = default_hash_size,
        tokenizer: str = default_tokenizer,
        expected_corpus_format: str = "nested array",
    ):
        super().__init__(expected_corpus_format)

        load_dotenv()

        self.out_dir = out_dir
        self.rankers: Dict[str, Union[retriever.TfidfDocRanker, retriever.BM25DocRanker]] = {}
        self.withtitle = withtitle
        self.encoding = encoding
        self.ngram = ngram
        self.hash_size = hash_size
        self.tokenizer = tokenizer

        assert encoding in [
            "tfidf",
            "bm25",
        ], "encoding unknown, should be tfidf or bm25"

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
        path_to_out_dir = Path(self.out_dir)
        path_to_out_dir.mkdir(parents=True, exist_ok=True)

        out_path = get_filename(
            self.out_dir,
            self.encoding,
            self.withtitle,
            self.ngram,
            self.hash_size,
            self.tokenizer,
            dataset_name,
        )
        file_name = f"{dataset_name}_{self.encoding}_{self.withtitle}.json"
        path_to_persist_file = Path(self.out_dir) / file_name
        # either load or create converted corpus
        if Path(out_path).exists():
            print(f"file_name: {path_to_persist_file}")
            out_path = get_filename(
                self.out_dir,
                self.encoding,
                self.withtitle,
                self.ngram,
                self.hash_size,
                self.tokenizer,
                dataset_name,
            )
        else:
            converted_corpus = {}
            if path_to_persist_file.exists():
                with open(path_to_persist_file, "r") as file:
                    converted_corpus = json.load(file)
            if not path_to_persist_file.exists():
                converted_corpus = self.create_converted_corpus(corpus)
                with open(path_to_persist_file, "w") as file:
                    # Write the dictionary to a file in JSON format
                    json.dump(converted_corpus, file)
            builder = TFIDFBuilder()
            out_path = builder.build_tfidf(
                self.out_dir,
                converted_corpus,
                dataset_name=dataset_name,
                option=self.encoding,
                ngram=self.ngram,
                hash_size=self.hash_size,
                tokenizer=self.tokenizer,
                with_title=self.withtitle,
            )
        self.rankers[dataset_name] = retriever.get_class(self.encoding)(tfidf_path=out_path)

    def create_converted_corpus(
        self,
        corpus: Iterable[Dict],
    ) -> Dict:
        converted_corpus = {}
        for entry in corpus:
            for db_id, table_id, table, context in zip(
                entry["database_id"],
                entry["table_id"],
                entry["table"],
                entry["context"],
            ):
                # Setting to evaluate influence of table name in embedding

                tup = (db_id, table_id)

                converted_corpus[str(tup)] = convert_table_representation(
                    db_id,
                    table_id,
                    table,
                    context["section_title"] if context and "section_title" in context else "",
                    self.withtitle,
                )
        return converted_corpus
