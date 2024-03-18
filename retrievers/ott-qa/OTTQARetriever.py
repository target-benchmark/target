#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive mode for the tfidf DrQA retriever module."""

from drqa import retriever
import json
from AbsTargetDirectRetriever import AbsTargetDirectRetriever
from dataset_loaders.AbsTargetDatasetLoader import AbsTargetDatasetLoader


path_str = "retriever/title_sectitle_schema/index-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz"

class OTTQARetriever(AbsTargetDirectRetriever):
    def __init__(
            self,
            expected_corpus_format: str = 'nested array'
        ):
        super().__init__(expected_corpus_format)
        self.rankers: dict[str, retriever.TfidfDocRanker] = {}

    def retrieve(
        self,
        queries: dict[str, str],
        dataset_name: str,
        top_k: int,
        **kwargs,
    ) -> dict[str, list[str]]:
        retrieval_results = {}
        ranker = self.rankers[dataset_name]
        for query_id, query_str in queries.items():
            doc_names, doc_scores = ranker.closest_docs(query_str, top_k)
            retrieval_results[query_id, doc_names]
        return retrieval_results
    
    def embed_corpus(
        self,
        dataset_name: str,
        corpus: dict[str, object]
    ):

        self.rankers[dataset_name] = retriever.get_class('tfidf')(tfidf_path=path_str)
