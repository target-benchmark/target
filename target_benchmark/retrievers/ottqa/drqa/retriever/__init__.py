#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .BM25_doc_ranker import BM25DocRanker
from .doc_db import DocDB
from .tfidf_doc_ranker import TfidfDocRanker


def get_class(name):
    if name == "tfidf":
        return TfidfDocRanker
    if name == "bm25":
        return BM25DocRanker
    if name == "sqlite":
        return DocDB
    raise RuntimeError("Invalid retriever class: %s" % name)
