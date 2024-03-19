from ..OTTQARetriever import OTTQARetriever

table = {
    "table_default": [["col 1"], ["row 1"]]
}

ottqa_retriever = OTTQARetriever()

ottqa_retriever.embed_corpus("test_dataset", table)