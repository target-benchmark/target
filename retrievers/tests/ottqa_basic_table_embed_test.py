from retrievers.ottqa.OTTQARetriever import OTTQARetriever
import os
table = {
    "table_default": [["col 1"], ["row 1"]]
}

ottqa_retriever = OTTQARetriever(script_dir= os.path.dirname(os.path.abspath(__file__)))

ottqa_retriever.embed_corpus("test_dataset", table)