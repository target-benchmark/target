from retrievers.ottqa.OTTQARetriever import OTTQARetriever
from dataset_loaders.HFDatasetLoader import HFDatasetLoader
import os
table = {
    "table_default": [["col 1"], ["row 1"]]
}

test_dataset_name = 'fetaqa'
test_hf_corpus_dataset_path = 'target-benchmark/fetaqa-queries'
test_hf_queries_dataset_path = 'target-benchmark/fetaqa-corpus'
fetaqa_loader = HFDatasetLoader(
    dataset_name=test_dataset_name,
    hf_corpus_dataset_path=test_hf_corpus_dataset_path,
    hf_queries_dataset_path=test_hf_queries_dataset_path,
    )

fetaqa_loader.load() # load the fetaqa data
dict_iter = fetaqa_loader.convert_corpus_table_to()

ottqa_retriever = OTTQARetriever(script_dir= os.path.dirname(os.path.abspath(__file__)))

ottqa_retriever.embed_corpus(fetaqa_loader.dataset_name, dict_iter)