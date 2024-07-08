from dataset_loaders.LoadersDataModels import HFDatasetConfigDataModel

DEFAULT_FETAQA_DATASET_CONFIG = HFDatasetConfigDataModel(
    dataset_name="fetaqa",
    hf_corpus_dataset_path="target-benchmark/fetaqa-corpus",
    hf_queries_dataset_path="target-benchmark/fetaqa-queries",
    query_type="Table Question Answering",
)

DEFAULT_TABFACT_DATASET_CONFIG = HFDatasetConfigDataModel(
    dataset_name="tabfact",
    hf_corpus_dataset_path="target-benchmark/tabfact-corpus",
    hf_queries_dataset_path="target-benchmark/tabfact-queries",
    query_type="Fact Verification",
)

DEFAULT_OTTQA_DATASET_CONFIG = HFDatasetConfigDataModel(
    dataset_name="ottqa",
    hf_corpus_dataset_path="target-benchmark/ottqa-corpus",
    hf_queries_dataset_path="target-benchmark/ottqa-queries",
    query_type="Table Question Answering",
)

DEFAULT_INFAGENTDA_DATASET_CONFIG = HFDatasetConfigDataModel(
    dataset_name="infiagentda",
    hf_corpus_dataset_path="target-benchmark/infiagentda-corpus",
    hf_queries_dataset_path="target-benchmark/infiagentda-queries",
)

DEFAULT_SPIDER_DATASET_CONFIG = HFDatasetConfigDataModel(
    dataset_name="spider",
    hf_corpus_dataset_path="target-benchmark/spider-corpus",
    hf_queries_dataset_path="target-benchmark/spider-queries",
    query_type="Text to SQL",
    aux={
        "spider_zip_gdrive_url": "https://drive.google.com/uc?id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m"
    },
)

# TESTING ONLY
DEFAULT_DUMMY_DATASET_CONFIG = HFDatasetConfigDataModel(
    dataset_name="dummy-dataset",
    hf_corpus_dataset_path="jixy2012/mock-hf-corpus-dataset",
    hf_queries_dataset_path="jixy2012/mock-hf-queries-dataset",
    query_type="Table Question Answering",
)
