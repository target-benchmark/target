from target_benchmark.dataset_loaders.LoadersDataModels import (
    HFDatasetConfigDataModel,
    NeedleInHaystackDatasetConfigDataModel,
    Text2SQLDatasetConfigDataModel,
)

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
    split="validation",
)

DEFAULT_INFAGENTDA_DATASET_CONFIG = HFDatasetConfigDataModel(
    dataset_name="infiagentda",
    hf_corpus_dataset_path="target-benchmark/infiagentda-corpus",
    hf_queries_dataset_path="target-benchmark/infiagentda-queries",
    query_type="Other",
)

DEFAULT_SPIDER_TEST_DATASET_CONFIG = Text2SQLDatasetConfigDataModel(
    dataset_name="spider-test",
    hf_corpus_dataset_path="target-benchmark/spider-corpus-test",
    hf_queries_dataset_path="target-benchmark/spider-queries-test",
    query_type="Text to SQL",
    split="test",
)

DEFAULT_SPIDER_TRAIN_DATASET_CONFIG = Text2SQLDatasetConfigDataModel(
    dataset_name="spider-train",
    hf_corpus_dataset_path="target-benchmark/spider-corpus-train",
    hf_queries_dataset_path="target-benchmark/spider-queries-train",
    query_type="Text to SQL",
    split="train",
)

DEFAULT_SPIDER_VALIDATION_DATASET_CONFIG = Text2SQLDatasetConfigDataModel(
    dataset_name="spider-validation",
    hf_corpus_dataset_path="target-benchmark/spider-corpus-validation",
    hf_queries_dataset_path="target-benchmark/spider-queries-validation",
    query_type="Text to SQL",
    split="train",
)


DEFAULT_BIRD_VALIDATION_DATASET_CONFIG = Text2SQLDatasetConfigDataModel(
    dataset_name="bird-validation",
    split="validation",
    hf_corpus_dataset_path="target-benchmark/bird-corpus-validation",
    hf_queries_dataset_path="target-benchmark/bird-queries-validation",
    query_type="Text to SQL",
)

DEFAULT_GITTABLES_DATASET_CONFIG = NeedleInHaystackDatasetConfigDataModel(
    dataset_name="gittables",
    hf_corpus_dataset_path="target-benchmark/gittables-corpus",
    split="train",
)

# TESTING ONLY
DEFAULT_DUMMY_DATASET_CONFIG = HFDatasetConfigDataModel(
    dataset_name="dummy-dataset",
    hf_corpus_dataset_path="jixy2012/mock-hf-corpus-dataset",
    hf_queries_dataset_path="jixy2012/mock-hf-queries-dataset",
    query_type="Table Question Answering",
)


NEEDLE_IN_HAYSTACK_DATASETS = {
    DEFAULT_GITTABLES_DATASET_CONFIG.dataset_name: DEFAULT_GITTABLES_DATASET_CONFIG,
}

FACT_VER_DATASETS = {
    DEFAULT_TABFACT_DATASET_CONFIG.dataset_name: DEFAULT_TABFACT_DATASET_CONFIG,
}

TEXT_2_SQL_DATASETS = {
    DEFAULT_SPIDER_TEST_DATASET_CONFIG.dataset_name: DEFAULT_SPIDER_TEST_DATASET_CONFIG,
    DEFAULT_BIRD_VALIDATION_DATASET_CONFIG.dataset_name: DEFAULT_BIRD_VALIDATION_DATASET_CONFIG,
}

QUESTION_ANSWERING_DATASETS = {
    # this is for testing!!
    DEFAULT_FETAQA_DATASET_CONFIG.dataset_name: DEFAULT_FETAQA_DATASET_CONFIG,
    DEFAULT_OTTQA_DATASET_CONFIG.dataset_name: DEFAULT_OTTQA_DATASET_CONFIG,
}
