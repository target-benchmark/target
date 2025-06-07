from target_benchmark.dataset_loaders.HFDatasetLoader import HFDatasetLoader
from target_benchmark.dataset_loaders.LoadersDataModels import (
    DatasetConfigDataModel,
    HFDatasetConfigDataModel,
    NeedleInHaystackDatasetConfigDataModel,
    Text2SQLDatasetConfigDataModel,
)
from target_benchmark.dataset_loaders.Text2SQLDatasetLoader import Text2SQLDatasetLoader

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

DEFAULT_SPIDER_DATASET_CONFIG = Text2SQLDatasetConfigDataModel(
    dataset_name="spider",
    hf_corpus_dataset_path="target-benchmark/spider-corpus",
    hf_queries_dataset_path="target-benchmark/spider-queries",
    query_type="Text to SQL",
    split="test",
)


DEFAULT_BIRD_VALIDATION_DATASET_CONFIG = Text2SQLDatasetConfigDataModel(
    dataset_name="bird-validation",
    hf_corpus_dataset_path="target-benchmark/bird-corpus-validation",
    hf_queries_dataset_path="target-benchmark/bird-queries-validation",
    query_type="Text to SQL",
    split="validation",
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
    DEFAULT_SPIDER_DATASET_CONFIG.dataset_name: DEFAULT_SPIDER_DATASET_CONFIG,
    DEFAULT_BIRD_VALIDATION_DATASET_CONFIG.dataset_name: DEFAULT_BIRD_VALIDATION_DATASET_CONFIG,
}

QUESTION_ANSWERING_DATASETS = {
    DEFAULT_FETAQA_DATASET_CONFIG.dataset_name: DEFAULT_FETAQA_DATASET_CONFIG,
    DEFAULT_OTTQA_DATASET_CONFIG.dataset_name: DEFAULT_OTTQA_DATASET_CONFIG,
}

TABLE_RETRIEVAL_DATASETS = FACT_VER_DATASETS | TEXT_2_SQL_DATASETS | QUESTION_ANSWERING_DATASETS

ALL_DATASETS = TABLE_RETRIEVAL_DATASETS | NEEDLE_IN_HAYSTACK_DATASETS


def get_default_dataset(config: DatasetConfigDataModel) -> HFDatasetLoader | Text2SQLDatasetLoader:
    config = config.model_copy(deep=True)
    if config.dataset_name in TEXT_2_SQL_DATASETS:
        return Text2SQLDatasetLoader(**config.model_dump())
    elif config.dataset_name in ALL_DATASETS:
        return HFDatasetLoader(**config.model_dump())
    else:
        raise ValueError(f"{config.dataset_name} is not a TARGET default dataset.")


def get_default_bird_dataset() -> Text2SQLDatasetLoader:
    return get_default_dataset(DEFAULT_BIRD_VALIDATION_DATASET_CONFIG)


def get_default_spider_dataset() -> Text2SQLDatasetLoader:
    return get_default_dataset(DEFAULT_SPIDER_DATASET_CONFIG)


def get_default_tabfact_dataset() -> HFDatasetLoader:
    return get_default_dataset(DEFAULT_TABFACT_DATASET_CONFIG)


def get_default_fetaqa_dataset() -> HFDatasetLoader:
    return get_default_dataset(DEFAULT_FETAQA_DATASET_CONFIG)


def get_default_ottqa_dataset() -> HFDatasetLoader:
    return get_default_dataset(DEFAULT_OTTQA_DATASET_CONFIG)


def get_default_gittables_dataset() -> HFDatasetLoader:
    return get_default_dataset(DEFAULT_GITTABLES_DATASET_CONFIG)


def get_default_infiagentda_dataset() -> HFDatasetLoader:
    return get_default_dataset(DEFAULT_INFAGENTDA_DATASET_CONFIG)
