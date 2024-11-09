from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field


class DatasetConfigDataModel(BaseModel):
    """
    A base data model for dataset configs. DO NOT USE DIRECTLY. use `GenericDatasetConfigDataModel` or `HFDatasetConfigDataModel`.
    Target's dataset configs are written in `TargetDatasetConfig.py`.
    """

    dataset_name: str = Field(description="Name of the dataset. This must be filled out.")
    num_tables: Optional[int] = Field(
        default=None,
        description="Number of tables of the corpus to embed. If the field is left empty, the entire corpus will be embedded, otherwise randomly select `num_tables` tables from the corpus. Intended for needle-in-haystack datasets for incremental corpus sizes evaluations.",
    )
    split: Literal["test", "train", "validation"] = Field(
        default="test",
        description="Split to include. Defaults to only the test split.",
    )
    data_directory: Optional[str] = Field(
        default=None,
        description="directory for where to persist the data to. defaults to None.",
    )

    query_type: Optional[
        Literal["Fact Verification", "Text to SQL", "Table Question Answering", "Needle in Haystack", "Other"]
    ] = Field(default=None, description="Type of query in this dataset.")

    aux: Optional[Dict] = Field(default=None, description="Any additional information related to the dataset.")


class GenericDatasetConfigDataModel(DatasetConfigDataModel):
    """
    Data model for local datasets.
    """

    dataset_path: str = Field(description="Path to the local dataset directory.")
    datafile_ext: str = (Field(default=None, description="File type of the dataset. csv, tsv, etc."),)


class HFDatasetConfigDataModel(DatasetConfigDataModel):
    """
    Data model for huggingface datasets.
    """

    hf_corpus_dataset_path: str = Field(
        description="A huggingface dataset path to the query dataset. It will look something like target-benchmark/fetaqa-corpus (namespace/corpus-dataset-name)"
    )

    hf_queries_dataset_path: str = Field(
        description="A huggingface dataset path to the query dataset. It will look something like target-benchmark/fetaqa-queries (namespace/queries-dataset-name)"
    )


class NeedleInHaystackDatasetConfigDataModel(HFDatasetConfigDataModel):
    query_type: str = "Needle in Haystack"
    hf_queries_dataset_path: str = ""


class Text2SQLDatasetConfigDataModel(HFDatasetConfigDataModel):
    query_type: str = "Text to SQL"
