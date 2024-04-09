from dataset_loaders.AbsTargetDatasetLoader import AbsTargetDatasetLoader
from pathlib import Path
from datasets import load_dataset, DatasetDict
from dictionary_keys import *
class HFDatasetLoader(AbsTargetDatasetLoader):
    def __init__(
            self,
            dataset_name: str,
            hf_corpus_dataset_path: str,
            hf_queries_dataset_path: str,
            table_col_name: str = TABLE_COL_NAME,
            table_id_col_name: str = TABLE_ID_COL_NAME,
            database_id_col_name: str = DATABASE_ID_COL_NAME,
            query_col_name: str = QUERY_COL_NAME,
            query_id_col_name: str = QUERY_ID_COL_NAME,
            answer_col_name: str = ANSWER_COL_NAME,            
            splits: str | list[str] = "test",
            data_directory: str = None,
            query_type: str = "",
            **kwargs
        ):
        super().__init__(
            dataset_name=dataset_name, 
            table_col_name=table_col_name,
            table_id_col_name=table_id_col_name,
            database_id_col_name=database_id_col_name,
            query_col_name=query_col_name,
            query_id_col_name=query_id_col_name,
            answer_col_name=answer_col_name,            
            splits=splits,
            data_directory=data_directory,
            query_type=query_type,
            **kwargs
        )
        '''
        Constructor for a generic dataset loader that loads from a huggingface dataset.
        Parameters:
            hf_corpus_dataset_path (str): the path to your huggingface hub corpus dataset. it will look something like target-benchmark/fetaqa-corpus (namespace/dataset-name)
            hf_queries_dataset_path (str): the path to your huggingface hub queries dataset path. 
        '''

        self.hf_corpus_dataset_path = hf_corpus_dataset_path
        self.hf_queries_dataset_path = hf_queries_dataset_path


    def load(self, splits: str | list[str] = None) -> None:
        '''
        Load specific splits of a dataset, such as 'train', 'test', or 'validation'. It can accept either a single split as a string or a list of splits.

        Parameters:
            split (str | list[str], optional): The dataset split or splits to load. Defaults to None, which will load train split or the split specified when constructing this Generic Dataset Loader object
        '''
        if splits:
            if isinstance(splits, str):
                splits = [splits]
            for split in splits: 
                if split not in self.splits:
                    self.splits.append(split)
        self._load_corpus()
        self._load_queries()
        

    def _load_corpus(self) -> None:
        if not self.corpus:
            self.corpus = DatasetDict()
        for split in self.splits:
            if split not in self.corpus:
                self.corpus[split] = load_dataset(path=self.hf_corpus_dataset_path, split=split)            

    def _load_queries(self) -> None:
        if not self.queries:
            self.queries = DatasetDict()
        for split in self.splits:
            if split not in self.queries:
                self.queries[split] = load_dataset(path=self.hf_queries_dataset_path, split=split)

