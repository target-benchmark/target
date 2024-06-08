import unittest
from dataset_loaders import HFDatasetLoader
from dataset_loaders.TargetDatasetConfig import DEFAULT_SPIDER_DATASET_CONFIG, DEFAULT_TABFACT_DATASET_CONFIG
from dataset_loaders.AbsDatasetLoader import QueryType
class DataloadersTest(unittest.TestCase):
    def test_text2sql(self):
        spider_loader = HFDatasetLoader(**DEFAULT_SPIDER_DATASET_CONFIG.model_dump())
        self.assertEqual(spider_loader.query_type, QueryType.TEXT_2_SQL)
        self.assertEqual(spider_loader.dataset_name, "spider")
        print(spider_loader.dataset_name)

        spider_loader.load()
        self.assertIsNotNone(spider_loader.alt_corpus)
        self.assertEqual(len(spider_loader.alt_corpus["test"]), len(spider_loader.corpus["test"]))
    
    def test_factver(self):
        tabfact_loader = HFDatasetLoader(**DEFAULT_TABFACT_DATASET_CONFIG.model_dump())
        self.assertEqual(tabfact_loader.query_type, QueryType.FACT_VERIFICATION)
        tabfact_loader.load()
        self.assertIsNone(tabfact_loader.alt_corpus)



if __name__ == "__main__":
    unittest.main()