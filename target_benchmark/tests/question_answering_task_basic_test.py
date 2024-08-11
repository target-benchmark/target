import unittest
from unittest.mock import MagicMock
from target_benchmark.tasks import QuestionAnsweringTask
from target_benchmark.tasks.TasksDataModels import *
from target_benchmark.retrievers.AbsCustomEmbeddingRetriever import (
    AbsCustomEmbeddingRetriever as CustomEmbRetr,
)
from target_benchmark.retrievers.RetrieversDataModels import RetrievalResultDataModel

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("bert_score").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)

# Get a logger
logger = logging.getLogger(__name__)


class TestTableRetriever(unittest.TestCase):

    def setUp(self):
        self.qa_task = QuestionAnsweringTask()
        self.mock_retriever = MagicMock()

        self.mock_retriever.__class__ = CustomEmbRetr
        self.mock_retriever.retrieve_batch.return_value = [
            RetrievalResultDataModel(
                dataset_name="fetaqa",
                query_id=1,
                retrieval_results=[(0, "Table1"), (0, "Table2")],
            ),
            RetrievalResultDataModel(
                dataset_name="fetaqa",
                query_id=2,
                retrieval_results=[(0, "Table3"), (0, "Table4")],
            ),
        ]
        self.mock_dataset_loader = MagicMock()
        self.mock_dataset_loader.get_table_id_to_table.return_value = {
            (0, "Table1"): [["some random table"], ["some item"]],
            (0, "Table2"): [["some other random table"], ["another item"]],
            (0, "Table3"): [["third random table"], ["third item"]],
            (0, "Table4"): [["fourth random table"], ["fourth item"]],
            (0, "Table5"): [["fifth random table"], ["fifth item"]],
        }
        self.mock_dataset_loader.get_queries_for_task.side_effect = lambda batch_size: iter(
            [
                {
                    "query_id": [1, 2],
                    "query": ["Test query", "Test query 2"],
                    "answer": [
                        "I'm sorry, but I can't provide the information you're looking for because you didn't provide any tables or specific questions. Could you please provide more details?",
                        "I'm sorry, but I can't provide the information you're looking for because you didn't provide any tables or specific questions. Could you please provide more details?",
                    ],
                    "table_id": ["Table1", "Table5"],
                    "database_id": [0, 0],
                }
            ],
        )

    def test_basic_qa_task_run(self):
        results = self.qa_task.task_run(
            retriever=self.mock_retriever,
            dataset_loaders={"fetaqa": self.mock_dataset_loader},
            logger=logger,
            batch_size=1,
            top_k=2,
        )
        res = results["fetaqa"]
        self.assertIn("bleu", res.downstream_task_performance.scores)
        self.assertIn("sacrebleu", res.downstream_task_performance.scores)
        self.assertIn("rouge", res.downstream_task_performance.scores)
        print(res)

    def test_specified_metrics(self):
        qa_task = QuestionAnsweringTask(metrics=["sacrebleu", "rouge"])
        results = qa_task.task_run(
            retriever=self.mock_retriever,
            dataset_loaders={"fetaqa": self.mock_dataset_loader},
            logger=logger,
            batch_size=1,
            top_k=2,
        )
        res = results["fetaqa"]

        # self.assertIn("bertscore", res.downstream_task_performance.scores)
        # self.assertIn("precision", res.downstream_task_performance.scores["bertscore"])
        # self.assertIn("recall", res.downstream_task_performance.scores["bertscore"])
        # self.assertIn("f1", res.downstream_task_performance.scores["bertscore"])
        # self.assertIn("hashcode", res.downstream_task_performance.scores["bertscore"])

        # self.assertIn("bleurt", res.downstream_task_performance.scores)
        # self.assertIn("scores", res.downstream_task_performance.scores["bleurt"])

        # self.assertIn("meteor", res.downstream_task_performance.scores)
        # self.assertIn("meteor", res.downstream_task_performance.scores["meteor"])

        self.assertIn("sacrebleu", res.downstream_task_performance.scores)
        self.assertIn("score", res.downstream_task_performance.scores["sacrebleu"])
        self.assertIn("counts", res.downstream_task_performance.scores["sacrebleu"])
        self.assertIn("totals", res.downstream_task_performance.scores["sacrebleu"])
        self.assertIn("precisions", res.downstream_task_performance.scores["sacrebleu"])
        self.assertIn("bp", res.downstream_task_performance.scores["sacrebleu"])
        self.assertIn("sys_len", res.downstream_task_performance.scores["sacrebleu"])
        self.assertIn("ref_len", res.downstream_task_performance.scores["sacrebleu"])

        self.assertIn("rouge", res.downstream_task_performance.scores)
        self.assertIn("rouge1", res.downstream_task_performance.scores["rouge"])
        self.assertIn("rouge2", res.downstream_task_performance.scores["rouge"])
        self.assertIn("rougeL", res.downstream_task_performance.scores["rouge"])
        self.assertIn("rougeLsum", res.downstream_task_performance.scores["rouge"])
        print(res)


if __name__ == "__main__":
    unittest.main()
