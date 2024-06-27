import json
import os
import logging
import time
import unittest

from retrievers.hyse.HySERetriever import HySERetriever
from retrievers.naive.OpenAIEmbeddingRetriever import OpenAIEmbeddingRetriever
from retrievers.ottqa.OTTQARetriever import OTTQARetriever
from dataset_loaders.HFDatasetLoader import HFDatasetLoader
from evaluators.TARGET import TARGET
from tasks.QuestionAnsweringTask import QuestionAnsweringTask


class RetrieverEval:


    def __init__(
        self,
        retriever_name: str,
        retriever,
        persist_log: str = True,
        log_file_path: str = None,
    ):
        self.retriever_name = retriever_name
        self.out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"retrieval_files/{retriever_name.lower()}",
        )

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)

        self.retriever = retriever

        self.target = TARGET(["Table Question Answering Task"])


    def run_eval(self):
        """Runs evaluation and writes results (retrieval, downsream task, total eval time) to a json file."""
        self.target.logger.info(f"starting {self.retriever_name}")
        start = time.time()
        res = self.target.run(self.retriever, top_k=10)
        eval_time = time.time() - start

        res_dict = {}
        for task_name, task_dict in res.items():
            for dataset_name, dataset_dict in task_dict.items():
                res_dict["performance"] = dataset_dict.model_dump()
                res_dict["eval_time"] = eval_time

                with open(
                    os.path.join(self.out_dir, f"eval_{self.retriever_name}_{task_name}_{dataset_name}.json"), "w"
                ) as f:
                    json.dump(res_dict, f)


if __name__ == "__main__":
    # TODO: fix creation of dirs
    # TODO: make retrieval class keywords+functionality consistent in terms of paths
    hyse_retriever = HySERetriever(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"retrieval_files/hyse"
        )
    )
    RetrieverEval(retriever_name="HySE", retriever=hyse_retriever).run_eval()
    
    naive_openai_retriever = OpenAIEmbeddingRetriever(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"retrieval_files/openai"
        )
    )
    RetrieverEval(retriever_name="OpenAI", retriever=naive_openai_retriever).run_eval()

    # TODO: eval both bm25 and tfidf retrievers
    ottqa_retriever = OTTQARetriever()
    RetrieverEval(retriever_name="OTTQA", retriever=ottqa_retriever).run_eval()