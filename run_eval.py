import json
import os
import time
import unittest

from retrievers.hyse.HySERetriever import HySERetriever
from retrievers.ottqa.OTTQARetriever import OTTQARetriever
from dataset_loaders.HFDatasetLoader import HFDatasetLoader
from evaluators.TARGET import TARGET
from tasks.QuestionAnsweringTask import QuestionAnsweringTask


class RetrieverEval():

    def __init__(self, retriever_name, retriever):
        self.retriever_name = retriever_name
        self.out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"retrieval_files/{retriever_name.lower()}")

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)

        self.retriever = retriever

        self.target = TARGET()

    def run_eval(self):
        start = time.time()
        res = self.target.run(self.retriever, top_k=10)
        eval_time = time.time() - start
        
        res_dict = {}
        for task_name, task_dict in res.items():
            res_dict[task_name] = {}
            for dataset_name, dataset_dict in task_dict.items():
                res_dict[task_name][dataset_name] = dataset_dict.model_dump()
                res_dict[task_name][dataset_name]["eval_time"] = eval_time

        with open(os.path.join(self.out_dir, f"eval_{self.retriever_name}.json"), "w") as f:
            json.dump(res_dict, f)


if __name__ == "__main__":
    # TODO: fix creation of dirs
    # TODO: make retrieval class keywords+functionality consistent in terms of paths
    hyse_retriever = HySERetriever(os.path.join(os.path.dirname(os.path.abspath(__file__)),f"retrieval_files/hyse"))
    RetrieverEval(retriever_name="HySE", retriever=hyse_retriever).run_eval()

    ottqa_retriever = OTTQARetriever(os.path.dirname(os.path.abspath(__file__)))
    RetrieverEval(retriever_name="OTTQA", retriever=ottqa_retriever).run_eval()