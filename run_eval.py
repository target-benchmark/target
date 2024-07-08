import json
import os
import logging
import time
import unittest

from retrievers.hyse.HySERetriever import HySERetriever
from retrievers.naive.HNSWOpenAIEmbeddingRetriever import HNSWOpenAIEmbeddingRetriever
from retrievers.ottqa.OTTQARetriever import OTTQARetriever
from dataset_loaders.HFDatasetLoader import HFDatasetLoader
from evaluators.TARGET import TARGET
from tasks.QuestionAnsweringTask import QuestionAnsweringTask


class RetrieverEval:

    def __init__(
        self,
        retriever_name: str,
        retriever,
        tasks_list: list,
        result_appendix: str = None,
        persist_log: str = True,
        log_file_path: str = None,
    ):
        self.retriever_name = retriever_name
        self.result_appendix = result_appendix

        self.out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"retrieval_files/{retriever_name.lower()}",
        )
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)

        self.retriever = retriever

        self.target = TARGET(tasks_list)


    def run_eval(self, top_k=10):
        """Runs evaluation and writes results (retrieval, downsream task, total eval time) to a json file."""
        self.target.logger.info(f"starting {self.retriever_name}")
        start = time.time()
        res = self.target.run(self.retriever, split="validation", top_k=top_k)
        eval_time = time.time() - start

        num_schemas_appendix = ""
        if self.retriever_name == "HySE":
            num_schemas_appendix = f"_numschemas_{self.retriever.num_schemas}_withquery_{self.retriever.with_query}"

        if self.retriever_name == "OTTQA":
            self.retriever.num_rows = ""

        res_dict = {}
        for task_name, task_dict in res.items():
            for dataset_name, dataset_dict in task_dict.items():
                res_dict["performance"] = dataset_dict.model_dump()
                res_dict["eval_time"] = eval_time

                with open(
                    # outdir includes retriever name already
                    os.path.join(self.out_dir, f"eval_{task_name}_{dataset_name}_numrows_{self.retriever.num_rows}{num_schemas_appendix}_{self.result_appendix}.json"), "w"
                ) as f:
                    json.dump(res_dict, f)


if __name__ == "__main__":
    # TODO: fix creation of dirs
    # TODO: make retrieval class keywords+functionality consistent in terms of paths

    tasks_list = ["Table Question Answering Task", "Fact Verification Task"]

    for with_query in [False, True]:
        for num_schemas in [1, 2, 5]:
            hyse_retriever = HySERetriever(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), f"retrieval_files/hyse"
                ),
                num_rows=2,
                num_schemas=num_schemas,
                with_query=with_query,
            )
            RetrieverEval(retriever_name="HySE", retriever=hyse_retriever, tasks_list=tasks_list, result_appendix="agg").run_eval()
        
    # for num_rows in [0, 1, 2, 5]:
    naive_openai_retriever = HNSWOpenAIEmbeddingRetriever(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"retrieval_files/hnswopenai"
        ),
        num_rows=2,
    )
    RetrieverEval(retriever_name="HNSWOpenAI", retriever=naive_openai_retriever, tasks_list=tasks_list).run_eval()

    # OTTQA BM25 retriever
    ottqa_retriever = OTTQARetriever(encoding="bm25")
    RetrieverEval(retriever_name="OTTQA", retriever=ottqa_retriever, tasks_list=tasks_list, result_appendix="bm25").run_eval()

    ottqa_retriever = OTTQARetriever(encoding="tfidf")
    RetrieverEval(retriever_name="OTTQA", retriever=ottqa_retriever, tasks_list=tasks_list, result_appendix="tfidf").run_eval()