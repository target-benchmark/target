import json
import os
import time

from target_benchmark.evaluators.TARGET import TARGET
from target_benchmark.retrievers.analysis.NoContextRetriever import NoContextRetriever
from target_benchmark.retrievers.hyse.HySERetriever import HySERetriever
from target_benchmark.retrievers.naive.HNSWOpenAIEmbeddingRetriever import (
    HNSWOpenAIEmbeddingRetriever,
)
from target_benchmark.retrievers.ottqa.OTTQARetriever import OTTQARetriever


class RetrieverEval:
    def __init__(
        self,
        retriever_name: str,
        retriever,
        tasks_list: list,
        out_dir: str,
        result_appendix: str = None,
        persist_log: str = True,
        log_file_path: str = None,
    ):
        self.retriever_name = retriever_name
        self.result_appendix = result_appendix

        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)

        self.retriever = retriever

        self.target = TARGET(tasks_list)

    def run_eval(self, top_k=10):
        """Runs evaluation and writes results (retrieval, downsream task, total eval time) to a json file."""
        self.target.logger.info(f"starting {self.retriever_name}")
        start = time.time()
        res = self.target.run(
            self.retriever, split="validation", top_k=top_k, batch_size=100, debug=False
        )
        eval_time = time.time() - start

        num_schemas_appendix = ""
        if self.retriever_name == "HySE":
            num_schemas_appendix = f"_numschemas_{self.retriever.num_schemas}_withquery_{self.retriever.with_query}"

        if self.retriever_name in ["OTTQA", "NoContext"]:
            self.retriever.num_rows = ""

        res_dict = {}
        for task_name, task_dict in res.items():
            for dataset_name, dataset_dict in task_dict.items():
                res_dict["performance"] = dataset_dict.model_dump()
                res_dict["eval_time"] = eval_time

                with open(
                    # outdir includes retriever name already
                    os.path.join(
                        self.out_dir,
                        f"eval_{task_name}_{dataset_name}_numrows_{self.retriever.num_rows}{num_schemas_appendix}_{self.result_appendix}.json",
                    ),
                    "w",
                ) as f:
                    json.dump(res_dict, f)


if __name__ == "__main__":
    tasks_list = ["Table Question Answering Task", "Fact Verification Task"]

    model_name = "openai"
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"retrieval_files/hyse/{model_name}"
    )
    for aggregated in [False, True]:
        for with_query in [False, True]:
            for num_rows in [0, 2]:
                for num_schemas in [1, 2, 5]:
                    hyse_retriever = HySERetriever(
                        out_dir,
                        model_name=model_name,
                        num_rows=num_rows,
                        num_schemas=num_schemas,
                        with_query=with_query,
                        aggregated=aggregated,
                    )
                    RetrieverEval(
                        retriever_name="HySE",
                        out_dir=out_dir,
                        retriever=hyse_retriever,
                        tasks_list=tasks_list,
                        result_appendix=f"numrows_{num_rows}_numschemas_{num_schemas}_aggregated_{aggregated}_withquery_{with_query}",
                    ).run_eval()

    model_name = "tapas"
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"retrieval_files/hyse/{model_name}"
    )
    num_rows = 2
    num_schemas = 1
    for aggregated in [False, True]:
        for with_query in [False, True]:
            hyse_retriever = HySERetriever(
                out_dir=out_dir,
                model_name=model_name,
                num_rows=num_rows,
                num_schemas=num_schemas,
                with_query=with_query,
                aggregated=aggregated,
            )
            RetrieverEval(
                retriever_name="HySE",
                out_dir=out_dir,
                retriever=hyse_retriever,
                tasks_list=tasks_list,
                result_appendix=f"numrows_{num_rows}_numschemas_{num_schemas}_aggregated_{aggregated}_withquery_{with_query}",
            ).run_eval()

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "retrieval_files/hnswopenai"
    )
    for num_rows in [0, 2]:
        naive_openai_retriever = HNSWOpenAIEmbeddingRetriever(
            out_dir=out_dir,
            num_rows=num_rows,
        )
        RetrieverEval(
            retriever_name="HNSWOpenAI",
            out_dir=out_dir,
            retriever=naive_openai_retriever,
            tasks_list=tasks_list,
            result_appendix=f"numrows_{num_rows}",
        ).run_eval()

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "retrieval_files/ottqa"
    )
    # OTTQA retriever
    for encoding in ["bm25", "tfidf"]:
        for withtitle in [True, False]:
            ottqa_retriever = OTTQARetriever(
                encoding=encoding, out_dir=out_dir, withtitle=withtitle
            )
            RetrieverEval(
                retriever_name="OTTQA",
                out_dir=out_dir,
                retriever=ottqa_retriever,
                tasks_list=tasks_list,
                result_appendix=f"encoding_{encoding}_withtitle_{withtitle}",
            ).run_eval()

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "retrieval_files/analysis/no-context",
    )
    analysis_nocontext_retriever = NoContextRetriever(out_dir=out_dir)
    RetrieverEval(
        retriever_name="NoContext",
        out_dir=out_dir,
        retriever=analysis_nocontext_retriever,
        tasks_list=tasks_list,
    ).run_eval()
