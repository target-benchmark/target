import json
import os
import time

from target_benchmark.evaluators.TARGET import TARGET
from target_benchmark.retrievers.murre.MurreRetriever import MurreRetriever

RUN_LOCAL_MODELS = False
RUN_OPENAI_ONLY = True

class RetrieverEval:
    def __init__(
        self,
        retriever_name: str,
        retriever,
        tasks_list: list,
        out_dir: str,
        result_appendix: str = None,
    ):
        self.retriever_name = retriever_name
        self.result_appendix = result_appendix

        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)

        self.retriever = retriever

        self.target = TARGET(tasks_list)

    def run_eval(self, top_k=10):
        self.target.logger.info(f"starting {self.retriever_name}")
        start = time.time()
        res = self.target.run(
            self.retriever, split="validation", top_k=top_k, batch_size=1, debug=False, max_samples=5
        )
        eval_time = time.time() - start

        beam_size_appendix = f"_beamsize_{self.retriever.beam_size}"
        max_hops_appendix = f"_maxhops_{self.retriever.max_hops}"
        rewriting_appendix = f"_rewriting_{self.retriever.use_rewriting}"
        
        if hasattr(self.retriever, 'num_rows'):
            num_rows_appendix = f"_numrows_{self.retriever.num_rows}"
        else:
            self.retriever.num_rows = 0
            num_rows_appendix = "_numrows_0"

        res_dict = {}
        for task_name, task_dict in res.items():
            for dataset_name, dataset_dict in task_dict.items():
                res_dict["performance"] = dataset_dict
                res_dict["eval_time"] = eval_time

                filename = f"eval_{task_name}_{dataset_name}{num_rows_appendix}{beam_size_appendix}{max_hops_appendix}{rewriting_appendix}_{self.result_appendix}.json"
                
                with open(os.path.join(self.out_dir, filename), "w") as f:
                    json.dump(res_dict, f)


if __name__ == "__main__":
    tasks_list = [("Table Question Answering Task", "ottqa")]

    if RUN_OPENAI_ONLY:
        print("Running OpenAI Config")
        
        openai_model_name = "openai-text-embedding-3-large"
        openai_out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"retrieval_files/murre/{openai_model_name}"
        )
        
        for beam_size in [1, 3]:
            for max_hops in [1, 2]:
                use_rewriting = False
                model_name = "text-embedding-3-large"
                print(f"Model: {model_name}")
                print(f"Embedding Type: openai")
                print(f"Use Rewriting: {use_rewriting}")
                print(f"Beam Size: {beam_size}")
                print(f"Max Hops: {max_hops}")
                
                try:
                    murre_openai = MurreRetriever(
                        model=model_name,
                        embedding_type="openai",
                        use_rewriting=use_rewriting,
                        beam_size=beam_size,
                        max_hops=max_hops,
                        rewriter_model="gpt-4o-mini"
                    )
                    murre_openai.num_rows = 2
                    
                    RetrieverEval(
                        retriever_name="MURRE",
                        out_dir=openai_out_dir,
                        retriever=murre_openai,
                        tasks_list=tasks_list,
                        result_appendix=f"ottqa_beamsize_{beam_size}_maxhops_{max_hops}_rewriting_{use_rewriting}",
                    ).run_eval()
                    
                except Exception as e:
                    print(f"Error: {e}")
                    continue
    else:
        print("RUN_OPENAI_ONLY = False")

    print("Results in: experiments/retrieval_files/murre/")
