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
            self.retriever, split="validation", top_k=top_k, batch_size=100, debug=False
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
                res_dict["performance"] = dataset_dict.model_dump()
                res_dict["eval_time"] = eval_time

                filename = f"eval_{task_name}_{dataset_name}{num_rows_appendix}{beam_size_appendix}{max_hops_appendix}{rewriting_appendix}_{self.result_appendix}.json"
                
                with open(os.path.join(self.out_dir, filename), "w") as f:
                    json.dump(res_dict, f)


if __name__ == "__main__":
    tasks_list = ["Table Question Answering Task", "Fact Verification Task"]

    if RUN_LOCAL_MODELS:
        print("Running Local Model Configurations")
        
        model_name = "qwen3-embedding-0.6b"
        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"retrieval_files/murre/{model_name}"
        )
        
        for use_rewriting in [False, True]:
            for beam_size in [1, 3, 5]:
                for max_hops in [1, 2, 3]:
                    for num_rows in [0, 2]:
                        print(f"Model: Qwen/Qwen3-Embedding-0.6B")
                        print(f"Embedding Type: sentence-transformers")
                        print(f"Use Rewriting: {use_rewriting}")
                        print(f"Beam Size: {beam_size}")
                        print(f"Max Hops: {max_hops}")
                        print(f"Num Rows: {num_rows}")
                        
                        try:
                            murre_retriever = MurreRetriever(
                                model="Qwen/Qwen3-Embedding-0.6B",
                                embedding_type="sentence-transformers",
                                use_rewriting=use_rewriting,
                                beam_size=beam_size,
                                max_hops=max_hops,
                                rewriter_model="gpt-4o-mini"
                            )
                            murre_retriever.num_rows = num_rows
                            
                            RetrieverEval(
                                retriever_name="MURRE",
                                out_dir=out_dir,
                                retriever=murre_retriever,
                                tasks_list=tasks_list,
                                result_appendix=f"qwen3_0.6b_beamsize_{beam_size}_maxhops_{max_hops}_rewriting_{use_rewriting}_numrows_{num_rows}",
                            ).run_eval()
                            
                        except Exception as e:
                            print(f"Error: {e}")
                            continue
    else:
        print("RUN_LOCAL_MODELS = False")

    if RUN_OPENAI_ONLY:
        print("Running OpenAI Config")
        
        openai_model_name = "openai-text-embedding-3-small"
        openai_out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"retrieval_files/murre/{openai_model_name}"
        )
        
        for beam_size in [1, 3]:
            for max_hops in [1, 2]:
                for use_rewriting in [False, True]:
                    print(f"Model: text-embedding-3-small")
                    print(f"Embedding Type: openai")
                    print(f"Use Rewriting: {use_rewriting}")
                    print(f"Beam Size: {beam_size}")
                    print(f"Max Hops: {max_hops}")
                    
                    try:
                        murre_openai = MurreRetriever(
                            model="text-embedding-3-small",
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
                            result_appendix=f"openai_baseline_beamsize_{beam_size}_maxhops_{max_hops}_rewriting_{use_rewriting}",
                        ).run_eval()
                        
                    except Exception as e:
                        print(f"Error: {e}")
                        continue
    else:
        print("RUN_OPENAI_ONLY = False")

    # Custom embedding models 
    if RUN_LOCAL_MODELS:
        print("Running Custom Embedding")
        
        try:
            custom_models = [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2"
            ]
            
            for custom_model in custom_models:
                custom_out_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), f"retrieval_files/murre/{custom_model.replace('/', '_')}"
                )
                
                print(f"{custom_model}")
                
                murre_custom = MurreRetriever(
                    model=custom_model,
                    embedding_type="sentence-transformers",
                    use_rewriting=False,
                    beam_size=3,
                    max_hops=2
                )
                murre_custom.num_rows = 2
                
                RetrieverEval(
                    retriever_name="MURRE",
                    out_dir=custom_out_dir,
                    retriever=murre_custom,
                    tasks_list=tasks_list,
                    result_appendix=f"custom_{custom_model.replace('/', '_')}_comparison",
                ).run_eval()
                
                
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("RUN_LOCAL_MODELS = False")

    print("Results in: experiments/retrieval_files/murre/")
