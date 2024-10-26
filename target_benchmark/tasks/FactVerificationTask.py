from typing import Dict, List, Tuple

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from target_benchmark.dataset_loaders.LoadersDataModels import DatasetConfigDataModel
from target_benchmark.dataset_loaders.TargetDatasetConfig import FACT_VER_DATASETS
from target_benchmark.dictionary_keys import (
    ANSWER_COL_NAME,
    QUERY_COL_NAME,
    QUERY_ID_COL_NAME,
)
from target_benchmark.generators.AbsGenerator import AbsGenerator
from target_benchmark.generators.DefaultGenerator import DefaultGenerator
from target_benchmark.generators.GeneratorPrompts import (
    FACT_VER_SYSTEM_PROMPT,
    FACT_VER_USER_PROMPT,
)
from target_benchmark.generators.GeneratorsDataModels import (
    DownstreamGeneratedResultDataModel,
)
from target_benchmark.retrievers.RetrieversDataModels import RetrievalResultDataModel
from target_benchmark.tasks.AbsTask import AbsTask
from target_benchmark.tasks.TasksDataModels import (
    FactVerificationTaskPerformanceDataModel,
)
from target_benchmark.tasks.utils import build_table_content_string


class FactVerificationTask(AbsTask):
    AVAILABLE_METRICS = ["accuracy", "f1", "precision", "recall"]

    def __init__(
        self,
        datasets_config: Dict[str, Dict[str, str]] = None,
        task_generator: AbsGenerator = None,
        **kwargs,
    ):
        if task_generator is None:
            task_generator = DefaultGenerator(
                system_message=FACT_VER_SYSTEM_PROMPT,
                user_message=FACT_VER_USER_PROMPT,
            )
        super().__init__(
            task_name=self.get_default_task_name(),
            datasets_config=datasets_config,
            task_generator=task_generator,
            **kwargs,
        )

        # two lists, pred_answers contains the predicted answers, and ref_answers contains the ground truth answers.
        # True = 1, False = 0, Not Enough information = -1.
        self.pred_answers = []
        self.ref_answers = []

    @classmethod
    def get_default_task_name(cls) -> str:
        return "Fact Verification Task"

    @classmethod
    def get_available_metrics(cls) -> str:
        return str(FactVerificationTask.AVAILABLE_METRICS)

    @classmethod
    def _get_default_dataset_config(cls) -> Dict[str, DatasetConfigDataModel]:
        """
        Returns the default dataset config for fact verification.
        Includes the following datasets:
            TabFact
            TODO: more to come
        """
        return dict(FACT_VER_DATASETS)

    def _get_downstream_task_results(
        self,
        query_batch: Dict[str, List],
        retrieval_results: List[RetrievalResultDataModel],
        dataset_name: str,
        table_id_to_table: Dict[Tuple[str, str], List[List]],
    ) -> List[DownstreamGeneratedResultDataModel]:
        """
        Given the query and the retrieval results, generate downstream task results. Uses fact verification tasks's default generator to accept or refute the claim, or say there's not enough information.
        """
        return [
            DownstreamGeneratedResultDataModel(
                dataset_name=dataset_name,
                query_id=query_id,
                generated_results=self.task_generator.generate(
                    table_str=build_table_content_string(
                        result.retrieval_results,
                        table_id_to_table,
                    ),
                    query=query_str,
                ),
            )
            for query_id, query_str, result in zip(
                query_batch[QUERY_ID_COL_NAME],
                query_batch[QUERY_COL_NAME],
                retrieval_results,
            )
        ]

    def _update_downstream_task_metrics(
        self,
        query_batch: Dict[str, List],
        downstream_results: List[DownstreamGeneratedResultDataModel],
    ) -> None:
        """
        Update metric tracked for fact verification's performance calculation.
        Specifically, update the `self.pred_answers` and `self.ref_answers` lists
        based on the predicted answers in downstream_results and ground truth answers in query_batch.
        """
        # for downstream_answer, query_answer in zip(downstream_results, query_batch[ANSWER_COL_NAME]):
        #     if "true" in downstream_answer.generated_results.lower():
        #         self.pred_answers.append(1)
        #     elif "false" in downstream_answer.generated_results.lower():
        #         self.pred_answers.append(0)
        #     else:
        #         # self.pred_answers.append(1 - self.ref_answers[-1])
        #         continue
        #     if "true" in query_answer.lower():
        #         self.ref_answers.append(1)
        #     else:
        #         self.ref_answers.append(0)
        for downstream_answer in downstream_results:
            if "true" in downstream_answer.generated_results.lower():
                self.pred_answers.append(1)
            elif "false" in downstream_answer.generated_results.lower():
                self.pred_answers.append(0)
            else:
                self.pred_answers.append(-1)
        for query_answer in query_batch[ANSWER_COL_NAME]:
            if "true" in query_answer.lower():
                self.ref_answers.append(1)
            elif "false" in query_answer.lower():
                self.ref_answers.append(0)
            else:
                self.ref_answers.append(-1)

    def _calculate_downstream_task_performance(
        self, **kwargs
    ) -> FactVerificationTaskPerformanceDataModel:
        """
        Calculate downstream task metrics for the fact verification task.
        Metrics computed: accuracy, f1, precision, and recall.
        """
        assert len(self.ref_answers) == len(self.pred_answers)
        accuracy = accuracy_score(self.ref_answers, self.pred_answers)
        precision, recall, fbeta, _ = precision_recall_fscore_support(
            self.ref_answers, self.pred_answers, average="weighted"
        )

        result = FactVerificationTaskPerformanceDataModel(
            scores={
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": fbeta,
            }
        )

        self.pred_answers = []
        self.ref_answers = []
        return result
