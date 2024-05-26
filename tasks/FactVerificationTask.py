from dataset_loaders.LoadersDataModels import (
    DatasetConfigDataModel,
    QueryForTasksDataModel,
)
from dataset_loaders.TargetDatasetConfig import *

from generators.AbsGenerator import AbsGenerator
from generators.DefaultGenerator import DefaultGenerator
from generators.GeneratorPrompts import DEFAULT_FACT_VER_SYSTEM_MESSAGE
from generators.GeneratorsDataModels import DownstreamGeneratedResultDataModel

from retrievers.RetrieversDataModels import RetrievalResultDataModel

from tasks.AbsTask import AbsTask
from tasks.TasksDataModels import (
    FactVerificationTaskPerformanceDataModel,
)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from typing import List, Dict


class FactVerificationTask(AbsTask):
    AVAILABLE_METRICS = ["accuracy", "f1", "precision", "recall"]

    def __init__(
        self,
        datasets_config: Dict[str, Dict[str, str]] = None,
        overwrite_default_datasets: bool = False,
        task_generator: AbsGenerator = None,
        **kwargs,
    ):
        if task_generator == None:
            task_generator = DefaultGenerator(
                system_message=DEFAULT_FACT_VER_SYSTEM_MESSAGE
            )
        super().__init__(
            task_name=self.get_default_task_name(),
            datasets_config=datasets_config,
            overwrite_default_datasets=overwrite_default_datasets,
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

    def _get_default_dataset_config(self) -> Dict[str, DatasetConfigDataModel]:
        """
        Returns the default dataset config for fact verification.
        Includes the following datasets:
            TabFact
            TODO: more to come
        """
        # TODO: add more things here. this is for testing. carl note 4/24
        return {
            # this is for testing!!
            DEFAULT_DUMMY_DATASET_CONFIG.dataset_name: DEFAULT_DUMMY_DATASET_CONFIG,
        }

    def _get_downstream_task_results(
        self,
        query_batch: List[QueryForTasksDataModel],
        retrieval_results: List[RetrievalResultDataModel],
        dataset_name: str,
    ) -> List[DownstreamGeneratedResultDataModel]:
        """
        Given the query and the retrieval results, generate downstream task results. Uses fact verification tasks's default generator to accept or refute the claim, or say there's not enough information.
        """
        return [
            DownstreamGeneratedResultDataModel(
                dataset_name=dataset_name,
                query_id=query.query_id,
                generated_results=self.task_generator.generate(
                    table_str="\n".join(
                        table_str for table_str in result.retrieved_tables
                    ),
                    query=query.query,
                ),
            )
            for query, result in zip(query_batch, retrieval_results)
        ]

    def _update_downstream_task_metrics(
        self,
        query_batch: List[QueryForTasksDataModel],
        downstream_results: List[DownstreamGeneratedResultDataModel],
    ) -> None:
        """
        Update metric tracked for fact verification's performance calculation.
        Specifically, update the `self.pred_answers` and `self.ref_answers` lists
        based on the predicted answers in downstream_results and ground truth answers in query_batch.
        """
        for downstream_answer in downstream_results:
            if "true" in downstream_answer.generated_results.lower():
                self.pred_answers.append(1)
            elif "false" in downstream_answer.generated_results.lower():
                self.pred_answers.append(0)
            else:
                self.pred_answers.append(-1)
        for query in query_batch:
            if "true" in query.answer.lower():
                self.ref_answers.append(1)
            elif "false" in query.answer.lower():
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
