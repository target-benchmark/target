from typing import Dict, List, Tuple, Union

import evaluate

from target_benchmark.dataset_loaders.LoadersDataModels import DatasetConfigDataModel
from target_benchmark.dataset_loaders.TargetDatasetConfig import (
    QUESTION_ANSWERING_DATASETS,
)
from target_benchmark.dictionary_keys import (
    ANSWER_COL_NAME,
    QUERY_COL_NAME,
    QUERY_ID_COL_NAME,
)
from target_benchmark.generators.AbsGenerator import AbsGenerator
from target_benchmark.generators.DefaultGenerator import DefaultGenerator
from target_benchmark.generators.GeneratorPrompts import (
    QA_SYSTEM_PROMPT,
    QA_USER_PROMPT,
)
from target_benchmark.generators.GeneratorsDataModels import (
    DownstreamGeneratedResultDataModel,
)
from target_benchmark.retrievers.RetrieversDataModels import RetrievalResultDataModel
from target_benchmark.tasks.AbsTask import AbsTask
from target_benchmark.tasks.TasksDataModels import TableQATaskPerformanceDataModel
from target_benchmark.tasks.utils import build_table_content_string


class QuestionAnsweringTask(AbsTask):
    AVAILABLE_METRICS = set(
        ["bertscore", "bleu", "bleurt", "sacrebleu", "rouge", "meteor"]
    )
    DEFAULT_METRICS = set(["bleu", "sacrebleu", "rouge"])

    def __init__(
        self,
        datasets_config: Union[
            Dict[str, Union[Dict[str, str], DatasetConfigDataModel]], None
        ] = None,
        task_generator: AbsGenerator = None,
        lang: str = "en",
        metrics: Union[str, List[str]] = list(DEFAULT_METRICS),
        **kwargs,
    ):
        if not task_generator:
            task_generator = DefaultGenerator(QA_SYSTEM_PROMPT, QA_USER_PROMPT)
        super().__init__(
            task_name=self.get_default_task_name(),
            datasets_config=datasets_config,
            task_generator=task_generator,
            **kwargs,
        )
        # set up the evaluator objects
        if isinstance(metrics, str):
            metrics = [metrics]

        self.evals = {}
        for metric in metrics:
            if metric not in QuestionAnsweringTask.AVAILABLE_METRICS:
                raise ValueError(
                    f"the metric {metric} is not one of the available metrics!"
                )
            self.evals[metric] = evaluate.load(metric)

        self.language = lang
        self.pred_answers = []
        self.ref_answers = []

    @classmethod
    def get_default_task_name(cls) -> str:
        return "Table Question Answering Task"

    @classmethod
    def get_available_metrics(cls) -> str:
        return str(cls.AVAILABLE_METRICS)

    @classmethod
    def _get_default_dataset_config(cls) -> Dict[str, DatasetConfigDataModel]:
        """
        Returns the default dataset config for question answering.
        Includes the following datasets:
            FeTaQA
            OTTQA
            TODO: more to come
        """
        return dict(QUESTION_ANSWERING_DATASETS)

    def _get_downstream_task_results(
        self,
        query_batch: Dict[str, List],
        retrieval_results: List[RetrievalResultDataModel],
        dataset_name: str,
        table_id_to_table: Dict[Tuple[str, str], List[List]],
    ) -> List[DownstreamGeneratedResultDataModel]:
        """
        currently just markdown reps of table strings
        All downstreams tasks should fill out this method. ideally uses the retrieval results to generate the downstream answer, and return the performance of the downstream generation.
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
        Update any values you keep track of for the downstream tasks.
        """
        self.pred_answers.extend(
            [
                downstream_answer.generated_results
                for downstream_answer in downstream_results
            ]
        )
        self.ref_answers.extend(
            [query_answer for query_answer in query_batch[ANSWER_COL_NAME]]
        )
        # for downstream_answer, query_answer in zip(downstream_results, query_batch[ANSWER_COL_NAME]):
        #     generated_result = str(downstream_answer.generated_results).lower()
        #     if "not enough information" in generated_result:
        #         continue
        #     self.pred_answers.append(downstream_answer.generated_results)
        #     self.ref_answers.append(query_answer)

    def _calculate_downstream_task_performance(
        self, **kwargs
    ) -> TableQATaskPerformanceDataModel:
        """
        Calculate downstream task metrics for the question answering task.
        """
        scores = {}
        for metric_name, evaluator in self.evals.items():
            calculated_result = None
            if metric_name == "bertscore":
                calculated_result = evaluator.compute(
                    predictions=self.pred_answers,
                    references=self.ref_answers,
                    lang="en",
                )
            else:
                calculated_result = evaluator.compute(
                    predictions=self.pred_answers, references=self.ref_answers
                )
            scores[metric_name] = calculated_result

        result = TableQATaskPerformanceDataModel(scores=scores)

        self.pred_answers = []
        self.ref_answers = []
        return result
