import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Union

from target_benchmark.dataset_loaders.LoadersDataModels import DatasetConfigDataModel
from target_benchmark.dataset_loaders.TargetDatasetConfig import TEXT_2_SQL_DATASETS
from target_benchmark.dataset_loaders.Text2SQLDatasetLoader import Text2SQLDatasetLoader
from target_benchmark.dictionary_keys import (
    ANSWER_COL_NAME,
    DATABASE_ID_COL_NAME,
    DATASET_NAME,
    DIFFICULTY_COL_NAME,
    QUERY_COL_NAME,
    QUERY_ID_COL_NAME,
)
from target_benchmark.generators import AbsGenerator, Text2SQLGenerator
from target_benchmark.generators.GeneratorPrompts import NO_CONTEXT_TABLE_PROMPT
from target_benchmark.generators.GeneratorsDataModels import (
    DownstreamGeneratedResultDataModel,
)
from target_benchmark.retrievers.RetrieversDataModels import RetrievalResultDataModel
from target_benchmark.tasks.AbsTask import AbsTask
from target_benchmark.tasks.TasksDataModels import Text2SQLTaskPerformanceDataModel
from target_benchmark.tasks.utils import evaluate_sql_execution


class Text2SQLTask(AbsTask):
    AVAILABLE_METRICS = set(["execution_accuracy", "execution_ves"])
    DEFAULT_METRICS = set(["execution_accuracy"])

    def __init__(
        self,
        datasets_config: Dict[str, Dict[str, str]] = None,
        task_generator: AbsGenerator = None,
        metrics: Union[str, List[str]] = list(DEFAULT_METRICS),
        **kwargs,
    ):
        if task_generator is None:
            task_generator = Text2SQLGenerator()
        super().__init__(
            task_name=self.get_default_task_name(),
            datasets_config=datasets_config,
            task_generator=task_generator,
            **kwargs,
        )

        if isinstance(metrics, str):
            metrics = [metrics]

        for metric in metrics:
            if metric not in Text2SQLTask.AVAILABLE_METRICS:
                raise ValueError(f"the metric {metric} is not one of the available metrics!")
        self.include_ves = False
        if "execution_ves" in metrics:
            self.include_ves = True

        # two lists, pred_sql contains the predicted sql queries,
        # and ref_sql contains the ground truth sql queries.
        self.pred_sql = []
        self.ref_sql = []
        self.difficulties = []
        self.current_dataset: str = None
        self.database_dirs: Dict[str, str] = None

    @classmethod
    def get_default_task_name(cls) -> str:
        return "Text to SQL Task"

    @classmethod
    def get_available_metrics(cls) -> str:
        return str(Text2SQLTask.AVAILABLE_METRICS)

    def setup_database_dirs(self, dataloaders: Dict[str, Text2SQLDatasetLoader]):
        self.database_dirs = {name: loader.path_to_database_dir for name, loader in dataloaders.items()}

    @classmethod
    def _get_default_dataset_config(cls) -> Dict[str, DatasetConfigDataModel]:
        """
        Returns the default dataset config for text 2 sql.
        Includes the following datasets:
            BIRD
            Spider
            TODO: more to come
        """
        return dict(TEXT_2_SQL_DATASETS)

    def _get_schema(self, dataset_name: str, db_id: str, table_ids: List[str]) -> str:
        """Writes the `CREATE TABLE` statements for specified tables to a formatted string.

        Example:
            Database Name: conference
            Schema:

            CREATE TABLE "conference" (
                "Conference_ID" int,
                "Conference_Name" text,
                "Year" int,
                "Location" text,
                PRIMARY KEY ("Conference_ID")
            )

            CREATE TABLE institution (
                "Institution_ID" int,
                "Institution_Name" text,
                "Location" text,
                "Founded" int,
                PRIMARY KEY ("Institution_ID")
            )

            CREATE TABLE "staff" (
                "staff_ID" int,
                "name" text,
                "Age" int,
                "Nationality" text,
                "Institution_ID" int,
                PRIMARY KEY ("staff_ID"),
                FOREIGN KEY (`Institution_ID`) REFERENCES `institution`(`Institution_ID`)
            )

            ---

            Database Name: video_game
            Schema:

            CREATE TABLE "player" (
                "Player_ID" int,
                "Rank_of_the_year" int,
                "Player_name" text,
                "Position" text,
                "College" text,
                PRIMARY KEY ("Player_ID")
            )

            ---

            Database Name: bbc_channels
            Schema:

            CREATE TABLE "director" (
                "Director_ID" int,
                "Name" text,
                "Age" int,
                PRIMARY KEY ("Director_ID")
            )

            ---
        """
        if dataset_name not in self.database_dirs:
            raise ValueError(f"dataset {dataset_name} does not have a database directory setup.")
        if db_id == "":
            return NO_CONTEXT_TABLE_PROMPT
        db_path = Path(self.database_dirs[dataset_name], db_id, f"{db_id}.sqlite")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        # Fetch and print the schema of each table
        table_schemas = cur.execute(
            f"SELECT name, sql FROM sqlite_schema WHERE type='table' AND name IN ({','.join('?' * len(table_ids))})", table_ids
        ).fetchall()
        schema_str = f"\nDatabase Name: {db_id}\nSchema:"

        def normalize_schema(schema: str) -> str:
            # Make sure our commas only come after newlines - not the inverse
            schema = re.sub(r"\n,", r",\n", schema)
            # Remove any indentation after the newlines
            schema = re.sub(r"\n\s+(?=[^\s])", r"\n", schema)
            # Make sure we have tabs after each newline - just not the last one (with a r-paren after it)
            schema = re.sub(r"\n(?=[^\)])", r"\n\t", schema)
            return schema

        for _, schema in table_schemas:
            schema_str += f"\n\n{normalize_schema(schema)}"
        return schema_str + "\n\n---\n"

    def _preprocess_table(
        self,
        result: RetrievalResultDataModel,
        table_id_to_table: Dict[Tuple[str, str], List[List]],
    ):
        return "\n".join(self._get_schema(self.current_dataset, id[0]) for id in result.retrieval_results)

    def _get_downstream_task_results(
        self,
        query_batch: Dict[str, List],
        retrieval_results: List[RetrievalResultDataModel],
        dataset_name: str,
        table_id_to_table: Dict[Tuple[str, str], List[List]],
    ) -> List[DownstreamGeneratedResultDataModel]:
        """
        Given the query and the retrieval results, generate downstream task results. Uses generator to generate a sql query.
        """
        if not self.current_dataset:
            self.current_dataset = dataset_name

        downstream_task_results = []
        for query_id, query_str, result in zip(
            query_batch[QUERY_ID_COL_NAME],
            query_batch[QUERY_COL_NAME],
            retrieval_results,
        ):
            # First, aggregate together all table_ids coming from the same db_id
            db_id_to_tables: Dict[str, List[str]] = {}
            for db_id, table_id in result.retrieval_results:
                if db_id not in db_id_to_tables:
                    db_id_to_tables[db_id] = []
                db_id_to_tables[db_id].append(table_id)
            # Next, serialize the schema of those tables to a string
            table_str = ""
            for db_id, table_ids in db_id_to_tables.items():
                table_str += self._get_schema(dataset_name=self.current_dataset, db_id=db_id, table_ids=table_ids)
            generated_sql = self.task_generator.generate(
                table_str=table_str,
                query=query_str,
            )
            downstream_task_results.append(
                DownstreamGeneratedResultDataModel(
                    dataset_name=dataset_name,
                    query_id=query_id,
                    generated_results=(
                        generated_sql["sql_query"],
                        generated_sql["database_id"],
                    ),
                ),
            )

        return downstream_task_results

    def _update_downstream_task_metrics(
        self,
        query_batch: Dict[str, List],
        downstream_results: List[DownstreamGeneratedResultDataModel],
    ) -> None:
        """
        Update metric tracked for fact verification's performance calculation.
        Specifically, update the `self.pred_sql` and `self.ref_sql` lists
        based on the predicted answers in downstream_results and ground truth answers in query_batch.
        """

        for downstream_answer in downstream_results:
            self.pred_sql.append(downstream_answer.generated_results)
        self.ref_sql.extend(list(zip(query_batch[ANSWER_COL_NAME], query_batch[DATABASE_ID_COL_NAME])))
        if DIFFICULTY_COL_NAME in query_batch:
            self.difficulties.extend(query_batch[DIFFICULTY_COL_NAME])
        else:
            self.difficulties.extend(["Default"] * len(downstream_results))

    def _calculate_downstream_task_performance(self, **kwargs) -> Text2SQLTaskPerformanceDataModel:
        """
        Calculate downstream task metrics for the fact verification task.
        Metrics computed: accuracy, f1, precision, and recall.
        """
        if DATASET_NAME in kwargs:
            self.current_dataset = kwargs[DATASET_NAME]
        if self.current_dataset not in self.database_dirs:
            raise ValueError(f"{self.current_dataset} does not have path to database files.")
        db_path = self.database_dirs[self.current_dataset]
        result = Text2SQLTaskPerformanceDataModel(
            scores=evaluate_sql_execution(
                self.pred_sql,
                self.ref_sql,
                self.difficulties,
                db_path,
                1,
                60,
                self.include_ves,
            )
        )

        self.pred_sql = []
        self.ref_sql = []
        self.difficulties = []
        self.current_dataset = None
        return result
