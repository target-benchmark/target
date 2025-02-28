import json
import math
import multiprocessing as mp
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Protocol, Tuple, Union

import numpy as np
from func_timeout import FunctionTimedOut, func_timeout
from pydantic import BaseModel

from target_benchmark.dataset_loaders.AbsDatasetLoader import AbsDatasetLoader
from target_benchmark.dataset_loaders.LoadersDataModels import (
    DatasetConfigDataModel,
    NeedleInHaystackDatasetConfigDataModel,
    Text2SQLDatasetConfigDataModel,
)
from target_benchmark.dictionary_keys import DATASET_NAME
from target_benchmark.generators.GeneratorPrompts import NO_CONTEXT_TABLE_PROMPT
from target_benchmark.retrievers.RetrieversDataModels import RetrievalResultDataModel
from target_benchmark.retrievers.utils import markdown_table_str


class PreprocessTableCallable(Protocol):
    def __call__(
        self,
        result: RetrievalResultDataModel,
        table_id_to_table: Dict[Tuple[str, str], List[List]],
        **kwargs: Any,
    ) -> str:
        ...


class PreprocessQueryCallable(Protocol):
    def __call__(self, query_str: str, **kwargs) -> str:
        ...


class PostprocessGenerationCallable(Protocol):
    def __call__(self, generation: Dict[str, str], **kwargs) -> Union[str, Tuple[str, str], List[str]]:
        ...


def default_preprocess_query(query_str: str, **kwargs) -> str:
    """

    Returns the same query string. default value of the `AbsTask._parallelize` function's `preprocess_query` parameter.

    Parameters:
        query_str (str): the query to preprocess
    Returns:
        the same query string unchanged.

    """
    return query_str


def default_postprocess_generation(generation: Dict[str, str], **kwargs) -> str:
    """

    Returns content of the generator's response. default value of the `AbsTask._parallelize` function's `postprocess_generation` parameter.

    Parameters:
        generation (dict): generator response
    Returns:
        the content from the response

    """
    return generation["content"]


def default_preprocess_table(
    result: RetrievalResultDataModel, table_id_to_table: Dict[Tuple[str, str], List[List]], **kwargs
) -> str:
    """

    Returns the markdown string of table. default value of the `AbsTask._parallelize` function's `preprocess_table` parameter.

    Parameters:
        result (RetrievalResultDataModel): retrieval result with the (table_id, db_id) identifiers
        table_id_to_table (dict): mapping of table id to table nested lists
    Returns:
        formatted table content string of all (top k) retrieved tables.

    """
    return build_table_content_string(result.retrieval_results, table_id_to_table)


def build_table_content_string(
    retrieval_results: List[Tuple[str, str]],
    table_id_to_table: Dict[Tuple[str, str], List[List]],
) -> str:
    tables = set()
    for retrieved_table_id in retrieval_results:
        if retrieved_table_id not in table_id_to_table:
            return NO_CONTEXT_TABLE_PROMPT
        else:
            table_str = f"table name: {str(retrieved_table_id)}\n" + markdown_table_str(table_id_to_table[retrieved_table_id])
            tables.add(table_str)
    return "\n".join(table_content for table_content in tables)


def construct_persistence_path(dir: Union[Path, None], dataset_name: str, top_k: float):
    if not dir:
        return None
    final_path = dir / dataset_name / f"{top_k}.jsonl"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.touch()
    return final_path


def update_query_batch(
    query_batch: Dict[str, List],
    start_idx: int,
) -> Dict[str, List]:
    updated_batch = {}
    for key in query_batch:
        updated_batch[key] = query_batch[key][start_idx:]
    return updated_batch


def append_results(
    results: List[BaseModel],
    path_to_persistence: Union[Path, None],
):
    if not path_to_persistence:
        return
    path_to_persistence.touch()
    with open(path_to_persistence, "a") as file:
        for retrieval_result in results:
            file.write(retrieval_result.model_dump_json() + "\n")


def load_data_model_from_persistence_file(
    path_to_persistence: Path,
    datamodel: type[BaseModel],
) -> List[BaseModel]:
    loaded_models = []
    with open(path_to_persistence, "r") as file:
        for line in file.readlines():
            loaded_models.append(datamodel.model_validate_json(line))
    loaded_models.sort(lambda x: x.query_id)
    return loaded_models


def get_num_lines_in_file(path: Path):
    with open(path, "r") as file:
        return sum(1 for _ in file)


def generate_batches_from_file(
    path_to_persistence: Union[Path, None],
    batch_size: int,
    datamodel: type[BaseModel],
) -> Generator[List[BaseModel], None, None]:
    if path_to_persistence:
        # get total number of lines in a file
        num_lines = get_num_lines_in_file(path_to_persistence)
        loaded_models = []
        with open(path_to_persistence, "r") as file:
            for i, line in enumerate(file):
                # check if we've reached the end of the previously written file
                if i >= num_lines:
                    break
                loaded_models.append(datamodel.model_validate_json(line))
                if len(loaded_models) >= batch_size:
                    yield loaded_models
                    loaded_models = []
        if loaded_models:
            yield loaded_models


def find_resume_indices(
    dataset_loaders: Dict[str, AbsDatasetLoader],
    path_to_results: Union[Path, None] = None,
):
    start_query_idx = {dataset_name: 0 for dataset_name in dataset_loaders.keys()}
    if path_to_results is None:
        return start_query_idx
    with open(path_to_results, "r") as file:
        for line in file.readlines():
            start_query_idx[json.loads(line)[DATASET_NAME]] += 1
    return start_query_idx


def clean_abnormal(input):
    input = np.asarray(input)
    processed_list = []
    mean = np.mean(input, axis=0)
    std = np.std(input, axis=0)
    for x in input:
        if x < mean + 3 * std and x > mean - 3 * std:
            processed_list.append(x)
    return processed_list


def execute_sql(sql, cursor):
    start_time = time.time()
    cursor.execute(sql)
    exec_time = time.time() - start_time

    return exec_time


def iterated_execute_sql(
    predicted_sql_and_db: Tuple[str, str],
    ground_truth_sql_and_db: Tuple[str, str],
    db_root_path: str,
    iterate_num: int,
    include_ves: bool = False,
) -> float:
    assert len(predicted_sql_and_db) == 2, f"malformatted predicted sql db pairs: {predicted_sql_and_db}"
    assert len(ground_truth_sql_and_db) == 2, f"malformatted ground truth sql db pairs: {ground_truth_sql_and_db}"
    predicted_sql, predicted_db = predicted_sql_and_db
    ground_truth, ground_truth_db = ground_truth_sql_and_db
    # given a predicted sql, ground truth sql,
    # and the respective db paths of each, get efficiency results.
    pred_conn = sqlite3.connect(os.path.join(db_root_path, predicted_db, f"{predicted_db}.sqlite"))
    pred_cursor = pred_conn.cursor()

    gt_conn = sqlite3.connect(os.path.join(db_root_path, ground_truth_db, f"{ground_truth_db}.sqlite"))
    gt_cursor = gt_conn.cursor()

    diff_list = []

    pred_cursor.execute(predicted_sql)
    predicted_res = pred_cursor.fetchall()

    gt_cursor.execute(ground_truth)
    ground_truth_res = gt_cursor.fetchall()

    time_ratio = 0.0
    sql_execution_res = 0
    if set(predicted_res) == set(ground_truth_res):
        sql_execution_res = 1
        if include_ves:
            for _ in range(iterate_num):
                predicted_time = execute_sql(predicted_sql, pred_cursor)
                ground_truth_time = execute_sql(ground_truth, gt_cursor)
                diff_list.append(ground_truth_time / predicted_time)
            processed_diff_list = clean_abnormal(diff_list)
            time_ratio = sum(processed_diff_list) / len(processed_diff_list)

    pred_cursor.close()
    pred_conn.close()
    gt_cursor.close()
    gt_conn.close()
    return time_ratio, sql_execution_res


def execute_model(
    predicted_sql: Tuple[str, str],
    ground_truth: Tuple[str, str],
    db_root_path: str,
    idx: int,
    iterate_num: int,
    meta_time_out: float,
    include_ves: bool = False,
) -> Dict[str, Union[int, float]]:
    sql_execution_res = 0
    try:
        # you can personalize the total timeout number
        # larger timeout leads to more stable ves
        # while it needs more your patience....
        time_ratio, sql_execution_res = func_timeout(
            meta_time_out * iterate_num,
            iterated_execute_sql,
            args=(predicted_sql, ground_truth, db_root_path, iterate_num, include_ves),
        )
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        time_ratio = 0
        sql_execution_res = 0
    except Exception:
        time_ratio = 0
        sql_execution_res = 0
    return {
        "sql_idx": idx,
        "time_ratio": time_ratio,
        "sql_execution_res": sql_execution_res,
    }


def run_sqls_parallel(
    pred_sqls: List[Tuple[str, str]],
    gt_sqls: List[Tuple[str, str]],
    db_root_path: str,
    num_cpus=1,
    iterate_num=10,
    meta_time_out=60.0,
    include_ves: bool = False,
) -> List[Dict[str, Union[int, float]]]:
    pool = mp.Pool(processes=num_cpus)
    results = []
    for i, sql_pair in enumerate(zip(pred_sqls, gt_sqls)):
        predicted_sql, ground_truth = sql_pair
        future_result = pool.apply_async(
            execute_model,
            args=(
                predicted_sql,
                ground_truth,
                db_root_path,
                i,
                iterate_num,
                meta_time_out,
                include_ves,
            ),
        )
        results.append(future_result)

    pool.close()
    pool.join()
    exec_result = [result.get() for result in results]  # Safely collect results
    return exec_result


def sort_results(list_of_dicts):
    return sorted(list_of_dicts, key=lambda x: x["sql_idx"])


def compute_ves(exec_results: List[Dict[str, Union[int, float]]]) -> float:
    num_queries = len(exec_results)
    total_ratio = 0
    count = 0

    for i, result in enumerate(exec_results):
        if result["time_ratio"] != 0:
            count += 1
        total_ratio += math.sqrt(result["time_ratio"]) * 100
    ves = total_ratio / num_queries
    return ves


def compute_acc(exec_results: List[Dict[str, Union[int, float]]]) -> float:
    num_queries = len(exec_results)
    return sum(res["sql_execution_res"] for res in exec_results) / num_queries


def compute_performance_by_diff(
    exec_results: List[Dict[str, Union[int, float]]],
    difficulties: List[str],
    include_ves: bool = False,
):
    assert len(exec_results) == len(
        difficulties
    ), "number of executed results and number of difficulty ratings are not the same!"
    results_by_difficulty = {}
    for result, difficulty in zip(exec_results, difficulties):
        if difficulty in results_by_difficulty:
            results_by_difficulty[difficulty].append(result)
        else:
            results_by_difficulty[difficulty] = [result]
    results_by_difficulty["all"] = exec_results
    performance_by_difficulty = {}
    for difficulty, results in results_by_difficulty.items():
        performances = {}
        performances["accuracy"] = compute_acc(results)
        performances["num_sqls"] = len(results)
        if include_ves:
            performances["ves"] = compute_ves(results)
        performance_by_difficulty[difficulty] = performances

    return performance_by_difficulty


def evaluate_sql_execution(
    predicted_sqls: List[Tuple[str, str]],
    ground_truth_sqls: List[Tuple[str, str]],
    difficulties: List[str],
    db_root_path: str,
    num_cpus: int,
    meta_time_out: float,
    include_ves: bool = False,
) -> Dict[str, Dict[str, Union[int, float]]]:
    exec_result = run_sqls_parallel(
        predicted_sqls,
        ground_truth_sqls,
        db_root_path,
        num_cpus=num_cpus,
        meta_time_out=meta_time_out,
        include_ves=include_ves,
    )
    exec_result = sort_results(exec_result)
    return compute_performance_by_diff(exec_result, difficulties, include_ves)


def validate_dataset_configs(constructed_config: Dict[str, DatasetConfigDataModel]) -> bool:
    """
    Validate that the dataset configs are constructured correctly.
    Current rules (more to be added potentially):
    - cannot be empty
    - cannot be only needle in haystack datasets
    - cannot have NIH with text2sql (not tried but probably NIH datasets inserted into sqlite will not end well.)
    Returns:
        True if dataset configs are correctly constructed.
        Otherwise throw assertion error
    """
    num_non_nih = 0
    num_total = 0
    num_text2sql = 0
    for dataset_name, config in constructed_config.items():
        if not isinstance(config, NeedleInHaystackDatasetConfigDataModel):
            num_non_nih += 1
        if isinstance(config, Text2SQLDatasetConfigDataModel):
            num_text2sql += 1
        num_total += 1
    assert num_total != 0, "No datasets configurated!"
    assert num_non_nih != 0, "Cannot have only Needle in Haystack datasets!"
    assert not num_text2sql or (num_text2sql and num_non_nih >= num_total), "Cannot have T2SQL & NIH!"
    return True
