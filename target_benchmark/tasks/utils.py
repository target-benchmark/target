import json
import math
import multiprocessing as mp
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from func_timeout import FunctionTimedOut, func_timeout
from pydantic import BaseModel

from target_benchmark.dataset_loaders.AbsDatasetLoader import AbsDatasetLoader
from target_benchmark.dataset_loaders.LoadersDataModels import (
    DatasetConfigDataModel,
    NeedleInHaystackDatasetConfigDataModel,
)
from target_benchmark.dictionary_keys import DATASET_NAME
from target_benchmark.generators.GeneratorPrompts import NO_CONTEXT_TABLE_PROMPT
from target_benchmark.retrievers.utils import markdown_table_str


def build_table_content_string(
    retrieval_results: List[Tuple[str, str]],
    table_id_to_table: Dict[Tuple[str, str], List[List]],
) -> str:
    tables = set()
    for retrieved_table_id in retrieval_results:
        if retrieved_table_id not in table_id_to_table:
            return NO_CONTEXT_TABLE_PROMPT
        else:
            tables.add(markdown_table_str(table_id_to_table[retrieved_table_id]))
    return "\n".join(table_content for table_content in tables)


def load_data_model_from_persistence_file(
    path_to_persistence: Path,
    datamodel: type[BaseModel],
) -> List[BaseModel]:
    loaded_models = []
    with open(path_to_persistence, "r") as file:
        for line in file.readlines():
            loaded_models.append(datamodel.model_validate_json(line))
    return loaded_models


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
    assert (
        len(predicted_sql_and_db) == 2
    ), f"malformatted predicted sql db pairs: {predicted_sql_and_db}"
    assert (
        len(ground_truth_sql_and_db) == 2
    ), f"malformatted ground truth sql db pairs: {ground_truth_sql_and_db}"
    predicted_sql, predicted_db = predicted_sql_and_db
    ground_truth, ground_truth_db = ground_truth_sql_and_db
    # given a predicted sql, ground truth sql,
    # and the respective db paths of each, get efficiency results.
    pred_conn = sqlite3.connect(
        os.path.join(db_root_path, predicted_db, f"{predicted_db}.sqlite")
    )
    pred_cursor = pred_conn.cursor()

    gt_conn = sqlite3.connect(
        os.path.join(db_root_path, ground_truth_db, f"{ground_truth_db}.sqlite")
    )
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


def validate_dataset_configs(
    constructed_config: Dict[str, DatasetConfigDataModel]
) -> bool:
    """
    Validate that the dataset configs are constructured correctly.
    Current rules (more to be added potentially):
    - cannot be empty
    - cannot be only needle in haystack datasets
    Returns:
        True if dataset configs are correctly constructed.
        Otherwise throw assertion error
    """
    num_non_nih = 0
    num_total = 0
    for dataset_name, config in constructed_config.items():
        if not isinstance(config, NeedleInHaystackDatasetConfigDataModel):
            num_non_nih += 1
        num_total += 1
    assert num_total != 0, "No datasets configurated!"
    assert num_non_nih != 0, "Cannot have only Needle in Haystack datasets!"
    return True
