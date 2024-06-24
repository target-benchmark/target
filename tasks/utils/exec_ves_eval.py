import os
import pdb
import sys
import json
from typing import Dict, List, Tuple, Union
import numpy as np
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
import time
import math


def result_callback(result):
    exec_result.append(result)


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
    iterate_num,
) -> float:
    predicted_sql, predicted_db = predicted_sql_and_db
    ground_truth, ground_truth_db = ground_truth_sql_and_db
    # given a predicted sql, ground truth sql, and the respective db paths of each, get efficiency results.
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

    if set(predicted_res) == set(ground_truth_res):
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
    return time_ratio


def execute_model(
    predicted_sql: Tuple[str, str],
    ground_truth: Tuple[str, str],
    db_root_path: str,
    idx: int,
    iterate_num: int,
    meta_time_out: float,
):
    try:
        # you can personalize the total timeout number
        # larger timeout leads to more stable ves
        # while it needs more your patience....
        time_ratio = func_timeout(
            meta_time_out * iterate_num,
            iterated_execute_sql,
            args=(predicted_sql, ground_truth, db_root_path, iterate_num),
        )
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f"timeout",)]
        time_ratio = 0
    except Exception as e:
        result = [(f"error",)]  # possibly len(query) > 512 or not executable
        time_ratio = 0
    result = {"sql_idx": idx, "time_ratio": time_ratio}
    return result


def run_sqls_parallel(
    pred_sqls: List[Tuple[str, str]],
    gt_sqls: List[Tuple[str, str]],
    db_root_path: str,
    num_cpus=1,
    iterate_num=100,
    meta_time_out=30.0,
):
    pool = mp.Pool(processes=num_cpus)
    for i, sql_pair in enumerate(zip(pred_sqls, gt_sqls)):
        predicted_sql, ground_truth = sql_pair
        pool.apply_async(
            execute_model,
            args=(
                predicted_sql,
                ground_truth,
                db_root_path,
                i,
                iterate_num,
                meta_time_out,
            ),
            callback=result_callback,
        )
    pool.close()
    pool.join()


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


def compute_ves_by_diff(
    exec_results: List[Dict[str, Union[int, float]]], difficulties: List[str]
):
    assert len(exec_results) == len(
        difficulties
    ), "number of executed results and number of difficulty ratings are not the same!"
    results_by_difficulty = {}
    simple_results, moderate_results, challenging_results = [], [], []
    for result, difficulty in zip(exec_results, difficulties):
        if difficulty in results_by_difficulty:
            results_by_difficulty[difficulty].append(result)
        else:
            results_by_difficulty[difficulty] = [result]
    ves_by_difficulty = {
        difficulty: {"ves": compute_ves(results), "num_sqls": len(results)}
        for difficulty, results in results_by_difficulty.items()
    }
    ves_by_difficulty["all"] = {
        "ves": compute_ves(exec_results),
        "num_sqls": len(exec_results),
    }

    return ves_by_difficulty


def evaluate_ves(
    predicted_sqls: List[Tuple[str, str]],
    ground_truth_sqls: List[Tuple[str, str]],
    difficulties: List[str],
    db_root_path: str,
    num_cpus: int,
    meta_time_out: float,
) -> Dict[str, Dict[str, Union[int, float]]]:

    run_sqls_parallel(
        predicted_sqls,
        ground_truth_sqls,
        db_root_path,
        num_cpus=num_cpus,
        meta_time_out=meta_time_out,
    )
    exec_result = sort_results(exec_result)
    print("start calculate")
    return compute_ves_by_diff(exec_result, difficulties)
