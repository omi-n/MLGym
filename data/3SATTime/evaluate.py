"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import json
import pickle
import time

from heuristic import heuristic_fn
from solver import DPLL


def evaluate_dpll(test_data_path: str) -> tuple[float, int]:
    sat_data = pickle.load(open(test_data_path, "rb"))

    start = time.time()
    correct = 0
    incorrect = 0
    num_decisions = 0
    for ind, num_vars, num_clauses, formula, is_sat, _ in sat_data[3:]:
        alpha = num_clauses / num_vars
        dpll_solver = DPLL(formula, heuristic_fn)
        if not (dpll_solver.solve() ^ is_sat):
            correct += 1
        else:
            return 1e8, int(1e8)
        sample_decisions = dpll_solver.num_decisions
        num_decisions += sample_decisions
    wall_clock = round(time.time() - start, 3)
    metrics = {
        "Time": wall_clock,
        "Correct": correct,
        "Incorrect": incorrect,
        "Decisions": num_decisions,
    }
    print(json.dumps(metrics))
    return wall_clock, num_decisions


evaluate_dpll("data/dataset.pkl")
