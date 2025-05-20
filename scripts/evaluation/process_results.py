"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Script to process results.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from simple_parsing import parse


@dataclass
class Options:
    """Options for processing results."""

    traj_parent_dir: str  # path to the root trajectory directory. for example, "trajectories/dnathani"
    traj_pattern: str  # pattern to match trajectory directories. Should NOT include the model name (at the start), and the run seed marker (at the end). Eg: "imageClassificationCifar10__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__parallel_agents"
    priority_metric: str  # metric to use for selecting best agent score
    models: list[str] = field(default_factory=list)  # list of models to process
    metric_direction: str = (
        "maximize"  # direction of the priority metric. "maximize" or "minimize"
    )


def get_best_attempt(results: dict, priority_metric: str, metric_direction: str) -> int:
    """Returns the index of the best agent according to the priority metric."""
    best_attempt_idx = -1
    best_metric_value = (
        -float("inf") if metric_direction == "maximize" else float("inf")
    )
    for i, result in enumerate(results):
        if priority_metric not in result:
            print(
                f"Priority metric {priority_metric} not found in result for agent {i}"
            )
            continue
        metric_value = result[priority_metric]
        if (metric_direction == "maximize" and metric_value > best_metric_value) or (
            metric_direction == "minimize" and metric_value < best_metric_value
        ):
            best_metric_value = metric_value
            best_attempt_idx = i
    return best_attempt_idx


def get_scores(
    results: list[dict], priority_metric: str, metric_direction: str, model: str
) -> dict[str, list]:
    """Returns the average score for a single model across all parallel runs. Also returns the number of failed runs i.e. no agent scores were found"""
    scores = {}
    scores["best_attempt"] = defaultdict(list)
    scores["last_attempt"] = defaultdict(list)
    failed_runs = 0
    for i, result in enumerate(results):
        best_attempt_idx = get_best_attempt(
            result["agent"], priority_metric, metric_direction
        )
        # filter for failed runs
        if best_attempt_idx == -1:
            failed_runs += 1
            continue
        # we only get the last attempt if agent results are not empty
        last_attempt = result["agent"][-1]

        # print(f"Best score for Model {model}, run {i}: {result['agent'][best_agent_idx]}")
        for k, v in result["agent"][best_attempt_idx].items():
            scores["best_attempt"][k].append(v)
        # update last attempt
        for k, v in last_attempt.items():
            scores["last_attempt"][k].append(v)
    scores["failed_runs"] = [int(failed_runs)]
    return scores


def get_baseline_scores(
    results: list[dict], priority_metric: str, metric_direction: str
) -> dict[str, list]:
    """Returns the baseline scores for the task.
    Note: returns a dict with list of length 1 to match the format with agent scores.
    """
    baseline_scores = defaultdict(list)
    for result in results:
        if "baseline" not in result:
            continue
        for k, v in result["baseline"].items():
            baseline_scores[k].append(v)
        # break after the first baseline score is found
        break
    baseline_scores["failed_runs"] = [int(0)]
    return baseline_scores


def print_results(results: dict):
    """Prints the results for all models in a nice table."""
    print("Results by model:")
    for model, scores in results.items():
        print(f"\n{model}:")
        for key, value in scores.items():
            if key == "failed_runs":
                print(f"  {key}: {value[0]}")
            elif key == "incomplete_runs":
                print(f"  {key}: {value}")
            elif key == "exit_stats":
                print(f"  {key}: {value}")
            elif key == "best_attempt" or key == "last_attempt":
                print(f"  {key}:")
                for metric, scores in value.items():
                    print(f"    {metric}:")
                    print(f"      Values: {scores}")
                    print(f"      Min: {np.min(scores):.3f}, Max: {np.max(scores):.3f}")
                    print(
                        f"      Average: {np.mean(scores):.3f} ± {np.std(scores):.3f}"
                    )
            else:
                print(f"    {key}:")
                print(f"      Values: {value}")
                print(f"      Min: {np.min(value):.3f}, Max: {np.max(value):.3f}")
                print(f"      Average: {np.mean(value):.3f} ± {np.std(value):.3f}")


def main(options: Options):
    # get all results.json files from trajectory directory pattern
    scores = defaultdict(dict)
    acceptable_exit_statuses = ["autosubmission (max_steps)", "submitted"]
    for model in options.models:
        failed_runs = 0  # number of runs with no agent scores
        incomplete_runs = 0  # number of runs that exit due to error
        exit_statuses = []
        traj_dir_pattern = f"{options.traj_parent_dir}/*{model}*{options.traj_pattern}*"
        traj_dirs = sorted(list(Path().glob(traj_dir_pattern)))
        results = []
        for traj_dir in traj_dirs:
            if not (Path(traj_dir) / "results.json").exists():
                failed_runs += 1
            # find trajectory file ending in .traj
            traj_files = list(Path(traj_dir).glob("*.traj"))
            if len(traj_files) != 0:
                traj = json.load(open(traj_files[0]))
                last_action = "unknown"
                for history in reversed(traj["history"]):
                    if history["role"] == "assistant":
                        last_action = history["action"].strip()
                        break

                try:
                    exit_status = traj["info"]["exit_status"]
                    if exit_status == "":
                        exit_status = f"unknown_error ({last_action})"
                except KeyError:
                    exit_status = "unknown_error"
                if exit_status not in acceptable_exit_statuses:
                    incomplete_runs += 1
                exit_statuses.append(exit_status)

            results.append(json.load(open(Path(traj_dir) / "results.json")))

        # add baseline scores if not already added
        if "baseline" not in scores:
            scores["baseline"] = get_baseline_scores(
                results, options.priority_metric, options.metric_direction
            )
        scores[model] = get_scores(
            results, options.priority_metric, options.metric_direction, model
        )
        scores[model]["failed_runs"][0] += failed_runs
        scores[model]["incomplete_runs"] = incomplete_runs
        scores[model]["exit_stats"] = exit_statuses
    print_results(scores)


if __name__ == "__main__":
    args = parse(Options)
    main(args)
