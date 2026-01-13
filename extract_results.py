#!/usr/bin/env python3
"""
Extract results from MLGym trajectory directories and calculate reward_stats
similar to Harbor's format.

Usage:
    python extract_results.py <trajectories_dir>
    python extract_results.py trajectories/seed1000
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def extract_task_name(dirname: str) -> str:
    """Extract task name from directory name like 'litellm-gpt-5-mini__battleOfSexes__default__...'"""
    parts = dirname.split("__")
    if len(parts) >= 2:
        return parts[1]
    return dirname


def calculate_reward(agent_scores: dict, baseline_scores: dict) -> float:
    """
    Calculate continuous reward as (achieved - baseline) for each metric.
    Skips 'time' and 'reward std' metrics.
    Uses case-insensitive metric matching.
    """
    reward = 0.0
    # Create lowercase lookup for agent scores
    agent_lower = {k.lower(): v for k, v in agent_scores.items()}

    for metric_name, baseline_value in baseline_scores.items():
        if metric_name.lower() in ["time", "reward std"]:
            continue
        # Try exact match first, then case-insensitive
        achieved_value = agent_scores.get(metric_name)
        if achieved_value is None:
            achieved_value = agent_lower.get(metric_name.lower(), 0.0)
        reward += achieved_value - baseline_value
    return reward


def process_trajectories(trajectories_dir: Path) -> dict:
    """
    Process all trajectory directories and extract reward stats.

    Returns a dict with reward_stats in Harbor format.
    """
    reward_stats = defaultdict(list)
    all_results = []

    for trajectory_path in sorted(trajectories_dir.iterdir()):
        if not trajectory_path.is_dir():
            continue

        results_file = trajectory_path / "results.json"
        if not results_file.exists():
            print(f"Warning: No results.json in {trajectory_path.name}")
            continue

        try:
            with open(results_file, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing {results_file}: {e}")
            continue

        agent_entries = data.get("agent", [])
        baseline = data.get("baseline", {})

        if not agent_entries:
            print(f"Warning: No agent entries in {trajectory_path.name}")
            # default to zero scores
            first_agent_score = {k: 0.0 for k in baseline}
        else:
            # Get recent agent score
            first_agent_score = agent_entries[-1]

        task_name = extract_task_name(trajectory_path.name)

        # Override baseline for rlMountainCarContinuousReinforce to avoid massive negative penalty
        if task_name == "rlMountainCarContinuousReinforce":
            baseline = {k: 0.0 for k in baseline}

        # Calculate reward (difference from baseline)
        reward = calculate_reward(first_agent_score, baseline)

        # Store in Harbor-like format
        reward_key = str(reward)
        reward_stats[reward_key].append(task_name)

        all_results.append(
            {
                "task": task_name,
                "trajectory": trajectory_path.name,
                "agent_score": first_agent_score,
                "baseline": baseline,
                "reward": reward,
            }
        )

    return {
        "reward_stats": {"reward": dict(reward_stats)},
        "detailed_results": all_results,
        "summary": {
            "n_trials": len(all_results),
            "mean_reward": sum(r["reward"] for r in all_results) / len(all_results) if all_results else 0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Extract results from MLGym trajectory directories")
    parser.add_argument(
        "trajectories_dir",
        type=str,
        help="Directory containing trajectory subdirectories with results.json files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: prints to stdout)",
    )
    args = parser.parse_args()

    trajectories_dir = Path(args.trajectories_dir)
    if not trajectories_dir.exists():
        print(f"Error: Directory {trajectories_dir} does not exist")
        return 1

    results = process_trajectories(trajectories_dir)

    output_json = json.dumps(results, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Results written to {args.output}")
    else:
        print(output_json)

    return 0


if __name__ == "__main__":
    exit(main())
