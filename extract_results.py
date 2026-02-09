#!/usr/bin/env python3
"""
Extract results from MLGym trajectory directories and calculate reward_stats
similar to Harbor's format.

Usage:
    python extract_results.py <trajectories_dir>
    python extract_results.py trajectories/seed1000
    python extract_results.py --prefix trajectories/ --output-report results_summary.txt
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import statistics


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
    For "lower is better" metrics (rmse, loss): uses baseline - achieved.
    """
    reward = 0.0
    lower_is_better = ["rmse", "loss", "incorrect"]
    # Create lowercase lookup for agent scores
    agent_lower = {k.lower(): v for k, v in agent_scores.items()}

    for metric_name, baseline_value in baseline_scores.items():
        if metric_name.lower() in ["time", "reward std"]:
            continue

        achieved_value = agent_lower.get(metric_name.lower())
        if achieved_value is None:
            reward -= abs(baseline_value)  # Penalize missing metric
        elif any(lib in metric_name.lower() for lib in lower_is_better):
            reward += baseline_value - achieved_value  # Lower is better
        else:
            reward += achieved_value - baseline_value  # Higher is better
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
        # if task_name == "rlMountainCarContinuousReinforce":
        #     baseline = {k: 0.0 for k in baseline}

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


def process_multiple_runs(base_dir: Path, prefix: str) -> dict:
    """
    Process multiple run folders matching a prefix and calculate statistics.

    Args:
        base_dir: Base directory containing run folders
        prefix: Prefix to match run folder names (e.g., "trajectories/")

    Returns:
        Dictionary with per-environment and overall statistics
    """
    # Find all matching directories
    matching_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)])

    if not matching_dirs:
        print(f"Warning: No directories found matching prefix '{prefix}' in {base_dir}")
        return {}

    print(f"Found {len(matching_dirs)} run folders matching prefix '{prefix}'")

    # Collect results across all runs
    all_run_results = []
    env_results = defaultdict(lambda: defaultdict(list))  # env -> metric -> [values]

    for run_dir in matching_dirs:
        print(f"Processing {run_dir.name}...")
        results = process_trajectories(run_dir)
        all_run_results.append({"run_dir": run_dir.name, "results": results})

        # Organize by environment (task)
        for detail in results.get("detailed_results", []):
            task = detail["task"]
            reward = detail["reward"]
            env_results[task]["rewards"].append(reward)
            env_results[task]["agent_scores"].append(detail["agent_score"])
            env_results[task]["baselines"].append(detail["baseline"])

    # Calculate statistics per environment
    env_stats = {}
    for env_name, metrics in env_results.items():
        rewards = metrics["rewards"]
        if rewards:
            env_stats[env_name] = {
                "n_trials": len(rewards),
                "mean_reward": statistics.mean(rewards),
                "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
                "min_reward": min(rewards),
                "max_reward": max(rewards),
            }

    # Calculate overall statistics
    all_rewards = [r for metrics in env_results.values() for r in metrics["rewards"]]
    overall_stats = {}
    if all_rewards:
        overall_stats = {
            "n_environments": len(env_stats),
            "n_total_trials": len(all_rewards),
            "mean_reward": statistics.mean(all_rewards),
            "std_reward": statistics.stdev(all_rewards) if len(all_rewards) > 1 else 0.0,
            "min_reward": min(all_rewards),
            "max_reward": max(all_rewards),
        }

    return {
        "run_folders": [r["run_dir"] for r in all_run_results],
        "per_environment_stats": env_stats,
        "overall_stats": overall_stats,
        "all_run_results": all_run_results,
    }


def generate_report(stats: dict) -> str:
    """Generate a human-readable report from statistics."""
    lines = []
    lines.append("=" * 80)
    lines.append("MLGym Benchmark Results Summary")
    lines.append("=" * 80)
    lines.append("")

    # Overall statistics
    overall = stats.get("overall_stats", {})
    if overall:
        lines.append("OVERALL STATISTICS")
        lines.append("-" * 80)
        lines.append(f"Total Environments: {overall['n_environments']}")
        lines.append(f"Total Trials: {overall['n_total_trials']}")
        lines.append(f"Mean Reward: {overall['mean_reward']:.4f} ± {overall['std_reward']:.4f}")
        lines.append(f"Min Reward: {overall['min_reward']:.4f}")
        lines.append(f"Max Reward: {overall['max_reward']:.4f}")
        lines.append("")

    # Run folders processed
    run_folders = stats.get("run_folders", [])
    if run_folders:
        lines.append(f"Run Folders Processed ({len(run_folders)}):")
        lines.append("-" * 80)
        for folder in run_folders:
            lines.append(f"  - {folder}")
        lines.append("")

    # Per-environment statistics
    env_stats = stats.get("per_environment_stats", {})
    if env_stats:
        lines.append("PER-ENVIRONMENT STATISTICS")
        lines.append("-" * 80)
        lines.append(f"{'Environment':<40} {'N':<6} {'Mean ± Std':<20} {'Min':<12} {'Max':<12}")
        lines.append("-" * 80)

        # Sort by mean reward descending
        sorted_envs = sorted(env_stats.items(), key=lambda x: x[1]["mean_reward"], reverse=True)

        for env_name, stats_data in sorted_envs:
            n_trials = stats_data["n_trials"]
            mean = stats_data["mean_reward"]
            std = stats_data["std_reward"]
            min_val = stats_data["min_reward"]
            max_val = stats_data["max_reward"]

            # Truncate long environment names
            env_display = env_name[:38] + ".." if len(env_name) > 40 else env_name

            lines.append(
                f"{env_display:<40} {n_trials:<6} {mean:>8.4f} ± {std:<8.4f} {min_val:<12.4f} {max_val:<12.4f}"
            )

        lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Extract results from MLGym trajectory directories")
    parser.add_argument(
        "trajectories_dir",
        type=str,
        nargs="?",
        default=".",
        help="Directory containing trajectory subdirectories with results.json files",
    )
    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        default=None,
        help="Prefix for run folder names to process multiple runs (e.g., 'trajectories/')",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: prints to stdout)",
    )
    parser.add_argument(
        "--output-report",
        "-r",
        type=str,
        default=None,
        help="Output report file path for human-readable summary",
    )
    args = parser.parse_args()

    base_dir = Path(args.trajectories_dir)
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return 1

    if args.prefix:
        # Process multiple runs with prefix
        results = process_multiple_runs(base_dir, args.prefix)

        # Generate and save report
        if args.output_report:
            report = generate_report(results)
            with open(args.output_report, "w") as f:
                f.write(report)
            print(f"Report written to {args.output_report}")

        # Output JSON if requested
        if args.output:
            output_json = json.dumps(results, indent=2)
            with open(args.output, "w") as f:
                f.write(output_json)
            print(f"JSON results written to {args.output}")
        elif not args.output_report:
            # Print JSON to stdout if no report requested
            print(json.dumps(results, indent=2))

        # Always print summary to console
        if results.get("overall_stats"):
            overall = results["overall_stats"]
            print("\n" + "=" * 60)
            print(f"Summary: {overall['n_environments']} environments, {overall['n_total_trials']} trials")
            print(f"Overall: {overall['mean_reward']:.4f} ± {overall['std_reward']:.4f}")
            print("=" * 60)
    else:
        # Single directory processing (original behavior)
        results = process_trajectories(base_dir)
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
