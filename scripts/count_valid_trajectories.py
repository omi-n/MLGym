"""
Script to count valid trajectories for each model and task combination.
"""

import csv
from pathlib import Path

# Models we're interested in
MODELS = [
    "llama3-405b-tools",
    "deepseek-r1",
    "gpt4o2",
    "claude-35-sonnet-new",
    "claude-37-sonnet",
    "gemini-15-pro",
    "gemini-20-flash-thinking",
    # "gemini-20-pro",
    "gemini-25-pro",
    "gpt-o1",
    "gpt-o3-mini",
    "llama4-17b-16",
    "llama4-17b-128",
]

# Tasks from utils.py
TASKS = {
    "regressionKaggleHousePrice": "House Price",
    "3SATTime": "3-SAT",
    "imageClassificationCifar10": "CIFAR-10",
    "imageClassificationFMnist": "Fashion MNIST",
    "imageCaptioningCOCO": "Image Captioning",
    "languageModelingFineWeb": "Language Modeling",
    "naturalLanguageInferenceMNLI": "MNLI",
    "battleOfSexes": "Battle of Sexes",
    "prisonersDilemma": "Prisoners Dilemma",
    "blotto": "Blotto",
    "rlBreakoutMinAtar": "Breakout",
    "rlMetaMazeMisc": "Meta Maze",
    "rlMountainCarContinuous": "Mountain Car Continuous",
}


def count_valid_trajectories(traj_parent_dir: str, model: str, task: str) -> int:
    """
    Count the number of valid trajectories for a given model and task.

    Args:
        traj_parent_dir (str): Parent directory containing trajectory folders
        model (str): Model name to search for
        task (str): Task name to search for

    Returns:
        int: Number of valid trajectories found
    """
    pattern = f"*{model}__{task}__default__t-0.00__p-0.95__c-4.00__install-0__parallel_agents_run*"
    traj_dirs = sorted(list(Path(traj_parent_dir).glob(pattern)))

    valid_count = 0
    for traj_dir in traj_dirs:
        results_file = traj_dir / "results.json"
        if results_file.exists():
            valid_count += 1

    return valid_count


def main():
    """
    Main function to generate CSV with trajectory counts.
    """
    traj_parent_dir = "trajectories/mlgym_bench_v0"

    # Create CSV writer
    with open("trajectory_counts.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(["Task"] + MODELS)

        # Write data for each task
        for task_id, task_name in TASKS.items():
            row = [task_name]
            for model in MODELS:
                count = count_valid_trajectories(traj_parent_dir, model, task_id)
                row.append(str(count))
            writer.writerow(row)

    print("CSV file 'trajectory_counts.csv' has been generated.")


if __name__ == "__main__":
    main()
