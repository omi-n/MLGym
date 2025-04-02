"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Script to plot termination errors, failed runs and action analyses.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simple_parsing import parse

from mlgym.evaluation.utils import (  # EXIT_STATUS_MAP,; MODEL_LOGOS,
    ACTION_COLOR_MAP,
    COLOR_CHOICE,
    MODEL_COLOR_MAP,
    MODEL_SHORT_NAME_MAP,
    MODELS,
    TASKS,
    process_trajectories,
    set_custom_font,
)

# sns.set_style("dark")
set_custom_font()


@dataclass
class Options:
    """Options for processing results."""

    traj_parent_dir: str  # path to the root trajectory directory. for example, "trajectories/dnathani"
    traj_pattern: str  # pattern to match trajectory directories. Should NOT include the model name (at the start), and the run seed marker (at the end). Eg: "imageClassificationCifar10__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__parallel_agents"
    models: list[str] = field(
        default_factory=lambda: MODELS
    )  # list of models to process


def plot_es_counts_per_model(exit_status_results: dict, output_path: str) -> None:
    """
    Plot a stacked bar chart of exit status counts with flipped axes.

    Args:
        exit_status_results (dict): Dictionary containing exit status counts per model.
            Expected key 'es_counts_per_model' with structure:
            {model: {exit_status: count, ...}, ...}
        output_path (str): Path to save the plotted figure in PDF format.
    """
    # sns.set_style("dark")
    # set_custom_font()

    raw_counts: dict = exit_status_results.get("es_counts_per_model", {})
    data: dict = {model: dict(counts) for model, counts in raw_counts.items()}
    df: pd.DataFrame = pd.DataFrame.from_dict(data, orient="index").fillna(0)

    # sort the columns by the name of the model
    df = df[df.sum().sort_values(ascending=False).index]

    # Remove the "Success" exit sti]us column if it exists.
    if "Success" in df.columns:
        df.drop("Success", axis=1, inplace=True)
    if "Max Steps" in df.columns:
        df.drop("Max Steps", axis=1, inplace=True)
    if "API" in df.columns:
        df.drop("API", axis=1, inplace=True)

    # Transpose so that exit statuses are on x-axis and models are columns.
    df_flip: pd.DataFrame = df.T

    # Create figure with adjusted height ratios for legend
    fig, ax = plt.subplots(figsize=(6.5, 4))

    bottom = np.zeros(len(df_flip))
    for i, model in enumerate(df_flip.columns):
        print(f"Model: {model}, Color: {MODEL_COLOR_MAP[model]}")
        ax.bar(
            df_flip.index,
            df_flip[model],
            bottom=bottom,
            width=0.5,
            color=MODEL_COLOR_MAP[model],
            label=MODEL_SHORT_NAME_MAP.get(model, model),
        )
        bottom += df_flip[model]

    # yticks = list(range(0, 18, 3))
    # ax.set_yticks(yticks, yticks, fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)

    plt.xticks(
        range(len(df_flip.index)),
        list(df_flip.index),
        rotation=0,
        ha="center",
        fontsize=8,
    )
    plt.ylabel("Count", fontsize=10)
    # Add legend at the bottom
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", fontsize=8)

    ax.grid(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Adjust layout to make room for logos
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()


def plot_es_counts_per_task(exit_status_results: dict, output_path: str) -> None:
    """
    Plot a stacked bar chart of exit status counts with flipped axes.
    """
    pass


def plot_failed_incomplete_runs_per_model(
    exit_status_results: dict, output_path: str
) -> None:
    """
    Plot a bar chart of the failed and incomplete runs for each model.
    Failed runs have no agent scores, incomplete runs have scores but failed to submit.
    Model names are mapped using MODEL_NAME_MAP and bars are colored using MODEL_COLOR_MAP.
    Model logos are placed on top of each bar.

    Args:
        exit_status_results (dict): Dictionary containing failed and incomplete run counts.
            Expected keys: 'failed_runs_per_model', 'incomplete_runs_per_model'
        output_path (str): Path to save the plotted figure in PDF format.
    """
    # sns.set_style("dark")
    # set_custom_font()

    failed_runs = exit_status_results["failed_runs_per_model"]
    incomplete_runs = exit_status_results["incomplete_runs_per_model"]

    # Create DataFrame with model order
    df = pd.DataFrame(
        {
            "Failed": [failed_runs[m] for m in MODELS],
            "Incomplete": [incomplete_runs[m] for m in MODELS],
        },
        index=MODELS,
    )

    # Sort by total while preserving color mapping
    totals = df["Failed"]
    sort_order = totals.sort_values(ascending=False).index
    df = df.reindex(sort_order)
    colors = [MODEL_COLOR_MAP[m] for m in df.index]
    df.rename(index=lambda m: MODEL_SHORT_NAME_MAP.get(m, m), inplace=True)

    # Create figure with adjusted height ratios for legend
    fig, ax = plt.subplots(figsize=(6.5, 4))

    x = np.arange(len(df.index))
    width = 0.35

    failed_bars = ax.bar(
        x - width / 2, df["Failed"], width, label="Failed Runs", color=colors
    )

    incomplete_bars = ax.bar(
        x + width / 2,
        df["Incomplete"],
        width,
        label="Incomplete Runs",
        edgecolor=colors,
        facecolor="none",
        hatch="////",
        linewidth=2,
    )
    print(df.index)

    ax.set_xticks(x)
    ax.set_xticklabels(
        df.index, rotation=0, ha="center", fontsize=10, fontweight="bold"
    )
    plt.ylabel("Count", fontsize=10)

    # Add model logos
    # for idx, model in enumerate(models):
    #     if model in MODEL_LOGOS:
    #         logo_path, zoom = MODEL_LOGOS[model]
    #         try:
    #             img = plt.imread(logo_path)
    #             if img.shape[2] == 3:
    #                 img = np.dstack([img, np.ones((img.shape[0], img.shape[1]))])

    #             imagebox = OffsetImage(img, zoom=zoom)
    #             ab = AnnotationBbox(imagebox, (idx, -6), frameon=False,
    #                               box_alignment=(0.5, 1))
    #             ax.add_artist(ab)
    #         except Exception as e:
    #             print(f"Error loading logo for {model}: {e}")
    #             ax.text(idx, -6, MODEL_NAME_MAP.get(model, model),
    #                    ha='center', va='top', fontsize=10, fontweight='bold')

    yticks = list(range(0, 13, 2))
    ax.set_yticks(yticks, list(map(str, yticks)), fontsize=8)

    # Create black legend handles
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="black", label="Failed Runs"),
        Patch(
            facecolor="none", edgecolor="black", hatch="///", label="Incomplete Runs"
        ),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=8)

    ax.grid(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Adjust layout to make room for logos
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()


def plot_failed_incomplete_runs_per_task(
    exit_status_results: dict, output_path: str
) -> None:
    """
    Plot a bar chart of the failed and incomplete runs for each task.
    Failed runs have no agent scores, incomplete runs have scores but failed to submit.

    Args:
        exit_status_results (dict): Dictionary containing failed and incomplete run counts.
            Expected keys: 'failed_runs_per_task', 'incomplete_runs_per_task'
        output_path (str): Path to save the plotted figure in PDF format.
    """
    # sns.set_style("dark")
    # set_custom_font()

    failed_runs = exit_status_results["failed_runs_per_task"]
    incomplete_runs = exit_status_results["incomplete_runs_per_task"]

    # Create DataFrame with task names from TASKS dictionary
    df = pd.DataFrame(
        {
            "Failed": [failed_runs[t] for t in TASKS],
            "Incomplete": [incomplete_runs[t] for t in TASKS],
        },
        index=[TASKS[t]["shortname"] for t in TASKS],
    )

    df.drop(index="PD", inplace=True)
    df.drop(index="F-MNIST", inplace=True)

    # Sort by total while preserving task names
    totals = df["Failed"]
    df = df.reindex(totals.sort_values(ascending=False).index)

    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(df.index))
    width = 0.35

    # Using first two colors from MODEL_COLOR_MAP
    failed_color = COLOR_CHOICE[2]  # "#FD5901"
    incomplete_color = COLOR_CHOICE[2]  # "#F78104"

    # Solid bars for failed runs
    ax.bar(x - width / 2, df["Failed"], width, label="Failed Runs", color=failed_color)

    # Hollow hatched bars for incomplete runs
    ax.bar(
        x + width / 2,
        df["Incomplete"],
        width,
        label="Incomplete Runs",
        edgecolor=incomplete_color,
        facecolor="none",
        hatch="////",
        linewidth=2,
    )

    ax.set_xticks(x)
    # yticks = list(range(0, 25, 5))
    # ax.set_yticks(yticks, yticks, fontsize=12, fontweight='bold')
    ax.set_xticklabels(df.index, rotation=0, ha="center", fontsize=7)
    ax.tick_params(axis="y", labelsize=8)
    plt.ylabel("Count", fontsize=10)

    # Create legend handles with correct colors
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=failed_color, label="Failed Runs"),
        Patch(
            facecolor="none",
            edgecolor=incomplete_color,
            hatch="///",
            label="Incomplete Runs",
        ),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    # Add horizontal grid lines
    ax.grid(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()


def plot_total_actions(action_results: dict, output_path: str) -> None:
    """
    Plot a bar chart of the number of times each action type is taken across all tasks and models.

    Args:
        action_results (dict): Dictionary containing action counts.
            Expected key 'action_counts' with structure:
            {action_type: count, ...}
        output_path (str): Path to save the plotted figure in PDF format.
    """
    # sns.set_style("dark")
    # set_custom_font()

    # Get the total counts for each action type
    action_counts = action_results["action_counts"]

    # Create DataFrame
    df = pd.DataFrame({"Count": action_counts.values()}, index=action_counts.keys())

    # Sort by count while preserving action type colors
    sort_order = df["Count"].sort_values(ascending=False).index
    df = df.reindex(sort_order)
    colors = [ACTION_COLOR_MAP[action] for action in sort_order]

    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(df.index))

    # Create bars with hatched pattern
    bars = ax.bar(x, df["Count"], color=colors[: len(df)], width=0.5)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=0, ha="center", fontsize=8)

    # Set y-axis limits to start from 0 to show all bars
    ax.set_ylim(0, 4000)
    yticks = list(range(0, 4100, 500))
    ax.set_yticks(yticks, list(map(str, yticks)), fontsize=8)
    plt.ylabel("Count", fontsize=10)

    # Add y-axis grid lines
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Add legend with action types
    # legend_elements = [
    #     plt.Rectangle((0,0),1,1, facecolor=color, label=action)
    #     for action, color in zip(df.index, colors[:len(df)])
    # ]
    # ax.legend(handles=legend_elements, loc='upper right',
    #          fontsize=8)

    # Add some padding at the top for the value labels
    ax.margins(x=0.1, y=0.1)

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()


def plot_actions_per_step(action_results: dict, output_path: str) -> None:
    """
    Plot a stacked bar chart showing the distribution of actions at each step.

    Args:
        action_results (dict): Dictionary containing action counts per step.
            Expected key 'actions_per_step' with structure:
            {step_number: {action_type: count, ...}, ...}
        output_path (str): Path to save the plotted figure in PDF format.
    """
    # sns.set_style("dark")
    # set_custom_font()

    # Get the actions per step
    actions_per_step = action_results["actions_per_step"]

    # Create DataFrame with all steps from 0 to 50
    df = pd.DataFrame(index=range(51))  # 0 to 50 inclusive
    action_types = list(ACTION_COLOR_MAP.keys())

    # Fill in the counts for each action type at each step
    for action in action_types:
        df[action] = [
            actions_per_step.get(step, {}).get(action, 0) for step in range(51)
        ]

    # Fill NaN values with 0
    df = df.fillna(0)
    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 4))

    # Create stacked bars
    bottom = np.zeros(51)
    for action in action_types:
        ax.bar(
            df.index,
            df[action],
            bottom=bottom,
            color=ACTION_COLOR_MAP[action],
            label=action,
            width=1.0,
        )
        bottom += df[action]

    # Set x-axis ticks and labels
    xticks = [1] + list(range(5, 51, 5))
    yticks = list(range(0, 260, 50))
    ax.set_xticks(xticks, list(map(str, xticks)), fontsize=8)
    ax.set_yticks(yticks, list(map(str, yticks)), fontsize=8)
    plt.ylabel("Count", fontsize=10)
    ax.set_xlim(-0.5, 51)
    ax.set_xlabel("Step Number", fontsize=8)

    # Add y-axis grid lines
    ax.grid(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(loc="upper right", fontsize=8, ncols=len(action_types) // 2)

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()


def plot_actions_per_model(action_results: dict, output_path: str) -> None:
    """
    Plot a stacked bar chart showing the distribution of actions for each model.

    Args:
        action_results (dict): Dictionary containing action counts per model.
            Expected key 'actions_per_model' with structure:
            {model_id: {action_type: count, ...}, ...}
        output_path (str): Path to save the plotted figure in PDF format.
    """
    # sns.set_style("dark")
    # set_custom_font()

    actions_per_model = action_results["actions_per_model"]
    print(actions_per_model)
    action_types = list(ACTION_COLOR_MAP.keys())

    # Create DataFrame
    # custom order to fit the legends
    df = pd.DataFrame(index=MODELS)
    for action in action_types:
        df[action] = [actions_per_model[model].get(action, 0) for model in MODELS]

    # Sort by total while preserving model colors
    totals = df.sum(axis=1)
    print(totals)
    sort_order = totals.sort_values(ascending=False).index
    df = df.reindex(sort_order)

    # Rename model indices using rename
    df.rename(index=lambda m: MODEL_SHORT_NAME_MAP.get(m, m), inplace=True)

    # Create figure with adjusted height ratios for legend
    fig, ax = plt.subplots(figsize=(6.5, 4))

    bottom = np.zeros(len(df))
    for action in action_types:
        ax.bar(
            df.index,
            df[action],
            bottom=bottom,
            color=ACTION_COLOR_MAP[action],
            label=action,
            width=0.5,
        )
        bottom += df[action]

    ax.set_xticks(range(len(df.index)))
    ax.set_xticklabels(df.index, rotation=0, ha="center", fontsize=8)

    # # Add model logos
    # for idx, model in enumerate(models):
    #     if model in MODEL_LOGOS:
    #         logo_path, zoom = MODEL_LOGOS[model]
    #         try:
    #             img = plt.imread(logo_path)
    #             if img.shape[2] == 3:
    #                 img = np.dstack([img, np.ones((img.shape[0], img.shape[1]))])

    #             imagebox = OffsetImage(img, zoom=zoom)
    #             ab = AnnotationBbox(imagebox, (idx, -max(df.sum()) * 0.1),
    #                               frameon=False, box_alignment=(0.5, 1))
    #             ax.add_artist(ab)
    #         except Exception as e:
    #             print(f"Error loading logo for {model}: {e}")
    #             ax.text(idx, -max(df.sum()) * 0.1, MODEL_NAME_MAP.get(model, model),
    #                    ha='center', va='top', fontsize=10, fontweight='bold')

    plt.yticks(fontsize=8)
    plt.ylabel("Count", fontsize=10)

    # Add legend at the bottom
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="best", ncols=len(action_types) // 2, fontsize=8)

    ax.grid(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Adjust layout to make room for logos
    # plt.subplots_adjust(bottom=0.25)
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()


def plot_actions_per_task(action_results: dict, output_path: str) -> None:
    """
    Plot a stacked bar chart showing the distribution of actions for each task.

    Args:
        action_results (dict): Dictionary containing action counts per task.
            Expected key 'actions_per_task' with structure:
            {task_id: {action_type: count, ...}, ...}
        output_path (str): Path to save the plotted figure in PDF format.
    """
    # sns.set_style("dark")
    # set_custom_font()

    actions_per_task = action_results["actions_per_task"]
    action_types = list(ACTION_COLOR_MAP.keys())

    # Create DataFrame
    df = pd.DataFrame(index=list(TASKS.keys()))
    for action in action_types:
        df[action] = [actions_per_task[task].get(action, 0) for task in TASKS.keys()]

    # Rename task indices to display names
    df.rename(index=lambda t: TASKS[t]["shortname"], inplace=True)

    # Sort by total while preserving task names
    totals = df.sum(axis=1)
    df = df.reindex(totals.sort_values(ascending=False).index)

    # Create figure with adjusted height ratios for legend
    fig, ax = plt.subplots(figsize=(8, 4))
    # fig = plt.figure(figsize=(12, 9))
    # gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.2])

    # # Create main plot and legend areas
    # ax = fig.add_subplot(gs[0])
    # ax_legend = fig.add_subplot(gs[1])
    # ax_legend.axis('off')

    # Use colors from MODEL_COLOR_MAP plus extra color

    # Create stacked bars
    bottom = np.zeros(len(df))
    for action in action_types:
        ax.bar(
            df.index,
            df[action],
            bottom=bottom,
            color=ACTION_COLOR_MAP[action],
            label=action,
        )
        bottom += df[action]

    plt.xticks(rotation=0, ha="center", fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylabel("Count", fontsize=9)

    # Add legend at the bottom
    handles, labels = ax.get_legend_handles_labels()
    # ax_legend.legend(handles, labels, loc='center', ncol=len(action_types),
    #                 frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.legend(
        handles, labels, loc="upper right", fontsize=7, ncols=len(action_types) // 2
    )

    ax.grid(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()


def get_action_results(trajectories: dict) -> dict:
    """
    Get the number of times each action is taken across all tasks and models.

    Args:
        trajectories (dict): Dictionary containing trajectories for each task and model.
            Structure: {task_id: {model_id: {"trajectories": [trajectory_list], ...}, ...}, ...}
            Each trajectory_list contains a list of actions taken during that run.

    Returns:
        dict: Dictionary with action counts.
            Structure: {action_name: count, ...}
    """
    action_counts = defaultdict(int)
    actions_per_model = defaultdict(lambda: defaultdict(int))
    actions_per_task = defaultdict(lambda: defaultdict(int))
    actions_per_step = defaultdict(lambda: defaultdict(int))
    file_editing = ["create", "edit", "insert"]
    file_viewer = ["open", "goto", "scroll_up", "scroll_down"]
    validation = ["validate"]
    submit = ["submit"]
    search = ["search_dir", "search_file", "find_file"]
    python_scripts = ["torchrun", "python", "python3", "accelerate", "deepspeed"]

    for task_id, task_results in trajectories.items():
        for model_id, model_results in task_results.items():
            for trajectory in model_results["trajectories"]:
                for step_idx, step in enumerate(trajectory):
                    # Clean up the action string by removing whitespace
                    action = step["action"].strip()
                    # Map actions to categories
                    if any(action.startswith(cmd) for cmd in file_editing):
                        action = "Edit"
                    elif any(action.startswith(cmd) for cmd in file_viewer):
                        action = "View"
                    elif any(action.startswith(cmd) for cmd in validation):
                        action = "Validate"
                    elif any(action.startswith(cmd) for cmd in submit):
                        action = "Submit"
                    elif any(action.startswith(cmd) for cmd in search):
                        action = "Search"
                    elif any(action.startswith(cmd) for cmd in python_scripts):
                        action = "Python"
                    else:
                        action = "Bash"

                    action_counts[action] += 1
                    actions_per_model[model_id][action] += 1
                    actions_per_task[task_id][action] += 1
                    actions_per_step[step_idx + 1][action] += 1

    # Convert defaultdict to regular dict
    return {
        "action_counts": dict(action_counts),
        "actions_per_model": dict(actions_per_model),
        "actions_per_task": dict(actions_per_task),
        "actions_per_step": {
            step: dict(actions) for step, actions in actions_per_step.items()
        },
    }


def get_exit_status_results(trajectories: dict) -> dict[str, dict]:
    """
    Get the number of times each exit status occurs across all tasks and models.
    """
    exit_status_results = {}
    total_es_counts = defaultdict(lambda: 0)
    es_counts_per_model = defaultdict(lambda: defaultdict(lambda: 0))
    es_counts_per_task = defaultdict(lambda: defaultdict(lambda: 0))

    # no agent scores
    failed_runs_per_model = defaultdict(lambda: 0)

    # agent scores but failed to submit at the end of the run
    incomplete_runs_per_model = defaultdict(lambda: 0)
    failed_runs_per_task = defaultdict(lambda: 0)
    incomplete_runs_per_task = defaultdict(lambda: 0)

    for task_id, task_results in trajectories.items():
        for model_id, model_results in task_results.items():
            assert len(model_results["exit_statuses"]) == len(
                model_results["scores"]
            ), (
                f"Exit statuses and scores length mismatch for model {model_id} in task {task_id}"
            )
            for exit_status, score in zip(
                model_results["exit_statuses"], model_results["scores"]
            ):
                total_es_counts[exit_status] += 1
                es_counts_per_model[model_id][exit_status] += 1
                es_counts_per_task[task_id][exit_status] += 1
                failed = 0
                incomplete = 0
                if "agent" not in score:
                    failed = 1
                elif "agent" in score and len(score["agent"]) == 0:
                    failed = 1
                elif (
                    "agent" in score
                    and len(score["agent"]) > 0
                    and exit_status not in ["Success", "Max Steps"]
                ):
                    incomplete = 1

                failed_runs_per_model[model_id] += failed
                incomplete_runs_per_model[model_id] += incomplete
                failed_runs_per_task[task_id] += failed
                incomplete_runs_per_task[task_id] += incomplete

    print("All exit statuses:\n")
    print(f"{total_es_counts.keys()}")

    assert sum(total_es_counts.values()) == 432, (
        f"Total exit status counts: {total_es_counts}"
    )
    assert (
        sum(failed_runs_per_model.values()) + sum(incomplete_runs_per_model.values())
        <= 432
    ), f"Failed runs per model: {failed_runs_per_model}"
    assert sum(failed_runs_per_model.values()) == sum(failed_runs_per_task.values())
    assert sum(incomplete_runs_per_model.values()) == sum(
        incomplete_runs_per_task.values()
    )
    assert (
        sum([sum(es_counts_per_model[model].values()) for model in es_counts_per_model])
        == 432
    ), f"ES counts per model: {es_counts_per_model}"
    assert (
        sum([sum(es_counts_per_task[task].values()) for task in es_counts_per_task])
        == 432
    ), f"ES counts per task: {es_counts_per_task}"

    # print(f"Failed runs per model: {dict(failed_runs_per_model)}")
    # print(f"Incomplete runs per model: {dict(incomplete_runs_per_model)}")
    # print(f"Failed runs per task: {dict(failed_runs_per_task)}")
    # print(f"Incomplete runs per task: {dict(incomplete_runs_per_task)}")
    # print(f"Total Exit Statuses: {dict(total_es_counts)}")
    # print(f"ES counts per model: {dict({key: dict(value) for key, value in es_counts_per_model.items()})}")

    return {
        "total_es_counts": total_es_counts,
        "es_counts_per_model": es_counts_per_model,
        "es_counts_per_task": es_counts_per_task,
        "failed_runs_per_model": failed_runs_per_model,
        "incomplete_runs_per_model": incomplete_runs_per_model,
        "failed_runs_per_task": failed_runs_per_task,
        "incomplete_runs_per_task": incomplete_runs_per_task,
    }


def main(options: Options):
    # get all results.json files from trajectory directory pattern
    all_trajectories = defaultdict(dict)
    acceptable_exit_statuses = ["autosubmission (max_steps)", "submitted"]

    for task_id, _ in TASKS.items():
        task_results = process_trajectories(
            options.traj_parent_dir, options.traj_pattern, task_id, options.models
        )
        all_trajectories[task_id] = task_results

    exit_status_results = get_exit_status_results(all_trajectories)
    action_results = get_action_results(all_trajectories)
    # pprint(action_results)

    # Plotting
    plot_es_counts_per_model(exit_status_results, "assets/figs/error_per_model.pdf")
    plot_failed_incomplete_runs_per_model(
        exit_status_results, "assets/figs/failed_runs_model.pdf"
    )
    plot_failed_incomplete_runs_per_task(
        exit_status_results, "assets/figs/failed_runs_task.pdf"
    )
    plot_total_actions(action_results, "assets/figs/total_actions_analysis.pdf")
    plot_actions_per_step(action_results, "assets/figs/actions_per_step.pdf")
    plot_actions_per_model(action_results, "assets/figs/actions_per_model.pdf")
    plot_actions_per_task(action_results, "assets/figs/actions_per_task.pdf")

    # pprint(exit_status_results)


if __name__ == "__main__":
    args = parse(Options)
    main(args)
