# %%
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mlgym.evaluation.utils import (  # EXIT_STATUS_MAP,; MODEL_LOGOS,
    ACTION_COLOR_MAP,
    MODEL_NAME_MAP,
    MODEL_SHORT_NAME_MAP,
    MODELS,
    TASKS,
    get_action_results,
    get_exit_status_results,
    process_trajectories,
)

# set_custom_font()

# %%
# Variables
traj_parent_dir = "../trajectories/mlgym_bench_v0/"
traj_pattern = "default__t-0.00__p-0.95__c-4.00__install-0__parallel_agents"
models = MODELS
output_dir = Path("../assets/figs/")


# %%
# Get all trajectory information
all_trajectories = defaultdict(dict)
acceptable_exit_statuses = ["autosubmission (max_steps)", "submitted"]

for task_id, _ in TASKS.items():
    task_results = process_trajectories(traj_parent_dir, traj_pattern, task_id, models)
    all_trajectories[task_id] = task_results

exit_status_results = get_exit_status_results(all_trajectories)
action_results = get_action_results(all_trajectories)


# %%
def plot_es_counts_per_model(exit_status_results: dict, output_path: str) -> None:
    """
    Plot a stacked bar chart of exit status counts with flipped axes using seaborn.

    Args:
        exit_status_results (dict): Dictionary containing exit status counts per model.
            Expected key 'es_counts_per_model' with structure:
            {model: {exit_status: count, ...}, ...}
        output_path (str): Path to save the plotted figure in PDF format.
    """

    raw_counts: dict = exit_status_results.get("es_counts_per_model", {})
    data: dict = {model: dict(counts) for model, counts in raw_counts.items()}
    df: pd.DataFrame = pd.DataFrame.from_dict(data, orient="index").fillna(0)

    # sort the columns by the name of the model
    df = df[df.sum().sort_values(ascending=False).index]

    # Remove the "Success" exit status column if it exists.
    if "Success" in df.columns:
        df.drop("Success", axis=1, inplace=True)
    if "Max Steps" in df.columns:
        df.drop("Max Steps", axis=1, inplace=True)
    if "API" in df.columns:
        df.drop("API", axis=1, inplace=True)

    # Transpose so that exit statuses are on x-axis and models are columns.
    df_flip: pd.DataFrame = df.T

    # Convert to long format for seaborn
    df_melted = df_flip.reset_index().melt(
        id_vars="index", var_name="Model", value_name="Count"
    )
    df_melted.rename(columns={"index": "Exit Status"}, inplace=True)

    # Map model names to short names
    df_melted["Model_Short"] = df_melted["Model"].map(
        lambda x: MODEL_SHORT_NAME_MAP.get(x, x)
    )

    # Order exit statuses by total counts (descending)
    exit_status_totals = (
        df_melted.groupby("Exit Status")["Count"].sum().sort_values(ascending=False)
    )
    df_melted["Exit Status"] = pd.Categorical(
        df_melted["Exit Status"], categories=exit_status_totals.index, ordered=True
    )

    # Create pivot table for manual stacking
    pivot_df = df_melted.pivot(
        index="Exit Status", columns="Model_Short", values="Count"
    ).fillna(0)
    pivot_df = pivot_df.reindex(exit_status_totals.index)  # Maintain order

    # Create figure
    fig, ax = plt.subplots(figsize=(6.5, 4))

    # Get seaborn colors
    colors = sns.color_palette("deep", n_colors=len(pivot_df.columns))

    # Manual stacking with seaborn styling
    bottom_values = np.zeros(len(pivot_df.index))
    legend_handles = []

    for i, model in enumerate(pivot_df.columns):
        bars = ax.bar(
            range(len(pivot_df.index)),
            pivot_df[model],
            bottom=bottom_values,
            color=colors[i],
            alpha=1.0,
            label=model,
            width=0.6,
            edgecolor="black",
            linewidth=0.5,
        )
        legend_handles.append(bars[0])
        bottom_values += pivot_df[model]

    # Style the plot - keep all spines for box appearance
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    # Add tick marks
    ax.tick_params(axis="x", direction="out", length=4, width=0.8, labelsize=8)
    ax.tick_params(axis="y", direction="out", length=4, width=0.8, labelsize=8)

    # Set labels and ticks
    ax.set_xlabel("Exit Status", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_xticks(range(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.index, rotation=0, ha="center")

    # Style the legend with thinner frame
    legend = ax.legend(
        legend_handles,
        pivot_df.columns,
        loc="upper right",
        fontsize=8,
        frameon=True,
        title=None,
    )

    # Explicitly override global patch settings to make frame visible
    if legend:
        frame = legend.get_frame()
        frame.set_edgecolor("black")
        frame.set_linewidth(0.4)  # Thinner frame
        frame.set_facecolor("white")
        frame.set_alpha(1.0)
        frame.set_visible(True)

    # Add grid
    ax.grid(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


plot_es_counts_per_model(exit_status_results, "es_counts_per_model.pdf")


# %%


def plot_es_counts_per_model_heatmap(
    exit_status_results: dict, output_path: str
) -> None:
    """
    Plot a heatmap of exit status counts per model with counts displayed in each cell.

    Args:
        exit_status_results (dict): Dictionary containing exit status counts per model.
            Expected key 'es_counts_per_model' with structure:
            {model: {exit_status: count, ...}, ...}
        output_path (str): Path to save the plotted figure in PDF format.
    """

    raw_counts: dict = exit_status_results.get("es_counts_per_model", {})
    data: dict = {model: dict(counts) for model, counts in raw_counts.items()}
    df: pd.DataFrame = pd.DataFrame.from_dict(data, orient="index").fillna(0)

    # sort the columns by the name of the model
    df = df[df.sum().sort_values(ascending=False).index]

    # Remove the "Success" exit status column if it exists.
    if "Success" in df.columns:
        df.drop("Success", axis=1, inplace=True)
    if "Max Steps" in df.columns:
        df.drop("Max Steps", axis=1, inplace=True)
    if "API" in df.columns:
        df.drop("API", axis=1, inplace=True)

    df = df.reindex(MODELS)

    # Map model names to short names for y-axis
    df.index = [MODEL_NAME_MAP.get(col, col) for col in df.index]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Create heatmap with darker colors
    sns.heatmap(
        df,  # Transpose so models are on y-axis, exit statuses on x-axis
        annot=True,
        fmt="g",  # Format numbers as integers
        cmap=sns.light_palette(
            "navy", as_cmap=True
        ),  # Use continuous sequential colormap for count data
        robust=True,
        # square=True,
        linewidths=0.1,
        linecolor="lightgray",
        cbar_kws={"label": "Count"},
        ax=ax,
    )

    # Style the plot - keep all spines for box appearance
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    # Add tick marks
    # ax.tick_params(axis="x", direction="out", length=4, width=0.8, labelsize=8)
    # ax.tick_params(axis="y", direction="out", length=4, width=0.8, labelsize=8)

    # Set labels
    # ax.set_xlabel("Exit Status", fontsize=10)
    # ax.set_ylabel("Model", fontsize=10)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="center")
    plt.yticks(rotation=0)

    # Style the colorbar
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Count", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


plot_es_counts_per_model_heatmap(exit_status_results, "es_counts_per_model_heatmap.pdf")

# %%


def plot_failed_incomplete_runs_per_model(
    exit_status_results: dict, output_path: str
) -> None:
    """
    Plot a bar chart of the failed and incomplete runs for each model.
    Failed runs have no agent scores, incomplete runs have scores but failed to submit.
    Uses seaborn barplot with consistent color palette.

    Args:
        exit_status_results (dict): Dictionary containing failed and incomplete run counts.
            Expected keys: 'failed_runs_per_model', 'incomplete_runs_per_model'
        output_path (str): Path to save the plotted figure in PDF format.
    """

    failed_runs = exit_status_results["failed_runs_per_model"]
    incomplete_runs = exit_status_results["incomplete_runs_per_model"]

    # Debug: Print the data to see what's happening
    print(f"Failed runs sample: {dict(list(failed_runs.items())[:3])}")
    print(f"Incomplete runs sample: {dict(list(incomplete_runs.items())[:3])}")
    print(f"Total incomplete across all models: {sum(incomplete_runs.values())}")

    # Create DataFrame with model order
    df = pd.DataFrame(
        {
            "Failed Runs": [failed_runs[m] for m in MODELS],
            "Incomplete Runs": [incomplete_runs[m] for m in MODELS],
        },
        index=MODELS,
    )

    # Sort by total failed runs while preserving model order
    totals = df["Failed Runs"]
    sort_order = totals.sort_values(ascending=False).index
    df = df.reindex(sort_order)

    # Map model names to short names
    df.rename(index=lambda m: MODEL_SHORT_NAME_MAP.get(m, m), inplace=True)

    # Convert to long format for seaborn
    df_melted = df.reset_index().melt(
        id_vars="index", var_name="Status", value_name="Count"
    )
    df_melted.rename(columns={"index": "Model"}, inplace=True)

    # Print melted data for debugging
    print("Melted data sample:")
    print(df_melted.head(10))
    print(f"Max count: {df_melted['Count'].max()}")
    print(
        f"Incomplete run counts: {df_melted[df_melted['Status'] == 'Incomplete Runs']['Count'].tolist()}"
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use seaborn barplot with consistent color palette
    colors = sns.color_palette("deep", 2)
    sns.barplot(
        data=df_melted, x="Model", y="Count", hue="Status", palette=colors, ax=ax
    )

    # Style the plot - fix x-axis labels (fontsize=8, not bold)
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right", fontsize=8, fontweight="normal"
    )
    ax.set_ylabel("Count", fontsize=10)
    ax.set_xlabel("")

    # Fix y-axis scaling - use max value with minimal padding
    max_val = df_melted["Count"].max()
    y_max = max_val + max(2, int(max_val * 0.05))  # Add minimal padding
    ax.set_ylim(0, y_max)

    # Set y-axis ticks with appropriate steps
    if y_max <= 20:
        step = 5
    elif y_max <= 50:
        step = 10
    else:
        step = 15
    yticks = list(range(0, y_max + 1, step))
    ax.set_yticks(yticks)
    ax.set_yticklabels(list(map(str, yticks)), fontsize=8)

    # Style the legend
    legend = ax.legend(loc="upper right", fontsize=8, title=None)
    if legend:
        frame = legend.get_frame()
        frame.set_edgecolor("black")
        frame.set_linewidth(0.4)
        frame.set_facecolor("white")
        frame.set_alpha(1.0)

    # Add grid
    ax.grid(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Style spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    plt.tight_layout()
    plt.savefig(output_dir / output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


plot_failed_incomplete_runs_per_model(
    exit_status_results, "failed_incomplete_runs_per_model.pdf"
)

# %%


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
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(df.index))
    width = 0.35

    # Using first two colors from MODEL_COLOR_MAP
    failed_color = "#FD5901"
    incomplete_color = "#F78104"

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


# %%


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
    ax.set_ylim(0, max(df["Count"]) * 1.1)
    yticks = list(range(0, max(df["Count"]) + 900, 1000))
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


# %%


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
    yticks = list(range(0, 700, 100))
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


# %%


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
    print("=" * 20)
    print(totals)
    sort_order = totals.sort_values(ascending=False).index
    df = df.reindex(sort_order)

    # Rename model indices using rename
    df.rename(index=lambda m: MODEL_SHORT_NAME_MAP.get(m, m), inplace=True)

    # Create figure with adjusted height ratios for legend
    fig, ax = plt.subplots(figsize=(6.5, 4))

    fig, ax = plt.subplots(figsize=(9, 4))

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


# %%


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


# %%
