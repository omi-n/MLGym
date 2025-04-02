"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Script to get ranks of models, plot performance profiles, and compute AUP scores.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from simple_parsing import parse

from mlgym.evaluation.utils import (
    MODEL_COLOR_MAP,
    MODEL_COST_MAP,
    MODEL_LOGOS,
    MODEL_MARKER_MAP,
    MODEL_SHORT_NAME_MAP,
    MODELS,
    TASKS,
    get_best_attempt,
    set_custom_font,
)

set_custom_font()


@dataclass
class Options:
    """Options for processing aggregate results."""

    traj_parent_dir: str  # path to the root trajectory directory
    traj_pattern: str = "*"  # pattern to match trajectory directories
    models: list[str] = field(default_factory=lambda: MODELS)
    exclude_o1: bool = False
    output_file: str = "aggregate_results"  # output CSV file path

    def __post_init__(self):
        if self.exclude_o1:
            self.models = [model for model in self.models if model != "gpt-o1"]
            self.output_file = "aggregate_results_wo_o1"


# Example task dictionary structure - to be replaced with actual tasks
def calculate_rankings(
    scores: dict[str, float], all_models: list[str], metric_direction: str
) -> list[tuple[str, float]]:
    """
    Calculate rankings from scores dictionary.
    Models with no usable results get the lowest rank regardless of metric direction.
    """
    # First rank the models with actual scores
    if metric_direction == "maximize":
        rankings = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    else:  # minimize
        rankings = sorted(scores.items(), key=lambda x: x[1], reverse=False)

    # Add missing models at the end
    missing_models = set(all_models) - set(scores.keys())
    rankings.extend((model, float("nan")) for model in missing_models)

    return rankings


def compute_plackett_luce_ranking(
    rankings_dict: dict[str, list[tuple[str, float]]],
) -> list[str]:
    """Compute Plackett-Luce ranking from multiple rankings."""
    model_ranks = defaultdict(list)

    for task_rankings in rankings_dict.values():
        for rank, (model, _) in enumerate(task_rankings):
            model_ranks[model].append(rank + 1)

    avg_ranks = {model: np.mean(ranks) for model, ranks in model_ranks.items()}
    return [model for model, _ in sorted(avg_ranks.items(), key=lambda x: x[1])]  # type: ignore


def compute_broda_ranking(
    rankings_dict: dict[str, list[tuple[str, float]]],
) -> list[str]:
    """Compute BRODA (Borda count) ranking from multiple rankings."""
    model_scores = defaultdict(int)

    for task_rankings in rankings_dict.values():
        n_models = len(task_rankings)
        for rank, (model, _) in enumerate(task_rankings):
            model_scores[model] += n_models - rank

    return [
        model
        for model, _ in sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    ]


def calculate_performance_ratios(
    scores: dict[str, float],
    all_models: list[str],
    metric_direction: str,
    epsilon: float = 0.05,
) -> tuple[dict[str, float], float]:
    """Calculate performance ratios for each solver."""
    # if not scores:
    #     return {model: max_tau for model in all_models}

    ratios = {}
    if metric_direction == "minimize":
        # Find best score among available scores
        # best_score = min(scores.values())
        # worst_score = max(scores.values())

        best_score = float("inf")
        worst_score = float("-inf")
        feasible_models = []

        for model in all_models:
            # mark a method unfeasible if its score is negative or worse than the baseline
            if scores[model] < 0.0 or scores[model] > scores["baseline"]:
                continue
            else:
                feasible_models.append(model)
                best_score = min(best_score, scores[model])
                worst_score = max(worst_score, scores[model])

        # # Handle negative values by shifting all scores to be positive
        # if worst_score <= 0:
        #     shift = abs(worst_score) + 0.1  # Add 1 to ensure all values are positive
        #     shifted_scores = {k: v + shift for k, v in scores.items()}
        #     shifted_best = best_score + shift
        #     shifted_worst = worst_score + shift
        # else:
        #     shifted_scores = scores
        #     shifted_best = best_score
        #     shifted_worst = worst_score

        worst_ratio = worst_score / best_score
        max_tau = worst_ratio * (1 + epsilon)

        # Calculate raftios for all models
        ratios = {
            model: scores[model] / best_score if model in feasible_models else math.nan
            for model in all_models
        }

    else:  # maximize
        # Find best score among available scores
        best_score = float("-inf")
        worst_score = float("inf")
        feasible_models = []

        for model in all_models:
            # skip the unfeasible models with negative scores
            if scores[model] < 0.0 or scores[model] < scores["baseline"]:
                continue
            else:
                feasible_models.append(model)
                best_score = max(best_score, scores[model])
                worst_score = min(worst_score, scores[model])

        # # Handle negative values by shifting all scores to be positive
        # if worst_score <= 0:
        #     shift = abs(worst_score) + 0.1  # Add 1 to ensure all values are positive
        #     shifted_scores = {k: v + shift for k, v in scores.items()}
        #     shifted_best = best_score + shift
        #     shifted_worst = worst_score + shift
        # else:
        #     shifted_scores = scores
        #     shifted_best = best_score
        #     shifted_worst = worst_score

        worst_ratio = best_score / worst_score

        # Calculate ratios for all models
        ratios = {
            model: best_score / scores[model] if model in feasible_models else math.nan
            for model in all_models
        }

    for model in all_models:
        # Model doesn't have any valid scores. This branch generally shouldn't be hit, because in case of empty scores, we get a nan value which is handled in the next branch.
        if model not in ratios:
            ratios[model] = worst_ratio * (1 + epsilon)
        # Model doesn't have any valid scores
        elif math.isnan(ratios[model]):
            ratios[model] = worst_ratio * (1 + epsilon)

    max_tau = math.ceil(worst_ratio * (1 + epsilon))

    # ratios = {model: max_tau if ratios[model] > max_tau else ratios[model] for model in ratios}

    return ratios, max_tau


def compute_performance_profile(
    all_ratios: dict[str, dict[str, float]],
    all_models: list[str],
    tau_range: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute performance profile for each solver across all problems.
    Returns probability that solver is within factor tau of best for each tau.
    """
    profiles = {}
    n_problems = len(all_ratios)  # Total number of problems
    print(f"Number of problems: {n_problems}")

    if n_problems == 0:
        return {}

    # Compute profile for each solver
    for solver in all_models:
        profile = np.zeros_like(tau_range)
        for i, tau in enumerate(tau_range):
            # Count problems where ratio <= tau
            count = sum(
                1 for ratios in all_ratios.values() if np.log10(ratios[solver]) <= tau
            )
            profile[i] = count / n_problems
        profiles[solver] = profile

    return profiles


def plot_aup_vs_cost(
    aup_scores: dict[str, dict[str, float]],
    traj_parent_dir: str,
    traj_pattern: str,
    output_path: str,
) -> None:
    """
    Plot AUP vs average API cost for all models using company logos as markers.
    """

    # sns.set_style("dark")
    # set_custom_font()

    costs = defaultdict(list)
    tokens_sent = defaultdict(list)
    tokens_received = defaultdict(list)

    print("PRINTING COSTS AND TOKENS USED-----------------------------------\n")
    for traj_path in Path(traj_parent_dir).rglob(f"*{traj_pattern}*/*.traj"):
        if "rlMountainCarContinuousReinforce" in traj_path.name:
            continue
        model_name = str(traj_path).split("__")[0]
        prefix = f"{traj_parent_dir}/metagen-"
        if model_name.startswith(prefix):
            model_name = model_name[len(prefix) :]

        if model_name not in MODELS:
            print(f"Skipping model: {model_name}")
            continue

        with open(traj_path) as f:
            traj_data = json.load(f)
            if "info" in traj_data and "model_stats" in traj_data["info"]:
                stats = traj_data["info"]["model_stats"]

                input_price = MODEL_COST_MAP[model_name]["input_price"]
                output_price = MODEL_COST_MAP[model_name]["output_price"]

                if "tokens_sent" not in stats:
                    print(f"No tokens sent for {model_name} for task {traj_path.name}")
                    continue

                tokens_sent[model_name].append(stats["tokens_sent"])
                tokens_received[model_name].append(stats["tokens_received"])

                cost = (
                    stats["tokens_sent"] * input_price
                    + stats["tokens_received"] * output_price
                )
                costs[model_name].append(cost)

    avg_costs = {model: np.mean(model_costs) for model, model_costs in costs.items()}
    avg_tokens_sent = {
        model: np.mean(model_tokens_sent)
        for model, model_tokens_sent in tokens_sent.items()
    }
    avg_tokens_received = {
        model: np.mean(model_tokens_received)
        for model, model_tokens_received in tokens_received.items()
    }

    print(f"Average costs: {avg_costs}")
    print(f"Average tokens sent: {avg_tokens_sent}")
    print(f"Average tokens received: {avg_tokens_received}")

    fig, ax = plt.subplots(figsize=(6.5, 4))

    # Define logo paths and zoom levels for each model

    for model, scores in aup_scores.items():
        if model not in avg_costs:
            print(f"Model {model} not found in costs")
            continue

        x = avg_costs[model]
        y = scores.get("Best AUP", 0)
        print(f"Plotting {model} at ({x}, {y})")

        if model in MODEL_LOGOS:
            logo_path, zoom = MODEL_LOGOS[model]
            print(f"Attempting to load logo from {logo_path}")
            try:
                img = plt.imread(logo_path)
                # Make sure image has an alpha channel
                if img.shape[2] == 3:
                    img = np.dstack([img, np.ones((img.shape[0], img.shape[1]))])

                # Increase zoom by 50%
                imagebox = OffsetImage(img, zoom=zoom * 1.2)
                imagebox.image.axes = ax  # Set the axes reference

                ab = AnnotationBbox(
                    imagebox, (x, y), frameon=False, pad=0, box_alignment=(0.5, 0.5)
                )
                ax.add_artist(ab)
            except Exception as e:
                print(f"Error loading logo for {model}: {e}")
                ax.scatter(x, y, s=50, c=MODEL_COLOR_MAP[model])
        else:
            ax.scatter(x, y, s=50, c=MODEL_COLOR_MAP[model])

        # Add model name annotation
        # Set text position based on model
        if model in ["llama3-405b-tools", "gemini-15-pro", "gpt-o1"]:
            text_offset = (0, 10)  # Increased offset above logo
        else:
            text_offset = (0, -15)  # Increased offset below logo

        ax.annotate(
            MODEL_SHORT_NAME_MAP.get(model, model),
            (x, y),
            xytext=text_offset,
            textcoords="offset points",
            fontsize=8,
            ha="center",
        )
    xlim = math.ceil(max(avg_costs.values()))  # type: ignore
    xticks = np.arange(0, xlim + 2, 2)
    # Add grid lines
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)  # Put grid lines behind the points

    ax.set_xlabel("Average API Cost ($)", fontsize=10)
    ax.set_ylabel("Best Attempt AUP@4", fontsize=10)
    ax.set_xlim(0, xlim)  # Changed to start from 0 to show all points
    ax.set_xticks(xticks, xticks, fontsize=8)
    ax.set_ylim(0.895, 1.205)  # Set y-limits based on the data range
    yticks = np.arange(0.9, 1.25, 0.05)
    yticks = [np.round(y, 2) for y in yticks]
    ax.set_yticks(yticks, yticks, fontsize=8)

    plt.tight_layout()
    print(f"Saving plot to {output_path}")
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()


def plot_performance_profiles_dual(
    profiles: dict[str, np.ndarray],
    last_profiles: dict[str, np.ndarray],
    tau_range: np.ndarray,
    output_path: str,
    title: str,
):
    """Plot performance profiles for all solvers using seaborn styling."""

    # Set the style and color palette
    # sns.set_style("darkgrid")
    # set_custom_font()
    # sort profiles in the model order
    profiles = {m: profiles[m] for m in MODELS}
    last_profiles = {m: last_profiles[m] for m in MODELS}

    # Create figure with normal dimensions
    fig = plt.figure(figsize=(6.5, 4))

    # Create two main subplots with adjusted height ratios to leave space for legend
    # gs = fig.add_gridspec(2, 2, height_ratios=[15, 1])
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    # Create the two main subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Create a subplot for the legend spanning both columns at bottom
    # ax_legend = fig.add_subplot(gs[1, :])
    # ax_legend.axis('off')  # Hide the axis

    # Plot best attempts
    for solver, profile in profiles.items():
        ax1.step(
            np.round(tau_range, 3),
            profile,
            where="post",
            label=MODEL_SHORT_NAME_MAP.get(solver, solver),
            color=MODEL_COLOR_MAP[solver],
            linestyle=MODEL_MARKER_MAP[solver],
            linewidth=1.5,
        )

    xticks = np.arange(0, tau_range[-1] + 0.2, step=0.3).tolist()
    # xticks = np.arange(0, 1.7, step=0.4).tolist()
    xticks = np.round(xticks, 1)
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.set_xlabel(r"$\tau$", fontsize=10)
    ax1.set_ylabel(r"$P(\mathrm{ratio} \leq \tau)$", fontsize=10)
    ax1.set_title("Best Attempt Profile@4", fontsize=10)
    ax1.set_xticks(xticks, xticks, fontsize=8)
    yticks = np.round(np.arange(0, 1.2, 0.2), 1).tolist()
    ax1.set_yticks(yticks, yticks, fontsize=8)

    # Plot last attempts
    for solver, profile in last_profiles.items():
        ax2.step(
            np.round(tau_range, 3),
            profile,
            where="post",
            label=MODEL_SHORT_NAME_MAP.get(solver, solver),
            color=MODEL_COLOR_MAP[solver],
            linestyle=MODEL_MARKER_MAP[solver],
            linewidth=1.5,
        )

    xticks = np.arange(0, tau_range[-1] + 0.2, step=0.3).tolist()
    # xticks = np.arange(0, 1.7, step=0.4).tolist()
    xticks = np.round(xticks, 1)
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.set_xlabel(r"$\tau$", fontsize=10)
    ax2.set_title("Best Submission Profile@4", fontsize=10)
    ax2.set_xticks(xticks, xticks, fontsize=8)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([])

    # Add legend at the bottom
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, loc="best", fontsize=8, ncol=1)

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=500)
    plt.close()


def plot_performance_profiles(
    profiles: dict[str, np.ndarray], tau_range: np.ndarray, output_path: str, title: str
):
    """Plot performance profiles for all solvers using seaborn styling."""
    import matplotlib.pyplot as plt

    # Set the style and color palette
    # sns.set_style("darkgrid")
    colors = sns.color_palette("husl", n_colors=len(profiles))

    plt.figure(figsize=(4, 4))
    for (solver, profile), color in zip(profiles.items(), colors):
        plt.step(
            np.round(tau_range, 3),
            profile,
            where="post",
            label=MODEL_SHORT_NAME_MAP.get(solver, solver),
            color=color,
            linewidth=2.5,
        )

    xticks = list(range(0, int(tau_range[-1] + 1)))
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xlabel("Tau values (τ)", fontsize=8)
    plt.ylabel("P(ratio ≤ τ)", fontsize=8)
    plt.xticks(xticks, list(map(str, xticks)))
    # plt.title(title, fontsize=14, pad=20)

    # Customize legend
    # # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,
    #           frameon=True, fancybox=True, shadow=True)
    plt.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save with high DPI
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    # plt.savefig(output_path)
    plt.close()


def compute_aup(profile: np.ndarray, tau_range: np.ndarray) -> float:
    """
    Compute Area Under Performance curve for stepwise function.
    Each step spans from current tau to next tau value in log2 space.
    """
    total_area = 0.0

    # Calculate area for each step
    for i in range(len(tau_range) - 1):
        # Width of the step in log2 space
        step_width = tau_range[i + 1] - tau_range[i]
        # Height of the step (use current profile value as we have a step function)
        step_height = profile[i]
        # Add area of this step
        total_area += step_width * step_height

    return total_area


def compute_aup_trapezoidal(profile: np.ndarray, tau_range: np.ndarray) -> float:
    """Compute Area Under Performance curve using trapezoidal rule."""
    return np.trapezoid(profile, x=tau_range).item()


def round_up_to_one_decimal(arr):
    """Rounds up numbers in a NumPy array to 1 decimal place."""
    return np.ceil(arr * 10) / 10


def process_task_results(
    task_id: str, models: list[str], traj_parent_dir: str, traj_pattern: str
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Process results for a single task and return best and last attempt rankings."""
    best_scores = {}
    last_scores = {}
    task = TASKS[task_id]

    # Process model results
    for model in models:
        traj_dir_pattern = f"{traj_parent_dir}/*{model}__{task_id}__{traj_pattern}*"
        traj_dirs = sorted(list(Path().glob(traj_dir_pattern)))

        best_attempts = []
        last_attempts = []

        for traj_dir in traj_dirs:
            results_file = Path(traj_dir) / "results.json"
            if not results_file.exists():
                continue

            results = json.load(open(results_file))

            # Check for baseline score if we haven't found it yet
            if (
                "baseline" not in best_scores
                and "baseline" in results
                and task["priority_metric"] in results["baseline"]
            ):
                baseline_score = results["baseline"][task["priority_metric"]]
                best_scores["baseline"] = baseline_score
                last_scores["baseline"] = baseline_score

            if "agent" not in results:
                continue

            best_idx = get_best_attempt(
                results["agent"], task["priority_metric"], task["metric_direction"]
            )

            if best_idx != -1:
                best_attempts.append(
                    results["agent"][best_idx][task["priority_metric"]]
                )
                last_attempts.append(results["agent"][-1][task["priority_metric"]])

        if best_attempts:
            if task["metric_direction"] == "maximize":
                best_scores[model] = np.max(best_attempts)
                last_scores[model] = np.max(last_attempts)
            else:
                best_scores[model] = np.min(best_attempts)
                last_scores[model] = np.min(last_attempts)

    # Include baseline in the list of models for ranking
    all_models = models + ["baseline"]

    # Pass the metric direction to ensure proper ranking order
    best_rankings = calculate_rankings(
        best_scores, all_models, task["metric_direction"]
    )
    last_rankings = calculate_rankings(
        last_scores, all_models, task["metric_direction"]
    )

    return best_rankings, last_rankings


def save_scores_with_baseline(
    best_attempt_rankings: dict[str, list[tuple[str, float]]],
    last_attempt_rankings: dict[str, list[tuple[str, float]]],
    tasks: dict,
    output_file: str,
):
    """Save the actual scores and rankings in separate files."""
    # Get number of ranks based on number of models plus baseline
    n_ranks = len(MODELS) + 1  # Adding 1 for baseline
    ranks = range(1, n_ranks + 1)

    # Create DataFrames for rankings (using ranks as columns)
    best_ranks_df = pd.DataFrame(columns=["Rank"] + list(ranks))
    last_ranks_df = pd.DataFrame(columns=["Rank"] + list(ranks))

    # Create DataFrames for scores (using model names as columns)
    model_columns = ["baseline"] + MODELS
    best_scores_df = pd.DataFrame(columns=["Task"] + model_columns)
    last_scores_df = pd.DataFrame(columns=["Task"] + model_columns)

    # Process each task
    for task_id in tasks:
        task_name = tasks[task_id]["name"]
        # Convert rankings to dictionary for easier lookup
        best_scores = {
            model: round(score, 3) if not np.isnan(score) else score
            for model, score in best_attempt_rankings[task_id]
        }
        last_scores = {
            model: round(score, 3) if not np.isnan(score) else score
            for model, score in last_attempt_rankings[task_id]
        }

        # Get ordered models for rankings
        best_ordered_models = [model for model, _ in best_attempt_rankings[task_id]]
        last_ordered_models = [model for model, _ in last_attempt_rankings[task_id]]

        # Fill scores using model names as columns
        best_scores_row = [task_name] + [
            best_scores.get(model, float("nan")) for model in model_columns
        ]
        last_scores_row = [task_name] + [
            last_scores.get(model, float("nan")) for model in model_columns
        ]

        # Fill ranks using rank numbers as columns
        best_ranks_row = [task_name] + best_ordered_models
        last_ranks_row = [task_name] + last_ordered_models

        # Append rows
        best_scores_df.loc[len(best_scores_df)] = best_scores_row
        last_scores_df.loc[len(last_scores_df)] = last_scores_row
        best_ranks_df.loc[len(best_ranks_df)] = best_ranks_row
        last_ranks_df.loc[len(last_ranks_df)] = last_ranks_row

    # Add aggregate rankings
    best_pl = compute_plackett_luce_ranking(best_attempt_rankings)
    best_broda = compute_broda_ranking(best_attempt_rankings)
    last_pl = compute_plackett_luce_ranking(last_attempt_rankings)
    last_broda = compute_broda_ranking(last_attempt_rankings)

    best_ranks_df.loc[len(best_ranks_df)] = ["Plackett-Luce"] + best_pl
    best_ranks_df.loc[len(best_ranks_df)] = ["BORDA"] + best_broda
    last_ranks_df.loc[len(last_ranks_df)] = ["Plackett-Luce"] + last_pl
    last_ranks_df.loc[len(last_ranks_df)] = ["BORDA"] + last_broda

    # Save results
    best_scores_df.to_csv(f"results/tables/best_scores_{output_file}.csv", index=False)
    last_scores_df.to_csv(f"results/tables/last_scores_{output_file}.csv", index=False)
    best_ranks_df.to_csv(f"results/tables/best_ranks_{output_file}.csv", index=False)
    last_ranks_df.to_csv(f"results/tables/last_ranks_{output_file}.csv", index=False)


def save_performance_profiles(
    best_attempt_rankings: dict[str, list[tuple[str, float]]],
    last_attempt_rankings: dict[str, list[tuple[str, float]]],
    tasks: dict,
    all_models: list[str],
    output_prefix: str,
    traj_parent_dir: str,
    traj_pattern: str,
):
    """Calculate and save performance profiles and AUP scores."""
    # Generate tau range
    # tau_range = np.logspace(0, 1, base=2, num=20)
    # tau_range = np.array(range(1, 11))  # from 10^0 to 10^2
    # print(f"Tau range: {tseau_range}")

    # Calculate ratios for each task
    best_ratios = {}
    last_ratios = {}
    # ! We can also have separate suitable tau ranges for best and last attempts but I don't think the max tau is the issue here as long as all ratios are within the range (1, max_tau)
    suitable_tau = []

    for task_id, rankings in best_attempt_rankings.items():
        # Convert rankings to scores dictionary
        best_scores = {model: score for model, score in rankings}
        last_scores = {model: score for model, score in last_attempt_rankings[task_id]}

        # Calculate ratios
        best_ratios[task_id], tau = calculate_performance_ratios(
            best_scores, all_models, tasks[task_id]["metric_direction"], epsilon=0.05
        )
        suitable_tau.append(tau)

        last_ratios[task_id], tau = calculate_performance_ratios(
            last_scores, all_models, tasks[task_id]["metric_direction"], epsilon=0.05
        )
        suitable_tau.append(tau)

    # Compute profiles
    max_tau = round_up_to_one_decimal(np.log10(max(suitable_tau)))
    print(f"Max tau: {max_tau}")
    tau_range = np.linspace(0, max_tau, num=500)
    best_profiles = compute_performance_profile(best_ratios, all_models, tau_range)
    last_profiles = compute_performance_profile(last_ratios, all_models, tau_range)

    # Compute AUP for each model
    aup_scores = {
        model: {
            "Best AUP": round(compute_aup(best_profiles[model], tau_range), 3),
            "Last AUP": round(compute_aup(last_profiles[model], tau_range), 3),
        }
        for model in all_models
    }

    # Save AUP scores
    aup_df = pd.DataFrame.from_dict(aup_scores, orient="index")
    aup_df.index.name = "Model"
    aup_df.to_csv(f"results/tables/aup_scores_{output_prefix}.csv")

    plot_performance_profiles_dual(
        best_profiles,
        last_profiles,
        tau_range,
        "assets/figs/performance_profile.pdf",
        "Performance Profile (Best and Last Attempts)",
    )

    # ! keeping the cost analysis here for now because we are using AUP scores.
    plot_aup_vs_cost(
        aup_scores, traj_parent_dir, traj_pattern, "assets/figs/aup_vs_cost.pdf"
    )


def main(options: Options):
    # Store rankings for each task
    best_attempt_rankings = {}
    last_attempt_rankings = {}

    # Process each task
    for task_id in TASKS:
        best_rankings, last_rankings = process_task_results(
            task_id, options.models, options.traj_parent_dir, options.traj_pattern
        )
        best_attempt_rankings[task_id] = best_rankings
        last_attempt_rankings[task_id] = last_rankings

    # Save all results
    save_scores_with_baseline(
        best_attempt_rankings, last_attempt_rankings, TASKS, options.output_file
    )
    # Save performance profiles
    save_performance_profiles(
        best_attempt_rankings,
        last_attempt_rankings,
        TASKS,
        options.models + ["baseline"],
        options.output_file,
        options.traj_parent_dir,
        options.traj_pattern,
    )


if __name__ == "__main__":
    args = parse(Options)
    main(args)
