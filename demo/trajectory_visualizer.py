"""
Copyright (c) Meta Platforms, Inc. and affiliates.

MLGym Trajectory Visualizer

This module provides a Streamlit-based web application for visualizing MLGym trajectories.
It allows users to inspect step-by-step progression of agents through various ML tasks,
including their thought processes, actions taken, and execution results.

Usage:
    streamlit run trajectory_visualizer.py [--trajectory_dir PATH]
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import streamlit as st


def configure_page_style() -> None:
    """Configure the Streamlit page layout and apply custom CSS styling."""
    st.set_page_config(
        page_title="MLGym Demo",
        page_icon="👩‍🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
    <style>
        /* Base colors */
        :root {
            --slate-50: #f8fafc;
            --slate-100: #f1f5f9;
            --slate-200: #e2e8f0;
            --slate-300: #cbd5e1;
            --slate-400: #94a3b8;
            --slate-500: #64748b;
            --slate-600: #475569;
            --slate-700: #334155;
            --slate-800: #1e293b;
            --slate-900: #0f172a;
            --slate-950: #020617;
            --blue-500: #3b82f6;
            --purple-/00: #a855f7;
            --purple-500: #a855f7;
        }

        .stApp {
            background: linear-gradient(135deg, var(--slate-900) 0%, var(--slate-800) 100%);
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: var(--slate-50) !important;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin-bottom: 1.5rem;
        }

        /* Step indicator */
        .step-indicator {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--slate-50);
            margin: 2rem 0 1rem 0;
        }

        .step-caption {
            font-size: 1.25rem;
            color: var(--slate-300);
            margin-top: 0.5rem;
            margin-bottom: 1rem;
            font-weight: normal;
        }

        /* Content Boxes */
        .content-box {
            background: var(--slate-800);
            border: 1px solid var(--slate-700);
            border-radius: 1rem;
            padding: 2rem;
            margin-bottom: 2rem;
            width: 100%;
        }

        /* Add margin after progress bar */
        .stProgress {
            margin-bottom: 1rem;
        }

        .thought-box { border-top: 4px solid var(--blue-500); }
        .action-box { border-top: 4px solid var(--green-500); }
        .result-box { border-top: 4px solid var(--purple-500); }

        /* Headers */
        .box-header {
            font-size: 1.5rem !important;
            font-weight: 700;
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid var(--slate-700);
        }

        .thought-header { color: var(--blue-500) !important; }
        .action-header { color: var(--green-500) !important; }
        .result-header { color: var(--purple-500) !important; }

        /* Content styling */
        .box-content {
            padding: 0 1rem;
            font-size: 1.125rem;
            line-height: 1.75;
            color: var(--slate-200);
        }

        /* Sidebar styling */
        .sidebar-section {
            background: linear-gradient(to right, var(--slate-800) 0%, var(--slate-900) 100%);
            border-left: 6px solid var(--green-500);
            padding: 1.5rem;
            border-radius: 0 1rem 1rem 0;
            margin: 2rem 0;
        }

        /* Task card */
        .task-card {
            background: var(--slate-800);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--slate-700);
        }

        /* Content Summary Styling */
        .summary-box {
            background: var(--slate-800);
            border: 1px solid var(--slate-700);
            border-radius: 1rem;
            padding: 1rem;
            margin-bottom: 0.1em;
            width: 100%;
            font-size: 1.3rem;
        }
        
        /* Enhanced Sidebar Filter Styling */
        .sidebar .stExpander {
            background-color: var(--slate-800) !important;
            border-radius: 0.5rem !important;
            margin-bottom: 0.75rem !important;
        }
        
        .sidebar .stExpander > div:first-child {
            border-radius: 0.5rem !important;
            border: 1px solid var(--slate-700) !important;
            background-color: var(--slate-800) !important;
        }
        
        .sidebar .stExpander > div:last-child {
            background-color: var(--slate-850) !important;
            border-radius: 0 0 0.5rem 0.5rem !important;
            border: 1px solid var(--slate-700) !important;
            border-top: none !important;
            padding: 1rem !important;
        }
        
        /* Improved Checkbox Styling */
        .stCheckbox {
            position: relative;
            padding-left: 0;
            margin-bottom: 0.25rem;
        }
        
        .stCheckbox > div {
            display: flex !important;
            align-items: center !important;
        }
        
        .stCheckbox label {
            font-size: 0.9rem !important;
            color: var(--slate-300) !important;
            padding: 0.25rem 0 !important;
        }
        
        .stCheckbox label:hover {
            color: var(--slate-100) !important;
        }
        
        /* Trajectory Button Styling */
        button[data-testid="baseButton-secondary"] {
            background-color: var(--slate-800) !important;
            border: 1px solid var(--slate-700) !important;
            border-radius: 0.5rem !important;
            color: var(--slate-200) !important;
            font-family: monospace !important;
            text-align: left !important;
            white-space: pre-wrap !important;
            margin-bottom: 0.5rem !important;
            padding: 1rem !important;
            transition: all 0.2s ease !important;
            line-height: 1.5 !important;
        }
        
        button[data-testid="baseButton-secondary"]:hover {
            background-color: var(--slate-700) !important;
            border-color: var(--blue-500) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Search bar styling */
        .stTextInput > div > div > input {
            background-color: var(--slate-800) !important;
            border: 1px solid var(--slate-700) !important;
            color: var(--slate-200) !important;
            border-radius: 0.5rem !important;
            padding: 0.75rem 1rem !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: var(--blue-500) !important;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 0.5rem !important;
            padding: 0.5rem 1rem !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        
        .stButton > button:first-child {
            border-color: var(--blue-500) !important;
            background-color: var(--blue-500) !important;
            color: white !important;
        }
        
        .stButton > button:first-child:hover {
            background-color: var(--blue-600) !important;
            border-color: var(--blue-600) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Container for checkbox lists */
        .checkbox-container {
            max-height: 200px;
            overflow-y: auto;
            padding-right: 10px;
            margin-bottom: 10px;
        }
        
        .checkbox-container::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        
        .checkbox-container::-webkit-scrollbar-track {
            background: var(--slate-800);
            border-radius: 4px;
        }
        
        .checkbox-container::-webkit-scrollbar-thumb {
            background: var(--slate-600);
            border-radius: 4px;
        }
        
        .checkbox-container::-webkit-scrollbar-thumb:hover {
            background: var(--slate-500);
        }

        /* Highlight selected lines in trajectory display */
        .folder-line {
            color: var(--blue-400) !important;
            font-weight: 500;
        }
        
        .task-line {
            color: var(--green-400) !important;
            font-weight: 500;
        }
        
        .agent-line {
            color: var(--purple-400) !important;
            font-weight: 500;
        }
        
        .run-line {
            color: var(--slate-300) !important;
            font-weight: 400;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the trajectory visualizer.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="MLGym Trajectory Visualizer")
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default=os.path.join(os.getcwd(), "trajectories"),
        help="Directory containing trajectory files",
    )

    return parser.parse_known_args()[0]


def append_exit(content: Dict[str, Any]) -> Dict[str, Any]:
    """Append exit status and submission information to the content history.

    Args:
        content: Dictionary containing trajectory content and metadata

    Returns:
        Dict[str, Any]: Updated content dictionary with exit information

    Raises:
        ValueError: If submission is referenced but not found in content
    """
    last_entry = content["history"][-1]
    if last_entry["role"] == "system":
        return content

    exit_status = content.get("info", {}).get("exit_status")
    if not exit_status:
        return content

    if exit_status.startswith("submitted"):
        if "submission" in content["info"]:
            submission = content["info"]["submission"]
            content["history"].append(
                {
                    "role": "model_patch",
                    "content": submission,
                }
            )
        else:
            raise ValueError("No submission in history or info")
    return content


def format_metric_value(value: Optional[Union[int, float]]) -> str:
    """Format metric values for display with appropriate formatting.

    Args:
        value: Numeric value to format

    Returns:
        str: Formatted string representation of the value
    """
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return f"{value:,}"


def append_results(
    traj_path: Path,
    instance_id: str,
    content: Dict[str, Any],
    results: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Append evaluation results and statistics to the content history.

    Args:
        traj_path: Path to the trajectory file
        instance_id: Identifier for the current instance
        content: Content dictionary to update
        results: Results dictionary containing scores and metrics

    Returns:
        Dict[str, Any]: Updated content dictionary with results
    """
    stats: List[str] = []
    model_stats = {}
    exit_status = None

    # Load trajectory data and extract statistics
    if traj_path.exists():
        data = json.loads(traj_path.read_text())
        info = data.get("info", {})
        exit_status = info.get("exit_status")
        model_stats = info.get("model_stats", {})

    # Format model statistics
    instance_cost = format_metric_value(model_stats.get("total_cost"))
    tokens_sent = format_metric_value(model_stats.get("tokens_sent"))
    tokens_received = format_metric_value(model_stats.get("tokens_received"))
    api_calls = format_metric_value(model_stats.get("api_calls"))

    # Build statistics report
    stats.extend(
        [
            "*" * 39,
            "Run Stats",
            "*" * 39,
            f"Instance Cost: ${instance_cost}",
            f"Tokens Sent: {tokens_sent}",
            f"Tokens Received: {tokens_received}",
            f"API Calls: {api_calls}",
            f"Exit Status: {exit_status}",
        ]
    )

    # Process and format results
    status = process_results(results)

    # Create and insert evaluation report
    eval_report = {
        "role": "Evaluation Report",
        "content": "\n".join([*stats, *status]),
    }
    content["history"].insert(0, eval_report)
    content["history"].append(eval_report)

    return content


def process_results(results: Optional[Dict[str, Any]]) -> List[str]:
    """Process and format evaluation results for display.

    Args:
        results: Dictionary containing evaluation results

    Returns:
        List[str]: Formatted status messages for display
    """
    if not results:
        return ["No scores found"]

    agent_results = results.get("agent")
    baseline_results = results.get("baseline")

    if not agent_results and not baseline_results:
        return ["Baseline and Agent scores not found"]

    status = []

    if baseline_results and agent_results:
        status.extend(
            [
                "*" * 39,
                "Agent vs Baseline Scores",
                "*" * 39,
            ]
        )

        formatted_scores = defaultdict(dict)
        for score_type, score in baseline_results.items():
            formatted_scores[score_type]["Baseline"] = score

        for i, agent_score in enumerate(agent_results):
            for score_type, score in agent_score.items():
                formatted_scores[score_type][f"Attempt {i + 1}"] = score

        for score_type, scores in formatted_scores.items():
            status.append(f"Metric: {score_type}")
            status.extend(f"  {model}: {score:.3f}" for model, score in scores.items())

    elif baseline_results:
        status.append("**** Baseline Scores ****")
        status.extend(
            f"  {score_type}: {score}" for score_type, score in baseline_results.items()
        )

    elif agent_results:
        status.append("**** Agent Scores ****")
        status.extend(
            f"  {score_type}: {score}" for score_type, score in agent_results.items()
        )

    return status


def load_results(results_path: Path) -> Optional[Dict[str, Any]]:
    """Load results from a JSON file.

    Args:
        results_path: Path to the results file

    Returns:
        Optional[Dict[str, Any]]: Loaded results or None if file not found
    """
    if not results_path.exists():
        return None

    with open(results_path) as infile:
        return json.load(infile)


def load_content(file_name: str) -> Dict[str, Any]:
    """Load and process trajectory content from a file.

    Args:
        file_name: Path to the trajectory file

    Returns:
        Dict[str, Any]: Processed content with results and exit information
    """
    with open(file_name) as infile:
        content = json.load(infile)

    results_file = Path(file_name).parent / "results.json"
    results = load_results(results_file)

    content = append_exit(content)
    return append_results(
        Path(file_name),
        Path(file_name).stem,
        content,
        results,
    )


def parse_trajectory_filename(filepath: str) -> Dict[str, str]:
    """Parse a trajectory filename into its components.

    Args:
        filepath: Path to the trajectory file

    Returns:
        Dict[str, str]: Dictionary with parsed components
    """
    # Get the path components
    path = Path(filepath)
    filename = path.name

    # Extract base folder (one level up from the .traj file's parent)
    parts = path.parts
    base_folder = parts[-3] if len(parts) >= 3 else ""

    # Parse the parent directory name which contains metadata
    parent_dir = path.parent.name

    # Initialize with default values
    parsed = {
        "folder": base_folder,
        "task": "",
        "model": "",
        "temperature": "0.0",
        "top_p": "0.0",
        "cost_limit": "0.0",
        "install": "0",
        "suffix": "",
        "run_number": "0",
        "filepath": filepath,
        "filename": filename,
    }

    # Extract task name from the .traj filename
    if filename.endswith(".traj"):
        parsed["task"] = filename[:-5]  # Remove .traj extension

    # Handle different patterns for parent directory names
    # Pattern for full format: model__task__parser__t-temp__p-prob__c-cost__install-N__suffix(_run_N)?
    full_pattern = r"^(?:(?:submit|meta(?:gen)?)?[-_]?)?([^_]+)__([^_]+)__([^_]+)__t-([^_]+)__p-([^_]+)__c-([^_]+)__install-([^_]+)(?:__(.+?))?(?:_run_(\d+))?$"

    # Pattern for human format: human__task__parser__t-temp__p-prob__c-cost__install-N(_run_N)?
    human_pattern = r"^(human)__([^_]+)__([^_]+)__t-([^_]+)__p-([^_]+)__c-([^_]+)__install-([^_]+)(?:_run_(\d+))?$"

    # Try full pattern first
    match = re.match(full_pattern, parent_dir)
    if match:
        parsed["model"] = match.group(1)
        parsed["task"] = match.group(2)
        parsed["parser"] = match.group(3)
        parsed["temperature"] = match.group(4)
        parsed["top_p"] = match.group(5)
        parsed["cost_limit"] = match.group(6)
        parsed["install"] = match.group(7)
        parsed["suffix"] = match.group(8) if match.group(8) else ""
        parsed["run_number"] = match.group(9) if match.group(9) else "0"
        return parsed

    # Try human pattern
    match = re.match(human_pattern, parent_dir)
    if match:
        parsed["model"] = match.group(1)
        parsed["task"] = match.group(2)
        parsed["parser"] = match.group(3)
        parsed["temperature"] = match.group(4)
        parsed["top_p"] = match.group(5)
        parsed["cost_limit"] = match.group(6)
        parsed["install"] = match.group(7)
        parsed["suffix"] = ""
        parsed["run_number"] = match.group(8) if match.group(8) else "0"
        return parsed

    # Fallback to a more lenient approach if none of the patterns match
    parts = parent_dir.split("__")
    if len(parts) >= 3:
        parsed["model"] = parts[0]
        parsed["task"] = parts[1]

        # Try to extract other parameters if they exist
        for i, part in enumerate(parts[2:], 3):
            if part.startswith("t-"):
                parsed["temperature"] = part[2:]
            elif part.startswith("p-"):
                parsed["top_p"] = part[2:]
            elif part.startswith("c-"):
                parsed["cost_limit"] = part[2:]
            elif part.startswith("install-"):
                parsed["install"] = part[8:]
            elif "_run_" in part:
                run_parts = part.split("_run_")
                if len(run_parts) > 1:
                    parsed["suffix"] = run_parts[0]
                    parsed["run_number"] = run_parts[1]
            elif i == 3:  # Third position is typically parser
                parsed["parser"] = part
            elif i > 6:  # Later positions might be suffixes
                parsed["suffix"] += part + "_"

        # Clean up suffix
        parsed["suffix"] = parsed["suffix"].rstrip("_")

    return parsed


def find_trajectory_files(root_dir: str) -> List[Dict[str, Any]]:
    """Recursively find all trajectory files in the given directory.

    Args:
        root_dir: Root directory to search for trajectory files

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing structured file information
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        st.error(f"Directory not found: {root_dir}")
        return []

    trajectories = []

    for file in root_path.rglob("*.traj"):
        filepath = str(file.absolute())
        parsed_info = parse_trajectory_filename(filepath)

        # Format display string - using newlines instead of HTML for display in buttons
        run_details = f"Run {parsed_info['run_number']}"
        if parsed_info["suffix"]:
            run_details = f"{parsed_info['suffix']} {run_details}"

        display_info = (
            f"Folder: {parsed_info['folder']}\n"
            f"Task: {parsed_info['task']}\n"
            f"Agent: {parsed_info['model']}, t-{parsed_info['temperature']}, "
            f"p-{parsed_info['top_p']}, c-{parsed_info['cost_limit']}\n"
            f"Run Details: {run_details}"
        )

        trajectories.append(
            {
                **parsed_info,
                "display_info": display_info,
            }
        )

    # Sort by folder, model, suffix, and run_number (with run_number as int)
    return sorted(
        trajectories,
        key=lambda x: (
            x["folder"],
            x["task"],
            x["model"],
            x["suffix"],
            int(x["run_number"]),
        ),
    )


def filter_trajectories(
    trajectories: List[Dict[str, Any]], filters: Dict[str, List[str]], search_query: str
) -> List[Dict[str, Any]]:
    """Filter trajectories based on selected filter options and search query.

    Args:
        trajectories: List of trajectory dictionaries
        filters: Dictionary mapping filter names to lists of selected options
        search_query: Text to search for

    Returns:
        List[Dict[str, Any]]: Filtered list of trajectories
    """
    if not trajectories:
        return []

    # Apply filters (AND logic across different filters)
    filtered_trajectories = trajectories.copy()

    for filter_name, selected_options in filters.items():
        if not selected_options:  # Skip if no options selected for this filter
            continue

        # Apply OR logic within each filter type
        filtered_trajectories = [
            traj
            for traj in filtered_trajectories
            if traj[filter_name] in selected_options
        ]

    # Apply search if provided (search across all fields)
    if search_query:
        search_query = search_query.lower()
        filtered_trajectories = [
            traj
            for traj in filtered_trajectories
            if (
                search_query in traj["display_info"].lower()
                or search_query in traj["filepath"].lower()
            )
        ]

    return filtered_trajectories


def load_trajectory(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """Load trajectory data from a JSON file.

    Args:
        file_path: Path to the trajectory file

    Returns:
        Optional[List[Dict[str, Any]]]: Loaded trajectory data or None if error occurs
    """
    try:
        with open(file_path, "r") as file:
            return json.load(file)["trajectory"]
    except FileNotFoundError:
        st.error(f"Trajectory file not found: {file_path}")
    except json.JSONDecodeError:
        st.error(f"Invalid JSON in trajectory file: {file_path}")
    except KeyError:
        st.error(f"Missing 'trajectory' key in file: {file_path}")
    return None


def display_content_summary(content: str) -> None:
    """Display a collapsible summary of content.

    Args:
        content: Content to display in the summary
    """
    if not content:
        return

    st.markdown(
        f"""
        <div class="summary-box">
            <details class="content-summary">
                <summary class="summary-header">
                    📝 Evaluation Report
                </summary>
                <div class="summary-content">
                    <pre><code>{content}</code></pre>
                </div>
            </details>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_step(step_data: Dict[str, Any], step_num: int, total_steps: int) -> None:
    """Display a single step of the trajectory.

    Args:
        step_data: Data for the current step
        step_num: Current step number
        total_steps: Total number of steps
    """
    # Step indicator
    st.markdown(
        f"""
        <div class="step-indicator">Step {step_num + 1} / {total_steps}</div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.view_mode == "step":
        # Navigation controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "⬅️ Previous Step",
                key="prev_step",
                disabled=step_num == 0,
                use_container_width=True,
            ):
                st.session_state.current_step -= 1
                st.rerun()

        with col2:
            if st.button(
                "Next Step ➡️",
                key="next_step",
                disabled=step_num == total_steps - 1,
                use_container_width=True,
            ):
                st.session_state.current_step += 1
                st.rerun()

        # Progress tracking
        st.progress((step_num + 1) / total_steps)
        st.markdown(
            f"""
            <div class="step-caption">{step_data.get("caption", "")}</div>
            """,
            unsafe_allow_html=True,
        )

    # Display step components
    display_step_components(step_data)


def display_step_components(step_data: Dict[str, Any]) -> None:
    """Display the individual components of a step using pure Streamlit with custom styling."""
    # Inject custom CSS for colored borders and headers
    st.markdown(
        """
        <style>
        /* Custom container styling for specific sections */
        .thought-header {
            font-weight: 700 !important;
            color: #3b82f6 !important;
            font-size: 1.5rem !important;
            margin-bottom: 0.75rem !important;
            padding-bottom: 0.5rem !important;
            border-bottom: 2px solid #3b82f6 !important;
        }
        
        .action-header {
            font-weight: 700 !important;
            color: #10b981 !important;
            font-size: 1.5rem !important;
            margin-bottom: 0.75rem !important;
            padding-bottom: 0.5rem !important;
            border-bottom: 2px solid #10b981 !important;
        }
        
        .result-header {
            font-weight: 700 !important;
            color: #a855f7 !important;
            font-size: 1.5rem !important;
            margin-bottom: 0.75rem !important;
            padding-bottom: 0.5rem !important;
            border-bottom: 2px solid #a855f7 !important;
        }
        
        /* Style container borders */
        .blue-border {
            border-left: 4px solid #3b82f6 !important;
            border-radius: 0.5rem !important;
            padding-left: 1rem !important;
            margin-bottom: 2rem !important;
            background-color: #1e293b !important;
        }
        
        .green-border {
            border-left: 4px solid #10b981 !important;
            border-radius: 0.5rem !important;
            padding-left: 1rem !important;
            margin-bottom: 2rem !important;
            background-color: #1e293b !important;
        }
        
        .purple-border {
            border-left: 4px solid #a855f7 !important;
            border-radius: 0.5rem !important;
            padding-left: 1rem !important;
            margin-bottom: 2rem !important;
            background-color: #1e293b !important;
        }
        
        /* Style the code blocks - remove all syntax highlighting */
        .stCodeBlock div[data-baseweb="block"] {
            background-color: transparent !important;
        }
        
        .stCodeBlock pre {
            background-color: transparent !important;
            padding: 0 !important;
        }
        
        .stCodeBlock code {
            color: var(--slate-100) !important;
            background-color: transparent !important;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # THOUGHT PROCESS SECTION
    # Create header
    st.markdown(
        '<div class="thought-header">💭 Thought Process</div>', unsafe_allow_html=True
    )

    # Apply the blue border class to the container
    st.markdown('<div class="blue-border">', unsafe_allow_html=True)

    # Content
    st.write(step_data["thought"].replace("DISCUSSION", ""))

    # Close the div
    st.markdown("</div>", unsafe_allow_html=True)

    # ACTION SECTION
    # Create header
    st.markdown(
        '<div class="action-header">🤖 Action Taken</div>', unsafe_allow_html=True
    )

    # Apply the green border class to the container
    st.markdown('<div class="green-border">', unsafe_allow_html=True)

    # Content
    st.code(step_data["action"], language="python")

    # Close the div
    st.markdown("</div>", unsafe_allow_html=True)

    # RESULT SECTION
    # Create header
    st.markdown(
        '<div class="result-header">💻 Execution Result</div>', unsafe_allow_html=True
    )

    # Apply the purple border class to the container
    st.markdown('<div class="purple-border">', unsafe_allow_html=True)

    # Content
    st.code(step_data["observation"], language="bash")

    # Close the div
    st.markdown("</div>", unsafe_allow_html=True)


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables for app state management."""
    # Initialize any missing states with default values
    if "current_trajectory" not in st.session_state:
        st.session_state.current_trajectory = None

    if "current_step" not in st.session_state:
        st.session_state.current_step = 1

    if "highlighted_trajectory" not in st.session_state:
        st.session_state.highlighted_trajectory = None

    # Filter management
    # Non-reactive versions of the filter state to prevent reruns on checkbox interactions
    if "temp_selected_folders" not in st.session_state:
        st.session_state.temp_selected_folders = []

    if "temp_selected_tasks" not in st.session_state:
        st.session_state.temp_selected_tasks = []

    if "temp_selected_models" not in st.session_state:
        st.session_state.temp_selected_models = []

    if "temp_selected_suffixes" not in st.session_state:
        st.session_state.temp_selected_suffixes = []

    # The actual filter state that affects rendering
    if "selected_folders" not in st.session_state:
        st.session_state.selected_folders = []

    if "selected_tasks" not in st.session_state:
        st.session_state.selected_tasks = []

    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []

    if "selected_suffixes" not in st.session_state:
        st.session_state.selected_suffixes = []

    # Actual filter lists used for filtering
    if "filter_folders" not in st.session_state:
        st.session_state.filter_folders = []
    if "filter_tasks" not in st.session_state:
        st.session_state.filter_tasks = []
    if "filter_models" not in st.session_state:
        st.session_state.filter_models = []
    if "filter_suffixes" not in st.session_state:
        st.session_state.filter_suffixes = []

    # Temporary filter selections that don't trigger reruns
    if "temp_filter_folders" not in st.session_state:
        st.session_state.temp_filter_folders = []
    if "temp_filter_tasks" not in st.session_state:
        st.session_state.temp_filter_tasks = []
    if "temp_filter_models" not in st.session_state:
        st.session_state.temp_filter_models = []
    if "temp_filter_suffixes" not in st.session_state:
        st.session_state.temp_filter_suffixes = []

    # Add a flag to track if filters have been initialized
    if "filter_initialized" not in st.session_state:
        st.session_state.filter_initialized = False

    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    if "filtered_trajectories" not in st.session_state:
        st.session_state.filtered_trajectories = []
    if "apply_filters_clicked" not in st.session_state:
        st.session_state.apply_filters_clicked = False


def display_welcome_message() -> None:
    """Display the welcome message when no trajectory is selected."""
    st.markdown(
        """
        <div style='text-align: center; padding: 4rem 2rem;'>
            <h1>👋 Welcome to the MLGym Demo</h1>
            <p style='font-size: 1.2rem; color: #e0e0e0; margin: 2rem 0;'>
                Select a task from the sidebar to view the MLGym Agent's trajectory.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def setup_sidebar(args: argparse.Namespace) -> None:
    """Set up the sidebar with trajectory selection options and filters.

    Args:
        args: Command line arguments containing trajectory directory
    """
    with st.sidebar:
        st.markdown("# 👩‍🔬 MLGym Agent")
        st.markdown("### Select Trajectory")
        st.markdown(f"**Current Directory:** {args.trajectory_dir}")

        # Fetch all trajectories
        all_trajectories = find_trajectory_files(args.trajectory_dir)

        if not all_trajectories:
            st.warning("No trajectory files found in the specified directory.")
            return

        # Extract unique filter options - don't filter out any values
        unique_folders = sorted(set(traj["folder"] for traj in all_trajectories))
        unique_tasks = sorted(set(traj["task"] for traj in all_trajectories))
        unique_models = sorted(set(traj["model"] for traj in all_trajectories))
        unique_suffixes = sorted(set(traj["suffix"] for traj in all_trajectories))

        # Store the unique lists for callbacks
        st.session_state.all_folders_list = unique_folders
        st.session_state.all_tasks_list = unique_tasks
        st.session_state.all_models_list = unique_models
        st.session_state.all_suffixes_list = unique_suffixes

        # Initialize filter selections only once
        if not st.session_state.filter_initialized:
            st.session_state.temp_filter_folders = unique_folders.copy()
            st.session_state.filter_folders = unique_folders.copy()
            st.session_state.temp_filter_tasks = unique_tasks.copy()
            st.session_state.filter_tasks = unique_tasks.copy()
            st.session_state.temp_filter_models = unique_models.copy()
            st.session_state.filter_models = unique_models.copy()
            st.session_state.temp_filter_suffixes = unique_suffixes.copy()
            st.session_state.filter_suffixes = unique_suffixes.copy()
            st.session_state.filter_initialized = True

        # Search bar
        st.text_input(
            "Search trajectories", key="search_query", placeholder="Type to search..."
        )

        # Filter sections
        folders_expander = st.expander(
            f"🗂️ Filter by Folder ({len(st.session_state.filter_folders)}/{len(unique_folders)})",
            expanded=False,
        )

        with folders_expander:
            # All/None checkboxes for folders
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Select All", key="all_folders_btn"):
                    select_all_folders()
            with col2:
                if st.button("Clear All", key="none_folders_btn"):
                    clear_all_folders()

            # Create a container for checkboxes with scrolling
            st.markdown('<div class="checkbox-container">', unsafe_allow_html=True)

            # Individual checkboxes for each folder
            for folder in unique_folders:
                folder_selected = folder in st.session_state.temp_filter_folders
                checkbox_key = f"folder_{folder}"
                # Use a checkbox but handle its state manually
                if st.checkbox(
                    folder,
                    value=folder_selected,
                    key=checkbox_key,
                    on_change=toggle_folder,
                    args=(folder,),
                ):
                    pass  # State is handled in the on_change callback

            st.markdown("</div>", unsafe_allow_html=True)

        tasks_expander = st.expander(
            f"📋 Filter by Task ({len(st.session_state.filter_tasks)}/{len(unique_tasks)})",
            expanded=False,
        )

        with tasks_expander:
            # All/None checkboxes for tasks
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Select All", key="all_tasks_btn"):
                    select_all_tasks()
            with col2:
                if st.button("Clear All", key="none_tasks_btn"):
                    clear_all_tasks()

            # Create a container for checkboxes with scrolling
            st.markdown('<div class="checkbox-container">', unsafe_allow_html=True)

            # Individual checkboxes for each task
            for task in unique_tasks:
                task_selected = task in st.session_state.temp_filter_tasks
                checkbox_key = f"task_{task}"
                # Use a checkbox but handle its state manually
                if st.checkbox(
                    task,
                    value=task_selected,
                    key=checkbox_key,
                    on_change=toggle_task,
                    args=(task,),
                ):
                    pass  # State is handled in the on_change callback

            st.markdown("</div>", unsafe_allow_html=True)

        models_expander = st.expander(
            f"🤖 Filter by Model ({len(st.session_state.filter_models)}/{len(unique_models)})",
            expanded=False,
        )

        with models_expander:
            # All/None checkboxes for models
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Select All", key="all_models_btn"):
                    select_all_models()
            with col2:
                if st.button("Clear All", key="none_models_btn"):
                    clear_all_models()

            # Create a container for checkboxes with scrolling
            st.markdown('<div class="checkbox-container">', unsafe_allow_html=True)

            # Individual checkboxes for each model
            for model in unique_models:
                model_selected = model in st.session_state.temp_filter_models
                checkbox_key = f"model_{model}"
                # Use a checkbox but handle its state manually
                if st.checkbox(
                    model,
                    value=model_selected,
                    key=checkbox_key,
                    on_change=toggle_model,
                    args=(model,),
                ):
                    pass  # State is handled in the on_change callback

            st.markdown("</div>", unsafe_allow_html=True)

        suffixes_expander = st.expander(
            f"🔄 Filter by Run Details ({len(st.session_state.filter_suffixes)}/{len(unique_suffixes)})",
            expanded=False,
        )

        with suffixes_expander:
            # All/None checkboxes for suffixes
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Select All", key="all_suffixes_btn"):
                    select_all_suffixes()
            with col2:
                if st.button("Clear All", key="none_suffixes_btn"):
                    clear_all_suffixes()

            # Create a container for checkboxes with scrolling
            st.markdown('<div class="checkbox-container">', unsafe_allow_html=True)

            # Individual checkboxes for each suffix
            for suffix in unique_suffixes:
                suffix_selected = suffix in st.session_state.temp_filter_suffixes
                checkbox_key = f"suffix_{suffix}"
                # Use a checkbox but handle its state manually
                if st.checkbox(
                    suffix,
                    value=suffix_selected,
                    key=checkbox_key,
                    on_change=toggle_suffix,
                    args=(suffix,),
                ):
                    pass  # State is handled in the on_change callback

            st.markdown("</div>", unsafe_allow_html=True)

        # Apply and Reset buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply Filters", use_container_width=True):
                apply_filters(all_trajectories)
                st.rerun()

        with col2:
            if st.button("Reset Filters", use_container_width=True):
                st.session_state.temp_filter_folders = unique_folders.copy()
                st.session_state.temp_filter_tasks = unique_tasks.copy()
                st.session_state.temp_filter_models = unique_models.copy()
                st.session_state.temp_filter_suffixes = unique_suffixes.copy()
                # Force reinitialization on next run
                st.session_state.filter_initialized = False
                apply_filters(all_trajectories)
                st.rerun()

        # Display filtered trajectories
        if st.session_state.apply_filters_clicked:
            if not st.session_state.filtered_trajectories:
                st.markdown(
                    """
                <div style="background-color: rgba(255, 171, 0, 0.2); 
                           border-left: 4px solid #ffab00; 
                           padding: 1rem; 
                           border-radius: 0.5rem; 
                           margin: 1rem 0;">
                    <p style="color: #ffab00; font-weight: 600; margin: 0;">No matches found</p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #cbd5e1;">
                        No trajectories match your current filter selections. Try adjusting your filters or search query.
                    </p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"### Showing {len(st.session_state.filtered_trajectories)} trajectories"
                )

                for i, trajectory in enumerate(st.session_state.filtered_trajectories):
                    # Create a well-formatted run details string
                    run_details = f"Run {trajectory['run_number']}"
                    if trajectory["suffix"]:
                        run_details = f"{trajectory['suffix']} {run_details}"

                    # Simple but efficient display format
                    display_text = (
                        f"Folder: {trajectory['folder']} | "
                        f"Task: {trajectory['task']} | "
                        f"Agent: {trajectory['model']} | "
                        f"Run: {run_details}"
                    )

                    # Check if this is the currently selected trajectory
                    is_selected = (
                        "current_trajectory" in st.session_state
                        and st.session_state.current_trajectory
                        == trajectory["filepath"]
                    )

                    # Apply different styling for selected trajectory
                    button_style = {}
                    if is_selected:
                        st.markdown(
                            f"""
                            <div style="background-color: var(--blue-500); color: white; 
                                       border-radius: 0.5rem; padding: 0.5rem; margin-bottom: 0.5rem;
                                       border: 2px solid white; font-weight: bold;">
                                ✓ {display_text}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        # Use a simple button for better performance
                        if st.button(
                            display_text,
                            key=f"traj_{i}",
                            help=trajectory["filepath"],
                            use_container_width=True,
                        ):
                            st.session_state.current_trajectory = trajectory["filepath"]
                            st.session_state.view_mode = "step"
                            st.session_state.current_step = 0
                            st.rerun()
        else:
            # Initial load - show all trajectories
            filters = {
                "folder": unique_folders,
                "task": unique_tasks,
                "model": unique_models,
                "suffix": unique_suffixes,
            }

            st.session_state.filtered_trajectories = filter_trajectories(
                all_trajectories, filters, ""
            )

            st.markdown(
                f"### Showing {len(st.session_state.filtered_trajectories)} trajectories"
            )

            for i, trajectory in enumerate(st.session_state.filtered_trajectories):
                # Create a well-formatted run details string
                run_details = f"Run {trajectory['run_number']}"
                if trajectory["suffix"]:
                    run_details = f"{trajectory['suffix']} {run_details}"

                # Simple but efficient display format
                display_text = (
                    f"Folder: {trajectory['folder']} | "
                    f"Task: {trajectory['task']} | "
                    f"Agent: {trajectory['model']} | "
                    f"Run: {run_details}"
                )

                # Check if this is the currently selected trajectory
                is_selected = (
                    "current_trajectory" in st.session_state
                    and st.session_state.current_trajectory == trajectory["filepath"]
                )

                # Apply different styling for selected trajectory
                if is_selected:
                    st.markdown(
                        f"""
                        <div style="background-color: var(--blue-500); color: white; 
                                   border-radius: 0.5rem; padding: 0.5rem; margin-bottom: 0.5rem;
                                   border: 2px solid white; font-weight: bold;">
                            ✓ {display_text}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    # Use a simple button for better performance
                    if st.button(
                        display_text,
                        key=f"traj_{i}",
                        help=trajectory["filepath"],
                        use_container_width=True,
                    ):
                        st.session_state.current_trajectory = trajectory["filepath"]
                        st.session_state.view_mode = "step"
                        st.session_state.current_step = 0
                        st.rerun()


def display_trajectory_content(args: argparse.Namespace) -> None:
    """Display the selected trajectory content and visualization."""
    if (
        "current_trajectory" not in st.session_state
        or st.session_state.current_trajectory is None
    ):
        # If no trajectory is selected, show welcome message
        display_welcome_message()
        setup_sidebar(args)
        return

    st.title("👩‍🔬 MLGym Agent")

    # Safety checks before loading data
    if not os.path.exists(st.session_state.current_trajectory):
        st.error(f"Trajectory file not found: {st.session_state.current_trajectory}")
        setup_sidebar(args)
        return

    # Load trajectory data
    data = load_trajectory(st.session_state.current_trajectory)
    if not data:
        setup_sidebar(args)
        return

    # Load content safely
    try:
        content = load_content(st.session_state.current_trajectory)
        if (
            "history" in content
            and content["history"]
            and isinstance(content["history"][0], dict)
        ):
            display_content_summary(content["history"][0].get("content", ""))
    except Exception as e:
        st.error(f"Error loading trajectory content: {str(e)}")
        setup_sidebar(args)
        return

    # Set up sidebar
    setup_sidebar(args)

    # Display trajectory steps
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "step"

    if st.session_state.view_mode == "full":
        for i, step_data in enumerate(data):
            display_step(step_data, i, len(data))
    else:
        current_step = st.session_state.current_step
        if current_step < len(data):
            display_step(data[current_step], current_step, len(data))
        else:
            st.warning(
                f"Invalid step index: {current_step}. Maximum is {len(data) - 1}."
            )
            st.session_state.current_step = 0
            st.rerun()


def toggle_folder(folder: str) -> None:
    """Toggle a folder selection without triggering a rerun."""
    # Create a temporary copy
    temp_folders = st.session_state.temp_filter_folders.copy()

    # Modify the copy
    if folder in temp_folders:
        temp_folders.remove(folder)
    else:
        temp_folders.append(folder)

    # Update session state and break reactivity chain
    st.session_state.temp_filter_folders = temp_folders
    # This line interrupts the clean-up process
    st.session_state.temp_filter_folders = st.session_state.temp_filter_folders


def toggle_task(task: str) -> None:
    """Toggle a task selection without triggering a rerun."""
    temp_tasks = st.session_state.temp_filter_tasks.copy()
    if task in temp_tasks:
        temp_tasks.remove(task)
    else:
        temp_tasks.append(task)
    st.session_state.temp_filter_tasks = temp_tasks
    st.session_state.temp_filter_tasks = st.session_state.temp_filter_tasks


def toggle_model(model: str) -> None:
    """Toggle a model selection without triggering a rerun."""
    temp_models = st.session_state.temp_filter_models.copy()
    if model in temp_models:
        temp_models.remove(model)
    else:
        temp_models.append(model)
    st.session_state.temp_filter_models = temp_models
    st.session_state.temp_filter_models = st.session_state.temp_filter_models


def toggle_suffix(suffix: str) -> None:
    """Toggle a suffix selection without triggering a rerun."""
    temp_suffixes = st.session_state.temp_filter_suffixes.copy()
    if suffix in temp_suffixes:
        temp_suffixes.remove(suffix)
    else:
        temp_suffixes.append(suffix)
    st.session_state.temp_filter_suffixes = temp_suffixes
    st.session_state.temp_filter_suffixes = st.session_state.temp_filter_suffixes


def select_all_folders() -> None:
    """Select all folders without triggering a rerun."""
    folders_copy = st.session_state.all_folders_list.copy()
    st.session_state.temp_filter_folders = folders_copy
    st.session_state.temp_filter_folders = st.session_state.temp_filter_folders


def clear_all_folders() -> None:
    """Clear all folder selections without triggering a rerun."""
    st.session_state.temp_filter_folders = []
    st.session_state.temp_filter_folders = st.session_state.temp_filter_folders


def select_all_tasks() -> None:
    """Select all tasks without triggering a rerun."""
    tasks_copy = st.session_state.all_tasks_list.copy()
    st.session_state.temp_filter_tasks = tasks_copy
    st.session_state.temp_filter_tasks = st.session_state.temp_filter_tasks


def clear_all_tasks() -> None:
    """Clear all task selections without triggering a rerun."""
    st.session_state.temp_filter_tasks = []
    st.session_state.temp_filter_tasks = st.session_state.temp_filter_tasks


def select_all_models() -> None:
    """Select all models without triggering a rerun."""
    models_copy = st.session_state.all_models_list.copy()
    st.session_state.temp_filter_models = models_copy
    st.session_state.temp_filter_models = st.session_state.temp_filter_models


def clear_all_models() -> None:
    """Clear all model selections without triggering a rerun."""
    st.session_state.temp_filter_models = []
    st.session_state.temp_filter_models = st.session_state.temp_filter_models


def select_all_suffixes() -> None:
    """Select all suffixes without triggering a rerun."""
    suffixes_copy = st.session_state.all_suffixes_list.copy()
    st.session_state.temp_filter_suffixes = suffixes_copy
    st.session_state.temp_filter_suffixes = st.session_state.temp_filter_suffixes


def clear_all_suffixes() -> None:
    """Clear all suffix selections without triggering a rerun."""
    st.session_state.temp_filter_suffixes = []
    st.session_state.temp_filter_suffixes = st.session_state.temp_filter_suffixes


def apply_filters(all_trajectories: List[Dict[str, Any]]) -> None:
    """Apply the temporary filters to the real filter lists and update filtered trajectories.

    Args:
        all_trajectories: All available trajectories
    """
    # Transfer temporary selections to the real filter lists
    st.session_state.filter_folders = st.session_state.temp_filter_folders.copy()
    st.session_state.filter_tasks = st.session_state.temp_filter_tasks.copy()
    st.session_state.filter_models = st.session_state.temp_filter_models.copy()
    st.session_state.filter_suffixes = st.session_state.temp_filter_suffixes.copy()

    # Apply filters
    filters = {
        "folder": st.session_state.filter_folders,
        "task": st.session_state.filter_tasks,
        "model": st.session_state.filter_models,
        "suffix": st.session_state.filter_suffixes,
    }

    st.session_state.filtered_trajectories = filter_trajectories(
        all_trajectories, filters, st.session_state.search_query
    )
    st.session_state.apply_filters_clicked = True


def main() -> None:
    """Main entry point for the MLGym Trajectory Visualizer application."""
    configure_page_style()
    initialize_session_state()  # Initialize session state
    args = parse_args()
    display_trajectory_content(args)


if __name__ == "__main__":
    main()
