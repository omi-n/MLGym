#!/bin/bash

# Simple MLGym experiments runner
set -euo pipefail

# Configuration
SEEDS=(1000 1001 1002)
TASKS=(
    "battleOfSexes.yaml"
    "blotto.yaml"
    "imageClassificationCifar10.yaml"
    "imageClassificationCifar10L1.yaml"
    "imageClassificationFMnist.yaml"
    "prisonersDilemma.yaml"
    "regressionKaggleHousePrice.yaml"
    "regressionKaggleHousePriceL1.yaml"
    "rlMetaMaze.yaml"
    "rlMountainCarContinuous.yaml"
    "rlMountainCarContinuousReinforce.yaml"
    "titanic.yaml"
)

# Base command
BASE_CMD="python run.py --container_type docker --model litellm:gpt-5 --per_instance_cost_limit 5.00 --agent_config_path configs/agents/default.yaml --temp 1 --gpus 0 --max_steps 100 --aliases_file ./dockerfiles/aliases.sh"

# Array to track background job PIDs
declare -a job_pids=()

# Function to count running jobs
count_running() {
    local count=0
    for pid in "${job_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            ((count++))
        fi
    done
    echo "$count"
}

# Function to clean up finished jobs from tracking
cleanup_finished() {
    local new_pids=()
    for pid in "${job_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            new_pids+=("$pid")
        fi
    done
    job_pids=("${new_pids[@]}")
}

# Function to wait until we have fewer than 2 running jobs
wait_for_slot() {
    while [[ $(count_running) -ge 2 ]]; do
        echo "Waiting for a slot (currently running: $(count_running))..."
        cleanup_finished
        sleep 5
    done
}

# Function to wait for all jobs to finish
wait_all_done() {
    while [[ $(count_running) -gt 0 ]]; do
        echo "Waiting for all experiments to finish (currently running: $(count_running))..."
        cleanup_finished
        sleep 5
    done
    # Final cleanup
    job_pids=()
}

# Main execution
for seed in "${SEEDS[@]}"; do
    echo "=== Starting seed $seed ==="
    
    # Run all tasks for this seed
    for task in "${TASKS[@]}"; do
        if [[ ! -f "configs/tasks/$task" ]]; then
            echo "Warning: $task not found, skipping"
            continue
        fi
        
        wait_for_slot
        
        task_name="${task%.yaml}"
        echo "Starting $task with seed $seed"
        
        # Create log file for this experiment
        log_file="logs/${task_name}_${seed}.log"
        mkdir -p logs
        
        # Run experiment in background and capture PID
        $BASE_CMD --task_config_path "tasks/$task" > "$log_file" 2>&1 &
        pid=$!
        job_pids+=("$pid")
        
        echo "Started $task_name (PID: $pid, log: $log_file)"
    done
    
    # Wait for all tasks of this seed to complete
    wait_all_done
    
    # Rename the trajectory folder
    if [[ -d "trajectories/$USER" ]]; then
        if [[ -d "trajectories/seed$seed" ]]; then
            echo "Merging into existing trajectories/seed$seed"
            rsync -av "trajectories/$USER/" "trajectories/seed$seed/"
            rm -rf "trajectories/$USER"
        else
            echo "Renaming trajectories/$USER to trajectories/seed$seed"
            mv "trajectories/$USER" "trajectories/seed$seed"
        fi
    else
        echo "Warning: No trajectory folder found for seed $seed"
    fi
    
    echo "=== Completed seed $seed ==="
done

echo "All experiments completed!"
echo "Check logs/ directory for individual experiment logs"
