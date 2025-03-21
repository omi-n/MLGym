# Copyright (c) Meta Platforms, Inc. and affiliates.

#!/bin/bash
# ./scripts/experiments/memory/imageClassificationFMnist.sh

# List of models to test
MODELS=(
    "llama3-405b-tools"
    "gpt4o2"
    "gpt-o1"
    "claude-35-sonnet-new"
    "gemini-15-pro"
)

# Function to run experiment
run_experiment() {
    local model=$1
    local gpu_start=$2  # Starting GPU index
    echo "Launching experiment with model: $model on GPUs $gpu_start-$((gpu_start+3))"

    python run.py \
        --container_type podman \
        --task_config_path tasks/imageClassificationFMnist.yaml \
        --model "$model" \
        --per_instance_cost_limit 4.00 \
        --agent_config_path configs/agents/default_memory.yaml \
        --temp 0 \
        --gpus $gpu_start $((gpu_start+1)) $((gpu_start+2)) $((gpu_start+3)) \
        --gpus_per_agent 1 \
        --num_agents 4 \
        --max_steps 50 \
        --suffix parallel_agents \
        --aliases_file ./docker/aliases.sh \
        --memory_enabled True
}

# Initialize counters
total_models=${#MODELS[@]}
current_model=0

# Launch first two experiments
run_experiment "${MODELS[$current_model]}" 0 &  # Using GPUs 0-3
pid1=$!
gpu1_in_use=true
current_model=$((current_model + 1))

run_experiment "${MODELS[$current_model]}" 4 &  # Using GPUs 4-7
pid2=$!
gpu2_in_use=true
current_model=$((current_model + 1))

# Keep launching new experiments when one finishes
while [ $current_model -lt $total_models ]; do
    # Wait for any child process to finish
    wait -n

    # Check which process finished
    if ! kill -0 $pid1 2>/dev/null; then
        # First process finished, launch new one with GPUs 0-3
        run_experiment "${MODELS[$current_model]}" 0 &
        pid1=$!
        current_model=$((current_model + 1))
    elif ! kill -0 $pid2 2>/dev/null; then
        # Second process finished, launch new one with GPUs 4-7
        run_experiment "${MODELS[$current_model]}" 4 &
        pid2=$!
        current_model=$((current_model + 1))
    fi
done

# Wait for final experiments to complete
wait

echo "All experiments completed!"