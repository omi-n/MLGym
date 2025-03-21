# Copyright (c) Meta Platforms, Inc. and affiliates.

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
    echo "Launching experiment with model: $model"

    python run.py \
        --container_type docker \
        --task_config_path tasks/battleOfSexes.yaml \
        --model "$model" \
        --per_instance_cost_limit 4.00 \
        --agent_config_path configs/agents/default.yaml \
        --temp 0 \
        --gpus_per_agent 0 \
        --num_agents 4 \
        --max_steps 50 \
        --suffix parallel_agents \
        --aliases_file ./docker/aliases.sh
}

# Initialize counters
total_models=${#MODELS[@]}
current_model=0

# Launch first two experiments
run_experiment "${MODELS[$current_model]}" &
pid1=$!
current_model=$((current_model + 1))

run_experiment "${MODELS[$current_model]}" &
pid2=$!
current_model=$((current_model + 1))

# Keep launching new experiments when one finishes
while [ $current_model -lt $total_models ]; do
    # Wait for any child process to finish
    wait -n

    # Check which process finished
    if ! kill -0 $pid1 2>/dev/null; then
        run_experiment "${MODELS[$current_model]}" &
        pid1=$!
        current_model=$((current_model + 1))
    elif ! kill -0 $pid2 2>/dev/null; then
        run_experiment "${MODELS[$current_model]}" &
        pid2=$!
        current_model=$((current_model + 1))
    fi
done

# Wait for final experiments to complete
wait

echo "All experiments completed!"