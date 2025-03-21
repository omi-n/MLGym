# Copyright (c) Meta Platforms, Inc. and affiliates.

#!/bin/bash

# List of models to test
MODELS=(
    "llama3-405b-tools"
    "gpt4o2"
    "gpt-o1"
    "claude-35-sonnet-new"
    "gemini-15-pro"
)

# Loop through each model and run the experiment
for model in "${MODELS[@]}"; do
    echo "Running experiment with model: $model"

    python run.py \
        --container_type docker \
        --task_config_path tasks/rlMetaMaze.yaml \
        --model "$model" \
        --per_instance_cost_limit 4.00 \
        --agent_config_path configs/agents/default.yaml \
        --temp 0 \
        --gpus 0 1 2 3 4 5 6 7 \
        --gpus_per_agent 2 \
        --num_agents 4 \
        --max_steps 50 \
        --suffix parallel_agents \
        --aliases_file ./docker/aliases.sh

    sleep 300
done

# wait for all background processes to complete
wait

echo "All experiments completed!"