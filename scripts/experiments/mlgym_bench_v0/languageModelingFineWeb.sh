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
        --task_config_path tasks/languageModelingFineWeb.yaml \
        --model "$model" \
        --per_instance_cost_limit 4.00 \
        --config_file configs/agents/better_thought_action_parser_with_insert.yaml \
        --temp 0 \
        --gpus 0 1 2 3 4 5 6 7 \
        --gpus_per_agent 2 \
        --num_agents 4 \
        --max_steps 50 \
        --suffix parallel_agents \
        --aliases_file ./docker/aliases_docker.sh

    sleep 300
done

# wait for all background processes to complete
wait

python scripts/process_results.py \
    --traj_parent_dir trajectories/deepak/ \
    --traj_pattern languageModelingFineWeb__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__parallel_agents \
    --models "${MODELS[@]}" \
    --priority_metric "val_loss" \
    --metric_direction minimize > results/languageModelingFineWeb.res
