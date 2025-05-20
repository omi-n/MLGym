#!/bin/bash

# Script to count directories by model name and task name
# Directory format: metagen-{model_name}__{task_name}__*__parallel_agents_run_{number}

echo "Counting directories by model name and task name..."
echo "=================================================="

# Count by model name
echo "Count by model name:"
echo "-----------------"
find . -type d -name "metagen-*__*__*__parallel_agents_run_*" | 
  sed -E 's|^\./metagen-([^_]*)__.*$|\1|' | 
  sort | 
  uniq -c | 
  sort -nr

echo ""

# Count by task name
echo "Count by task name:"
echo "-----------------"
find . -type d -name "metagen-*__*__*__parallel_agents_run_*" | 
  sed -E 's|^\./metagen-[^_]*__([^_]*)__.*$|\1|' | 
  sort | 
  uniq -c | 
  sort -nr

echo ""

# Count by model name and task name
echo "Count by model name and task name:"
echo "-------------------------------"
find . -type d -name "metagen-*__*__*__parallel_agents_run_*" | 
  sed -E 's|^\./metagen-([^_]*)__([^_]*)__.*$|\1 \2|' | 
  sort | 
  uniq -c | 
  sort -nr

echo ""
echo "Total directories: $(find . -type d -name "metagen-*__*__*__parallel_agents_run_*" | wc -l)" 