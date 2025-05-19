#!/bin/bash

# Script to rename folders:
# 1. Replace 'meta' with 'metagen' at the start of folder names
# 2. Replace 'deepseek_r1' with 'deepseek-r1' in folder names
# 3. Replace 'better_thought_action_parser_with_insert' with 'default' in folder names

# Set target directory to trajectories/mlgym_bench_v0 if not specified
DIRECTORY="${1:-./trajectories/mlgym_bench_v0}"
echo "Processing directories in: $DIRECTORY"

# Check if the directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory '$DIRECTORY' does not exist."
    exit 1
fi

# Variables to track progress
renamed_count=0
skipped_count=0

# Process each directory directly in the specified path
for dir in "$DIRECTORY"/*; do
    # Skip if not a directory
    if [ ! -d "$dir" ]; then
        continue
    fi
    
    # Get base name of directory
    old_name=$(basename "$dir")
    new_name="$old_name"
    
    # Replace 'meta' with 'metagen' at the start of folder names
    if [[ "$old_name" == meta-* ]]; then
        new_name="metagen-${old_name#meta-}"
    fi
    
    # Replace 'deepseek_r1' with 'deepseek-r1' in folder names
    if [[ "$new_name" == *deepseek_r1* ]]; then
        new_name="${new_name//deepseek_r1/deepseek-r1}"
    fi
    
    # Replace 'better_thought_action_parser_with_insert' with 'default' in folder names
    if [[ "$new_name" == *better_thought_action_parser_with_insert* ]]; then
        new_name="${new_name//better_thought_action_parser_with_insert/default}"
    fi
    
    # Only rename if the name would actually change
    if [ "$new_name" != "$old_name" ]; then
        echo "Renaming: $old_name → $new_name"
        mv "$DIRECTORY/$old_name" "$DIRECTORY/$new_name"
        if [ $? -eq 0 ]; then
            ((renamed_count++))
        else
            echo "Error renaming $DIRECTORY/$old_name"
        fi
    else
        ((skipped_count++))
    fi
done

echo ""
echo "Rename operation completed."
echo "Renamed directories: $renamed_count"
echo "Skipped directories: $skipped_count" 