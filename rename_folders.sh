#!/bin/bash

# Script to rename folders:
# 1. Replace 'meta' with 'metagen' at the start of folder names
# 2. Replace 'deepseek_r1' with 'deepseek-r1' in folder names

# Set working directory to script location if not specified
DIRECTORY="${1:-.}"
cd "$DIRECTORY"

# Variables to track progress
renamed_count=0
skipped_count=0

# Process each directory
for dir in */; do
    # Remove trailing slash
    old_name="${dir%/}"
    new_name="$old_name"
    
    # Replace 'meta' with 'metagen' at the start of folder names
    if [[ "$old_name" == meta-* ]]; then
        new_name="metagen-${old_name#meta-}"
    fi
    
    # Replace 'deepseek_r1' with 'deepseek-r1' in folder names
    if [[ "$new_name" == *deepseek_r1* ]]; then
        new_name="${new_name//deepseek_r1/deepseek-r1}"
    fi
    
    # Only rename if the name would actually change
    if [ "$new_name" != "$old_name" ]; then
        echo "Renaming: $old_name → $new_name"
        mv "$old_name" "$new_name"
        if [ $? -eq 0 ]; then
            ((renamed_count++))
        else
            echo "Error renaming $old_name"
        fi
    else
        ((skipped_count++))
    fi
done

echo ""
echo "Rename operation completed."
echo "Renamed directories: $renamed_count"
echo "Skipped directories: $skipped_count" 