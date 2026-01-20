#!/bin/bash

model=$1
attack=$2
start_offset=$3
dataset=$4

NUM_RUNS=8

for ((i=0; i<NUM_RUNS; i++)); do
    current_offset=$((start_offset + i))
    echo "===== Run $((i+1)) / $NUM_RUNS | offset=${current_offset} ====="
    bash run_direct.sh "$model" "$attack" "$current_offset" "$dataset"
done
