#!/usr/bin/env bash
set -euo pipefail

# List of model names to run
models=(
  qwen0.5b
  qwen1.5b
  qwen3b
  qwen7b
  smol-lm
  llama1b
  llama3b
  llama8b
  qwen1.5b-think
  qwen7b-think
)

# Common args
FEATURES="metal"
GRAMMAR="./src/firstname.sl"
TASK="./src/prompt"

for model in "${models[@]}"; do
  echo "=== Running model: $model ==="
  cargo run --features "${FEATURES}" --release -- \
    --model "${model}" \
    --grammar "${GRAMMAR}" \
    --task "${TASK}" > "./results/${model}.txt" 2>&1
  echo
done

echo "All models complete."
