#!/usr/bin/env bash
set -euo pipefail

# Make sure output directory exists
mkdir -p results

# Maximum number of attempts per model
max_retries=5

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
  attempt=1

  while [ $attempt -le $max_retries ]; do
    if cargo run --features "${FEATURES}" --release -- \
         --model "${model}" \
         --grammar "${GRAMMAR}" \
         --task "${TASK}" \
         --iterations 200 \
         > "results/${model}.txt" 2>&1
    then
      echo "[$model] succeeded on attempt #${attempt}"
      break
    else
      echo "[$model] attempt #${attempt} failed"
      attempt=$((attempt + 1))
      if [ $attempt -le $max_retries ]; then
        echo "Retrying (attempt $attempt of $max_retries)…"
        # sleep 1  # uncomment if you want a pause between retries
      fi
    fi
  done

  if [ $attempt -gt $max_retries ]; then
    echo "[$model] failed after $max_retries attempts, moving on."
  fi

  echo
done

echo "All models complete."
