#!/usr/bin/env bash
set -euo pipefail

# Make sure output directory exists
mkdir -p results

# Maximum number of attempts per run
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

# Flag combinations
fast_cases=(true false)

# Common args
FEATURES="metal"
GRAMMAR="./src/firstname.sl"
TASK="./src/prompt"
ITERATIONS=50

for model in "${models[@]}"; do
  for fast in "${fast_cases[@]}"; do
    combo_tag="fast-${fast}"
    echo "=== Running model: ${model} (${combo_tag}) ==="
    attempt=1

    while [ $attempt -le $max_retries ]; do
      if cargo run --features "${FEATURES}" --release -- \
            --model "${model}" \
            --grammar "${GRAMMAR}" \
            --task "${TASK}" \
            --iterations "${ITERATIONS}" \
            --fast-case "${fast}" \
            --recursion-depth 4 \
            > "results/${model}_${combo_tag}.txt" 2>&1
      then
        echo "[${model} | ${combo_tag}] succeeded on attempt #${attempt}"
        break
      else
        echo "[${model} | ${combo_tag}] attempt #${attempt} failed"
        attempt=$((attempt + 1))
        if [ $attempt -le $max_retries ]; then
          echo "Retrying (attempt $attempt of $max_retries)…"
          # sleep 1  # uncomment to pause between retries
        fi
      fi
    done

    if [ $attempt -gt $max_retries ]; then
      echo "[${model} | ${combo_tag}] failed after ${max_retries} attempts, moving on."
    fi
    echo
  done
done

echo "All model runs complete."
