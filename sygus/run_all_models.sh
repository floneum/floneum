#!/usr/bin/env bash
set -euo pipefail

# Make sure output directory exists
mkdir -p results

# Maximum number of attempts per run
max_retries=1

# List of model names to run
models=(
  smol-lm
  qwen1.5b
  qwen3b
  qwen7b
  qwen0.5b
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
GRAMMARS=(
  name-combine
  name-combine-2
  name-combine-3
  name-combine-4
  bikes
  dr-name
  firstname
  initials
  lastname
  phone-1
  phone-2
  phone-3
  phone-4
  phone-5
  phone-6
  phone-7
  phone-8
  phone-9
  phone-10
  phone
  reverse-name
  univ_1
  univ_2
  univ_3
  univ_4
  univ_5
  univ_6
)

TASK="./src/prompt"
TIME=$((60*5))


cargo build --features "${FEATURES}" --release

for model in "${models[@]}"; do
  for grammar in "${GRAMMARS[@]}"; do
  echo "Using grammar: ${grammar}"
    for fast in "${fast_cases[@]}"; do
      combo_tag="fast-${fast}"
      echo "=== Running model: ${model} (${combo_tag}) ==="
      attempt=1

      while [ $attempt -le $max_retries ]; do
        file_name="results/${grammar}_${model}_${combo_tag}.jsonl"
        # If the file exists skip
        if [ -f "${file_name}" ]; then
          echo "[${model} | ${combo_tag}] Output file ${file_name} already exists, skipping."
          break
        fi
        if "../target/release/sygus" \
              --model "${model}" \
              --grammar "sygus-strings/${grammar}.sl" \
              --task "${TASK}" \
              --time-seconds "${TIME}" \
              --fast-case "${fast}" \
              --recursion-depth 6 \
              > "${file_name}" 2>&1
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
done

echo "All model runs complete."
