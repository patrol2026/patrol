#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export HF_TOKEN="${HF_TOKEN:?export HF_TOKEN}"
python src/adaptive_delay.py
for L in 0 30 60 100 150 200; do
  cp data/adaptive_delay/custom_prompts_delay_L${L}.json data/custom_prompts.json
  python src/defense_patrol.py \
    --model_name vicuna --attacker custom --defender PATROL \
    --patrol_guard_model meta-llama/Meta-Llama-Guard-2-8B \
    --patrol_check_interval 4 --patrol_backtrack_length 20 --patrol_max_check_tokens 128 \
    --disable_GPT_judge --seed 0
done
