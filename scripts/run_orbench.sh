#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export HF_TOKEN="${HF_TOKEN:?export HF_TOKEN}"
MODEL="${1:-vicuna}"
COMMON="--model_name $MODEL --attacker custom --disable_GPT_judge --max_new_tokens 384 --seed 0"
python src/defense_patrol.py $COMMON --defense_off
python src/defense_patrol.py $COMMON --defender PATROL \
  --patrol_guard_model meta-llama/Meta-Llama-Guard-2-8B \
  --patrol_check_interval 4 --patrol_backtrack_length 20 --patrol_max_check_tokens 128
for D in PPL Self-Reminder ICD Retokenization Self-Exam SafeDecoding SmoothLLM; do
  python src/defense_patrol.py $COMMON --defender "$D"
done
