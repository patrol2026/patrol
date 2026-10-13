#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export HF_TOKEN="${HF_TOKEN:?export HF_TOKEN}"
MODEL="${1:-vicuna}"
COMMON="--model_name $MODEL --attacker RandomSearch --disable_GPT_judge \
  --patrol_check_interval 4 --patrol_backtrack_length 20 --patrol_max_check_tokens 128 --seed 0"
python src/defense_patrol.py $COMMON --defender PATROL-Keyword
for G in meta-llama/Meta-Llama-Guard-2-8B google/shieldgemma-9b Qwen/Qwen3Guard-Stream-8B; do
  python src/defense_patrol.py $COMMON --defender PATROL --patrol_guard_model "$G"
done
python src/bench_classifiers.py --model "$MODEL" --attacks RandomSearch DeepInception
