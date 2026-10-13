#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export HF_TOKEN="${HF_TOKEN:?export HF_TOKEN for gated models}"
python src/adaptive_gcg.py \
  --target_model lmsys/vicuna-7b-v1.5 \
  --guard_model meta-llama/Meta-Llama-Guard-2-8B \
  --incremental --n_prompts 30 \
  --target_rounds 2 --inner_steps 100 --grow_tokens 32 --max_target_len 64 \
  --opt_check_interval 8 --search_width 256 --topk 256 --topn_evade 48 \
  --suffix_len 20 --lam 1.0 --check_interval 4 --horizon 128 \
  --stop_target 0.18 --stop_evade 0.25 --seed 0 \
  --out results/adaptive_gcg_vicuna.json
