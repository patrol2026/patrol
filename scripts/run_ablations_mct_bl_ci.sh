#!/bin/bash

set -e

MODELS="llama2 llama3 vicuna"
COMMON="--attacker RandomSearch \
        --defender PATROL \
        --patrol_configs 1.0-1.0-50 \
        --max_new_tokens 400 \
        --seed 42 \
        --disable_GPT_judge"

run() {
    local model=$1 ci=$2 bl=$3 mct=$4
    echo "============================================================"
    echo "  $model  ci=$ci  bl=$bl  mct=$mct"
    echo "============================================================"
    python defense_patrol.py \
        --model_name $model \
        $COMMON \
        --patrol_check_interval $ci \
        --patrol_backtrack_length $bl \
        --patrol_max_check_tokens $mct
}

for MODEL in $MODELS; do
    for MCT in 4 8 16 32 64 128 256; do
        run $MODEL 4 20 $MCT
    done
done

for MODEL in $MODELS; do
    for BL in 8 12 16 20 24 28; do
        run $MODEL 4 $BL 128
    done
done

for MODEL in $MODELS; do
    for CI in 2 4 6 8 16; do
        run $MODEL $CI 20 128
    done
done

echo ""
echo "All ablations done."

echo ""
echo "============================================================"
echo "  Judging ablation results with GPT-4.1"
echo "============================================================"

cd "$(dirname "$0")/../.."

python judge_all.py \
    --judge_backend openai \
    --openai_model gpt-4.1 \
    --include_dirs results \
    --model_filter llama2 llama3 vicuna \
    --no_strict_filter \
    --spreadsheet results_summary_ablations_gpt41.xlsx

echo ""
echo "Ablation judging complete."
