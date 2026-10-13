#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY}"
MANIFEST="${1:?usage: run_judge.sh <manifest.txt> <harmful|refusal>}"
MODE="${2:-harmful}"
python src/judge_gpt.py --mode "$MODE" --manifest "$MANIFEST" --summary_csv judge_summary.csv
