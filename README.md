# PATROL: Adaptive Decoding with Safety Backtracking

Official implementation and reproduction scripts for **PATROL**, a training-free
decoding-time defence against jailbreak attacks on autoregressive LLMs.

> **Paper:** _PATROL: Adaptive Decoding with Safety Backtracking_ (anonymous submission, under review)
> **Anonymous code:** this repository

PATROL wraps any autoregressive LLM served through a decoding-loop interface
(Hugging Face `transformers`, vLLM, TGI). During generation it periodically
queries a safety classifier on the partial response; on an unsafe verdict it
backtracks the decoder to a recent state and resamples under a position-wise
blocklist that prevents re-entry into the same harmful trajectory.

---

## Repository layout

```
patrol/
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── defense_patrol.py           PATROL + baseline defenders
│   ├── safety_constrainers.py      classifier wrappers (LlamaGuard-2, ShieldGemma, Qwen3Guard-Stream, keyword)
│   ├── scm_defense.py              streaming-monitor baseline (SCM)
│   ├── run_patrol_sweep.py         hyperparameter sweep
│   ├── adaptive_gcg.py             adaptive attack on the classifier (dual-objective GCG)
│   ├── adaptive_delay.py           adaptive attack on the horizon (delayed payload)
│   ├── bench_classifiers.py        offline detection benchmark for candidate classifiers
│   ├── orbench_refusal.py          over-refusal rate on OR-Bench prompts
│   ├── judge_gpt.py                GPT-4.1 judge (ASR and over-refusal) via OpenAI-compatible API
│   ├── judge_all.py                batch ASR judge over result directories
│   ├── judge_alpaca_eval.py        AlpacaEval pairwise judge
│   ├── run_alpaca_eval_official.py AlpacaEval 2.0 protocol
│   ├── compute_early_commitment.py detection-CDF analysis
│   ├── plot_early_commitment.py    per-prefix detection figure
│   └── plot_ablations.py           ablation figures
├── scripts/
│   ├── run_classifier_comparison.sh  PATROL under four candidate classifiers
│   ├── run_adaptive_gcg.sh           adaptive GCG attack
│   ├── run_adaptive_delay.sh         delayed-payload attack sweep
│   ├── run_orbench.sh                over-refusal generation across defenders
│   ├── run_judge.sh                  GPT-4.1 judging of a result manifest
│   ├── run_fill_gaps.sh              generate missing baseline cells
│   └── run_ablations_mct_bl_ci.sh    c / ℓ / m ablations
├── configs/
│   └── weighted_alpaca_eval_gpt41/   AlpacaEval 2.0 with a GPT-4.1 annotator
└── data/                             attack prompts and intermediate files (see §2)
```

---

## 1. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The runtime classifier is **LlamaGuard-2-8B**, served as a standard Hugging Face
model; the first invocation downloads the weights. Export a token with access to
the gated models:

```bash
export HF_TOKEN="hf_..."
```

The GPT-4.1 judge (`judge_gpt.py`) talks to an OpenAI-compatible endpoint. Set the
key for whichever provider you use:

```bash
export OPENROUTER_API_KEY="..."     # or OPENAI_API_KEY for the OpenAI endpoint
```

---

## 2. External assets

These are not redistributed here; download them and place the prompt artefacts
under `data/`.

| Asset | Source |
|---|---|
| `lmsys/vicuna-7b-v1.5`, `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Meta-Llama-3-8B-Instruct` | Hugging Face |
| `meta-llama/Meta-Llama-Guard-2-8B` (runtime classifier) | Hugging Face |
| `google/shieldgemma-9b`, `Qwen/Qwen3Guard-Stream-8B` (classifier comparison) | Hugging Face |
| AdvBench / GCG | https://github.com/llm-attacks/llm-attacks |
| AutoDAN | https://github.com/SheltonLiu-N/AutoDAN |
| PAIR | https://github.com/patrickrchao/JailbreakingLLMs |
| DeepInception | https://github.com/tmlr-group/DeepInception |
| RandomSearch | https://github.com/tml-epfl/llm-adaptive-attacks |
| OR-Bench (over-refusal) | https://huggingface.co/datasets/bench-llm/or-bench |
| AlpacaEval 2.0 | https://github.com/tatsu-lab/alpaca_eval |
| SafeDecoding LoRA experts | https://github.com/uw-nsl/SafeDecoding |

The SCM baseline (`scm_defense.py`) additionally needs the streaming content
monitor checkpoint and its `QwenForDualTask` model class from the SCM authors'
release; pass both paths to `load_scm_monitor`.

---

## 3. Reproducing the paper

All commands assume `pwd` is `patrol_release/`. The default PATROL configuration
is `(c=4, ℓ=20, m=128)`.

### 3.1 Main results

```bash
bash scripts/run_fill_gaps.sh
python src/judge_all.py --judge_backend openai --openai_model gpt-4.1 \
    --include_dirs results --model_filter llama2 llama3 vicuna \
    --no_strict_filter --spreadsheet results_summary.xlsx
```

### 3.2 Utility (AlpacaEval 2.0)

```bash
ln -s "$(pwd)/configs/weighted_alpaca_eval_gpt41" \
      "$(python -c 'import alpaca_eval, os; print(os.path.dirname(alpaca_eval.__file__))')/evaluators_configs/weighted_alpaca_eval_gpt41"
python src/run_alpaca_eval_official.py --output_dir results_alpaca_eval_official \
    --annotators_config weighted_alpaca_eval_gpt41 --skip_existing \
    --input "results/*alpaca_eval*.json"
```

### 3.3 Over-refusal (OR-Bench)

```bash
bash scripts/run_orbench.sh vicuna
python src/judge_gpt.py --mode refusal --manifest orbench_manifest.txt \
    --summary_csv orbench_over_refusal.csv
python src/orbench_refusal.py orbench_manifest.txt
```

### 3.4 Choice of classifier

Runs PATROL end to end under the keyword rule, LlamaGuard-2, ShieldGemma, and
Qwen3Guard-Stream, then the offline detection benchmark.

```bash
bash scripts/run_classifier_comparison.sh vicuna
python src/judge_gpt.py --mode harmful --manifest classifier_manifest.txt \
    --summary_csv classifier_asr.csv
```

### 3.5 Adaptive attacks

The attack on the classifier optimises an adversarial suffix that both elicits the
target behaviour and keeps every checked prefix below the classifier threshold:

```bash
bash scripts/run_adaptive_gcg.sh
```

The attack on the horizon defers the harmful payload past the `m`-th token:

```bash
bash scripts/run_adaptive_delay.sh
```

Judge either with `src/judge_gpt.py --mode harmful`.

### 3.6 Ablations and early-commitment

```bash
bash scripts/run_ablations_mct_bl_ci.sh
python src/plot_ablations.py
python src/compute_early_commitment.py
python src/plot_early_commitment.py
```

---

## 4. Default hyperparameters

| symbol | meaning | default |
|---|---|---|
| `c` | classifier check interval (tokens) | 4 |
| `ℓ` | backtrack length on an unsafe verdict (tokens) | 20 |
| `m` | safety horizon (max prefix length checked) | 128 |

These are the defaults of `defense_patrol.py` and are used throughout the scripts.

---

## 5. Hardware

All experiments ran on a single NVIDIA RTX 6000 Ada (48 GB), CUDA 12.8. The same
GPU hosts both the target model and the classifier during decoding.

---

## 6. License

MIT (see [LICENSE](LICENSE)). The external attack and benchmark artefacts in §2
are released under their own licenses.

---

## 7. Citation

```bibtex
@inproceedings{patrol,
  title     = {PATROL: Adaptive Decoding with Safety Backtracking},
  author    = {Anonymous},
  year      = {2026}
}
```
