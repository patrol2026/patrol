from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_HERE    = Path(__file__).parent.resolve()
_SD_ROOT = _HERE.parent.resolve()

_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating the quality of responses to user "
    "instructions. Focus on helpfulness, accuracy, and completeness."
)

_JUDGE_TEMPLATE = """\
Compare the two responses below to the given instruction and decide which is better.

### Instruction
{instruction}

### Response A
{response_a}

### Response B
{response_b}

Which response better follows the instruction and is more helpful?
Reply with exactly one word: "A", "B", or "tie". No explanation.
"""

def _gpt_judge(instruction: str, response_a: str, response_b: str,
               client, model_id: str) -> str:
    prompt = _JUDGE_TEMPLATE.format(
        instruction=instruction,
        response_a=response_a,
        response_b=response_b,
    )
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0,
        max_tokens=5,
    )
    raw = resp.choices[0].message.content.strip().lower()
    if raw.startswith("a"):
        return "A"
    if raw.startswith("b"):
        return "B"
    return "tie"

def load_local_judge(model_name: str):
    import transformers
    print(f"Loading local judge: {model_name} …")
    tok   = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    return model, tok

@torch.no_grad()
def _local_judge(instruction: str, response_a: str, response_b: str,
                 model, tokenizer) -> str:
    prompt = _JUDGE_TEMPLATE.format(
        instruction=instruction,
        response_a=response_a,
        response_b=response_b,
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    text   = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out    = model.generate(
        **inputs, max_new_tokens=10, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0, inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower()
    if re.match(r"^a\b", raw):
        return "A"
    if re.match(r"^b\b", raw):
        return "B"
    return "tie"

def compute_win_rate(judgments: list[str]) -> dict:
    n      = len(judgments)
    n_win  = sum(1 for j in judgments if j == "A")
    n_tie  = sum(1 for j in judgments if j == "tie")
    n_loss = sum(1 for j in judgments if j == "B")
    win_rate = (n_win + 0.5 * n_tie) / n if n > 0 else 0.0
    return {
        "win_rate":  round(win_rate * 100, 2),
        "n_total":   n,
        "n_win":     n_win,
        "n_tie":     n_tie,
        "n_loss":    n_loss,
        "pct_win":   round(n_win  / n * 100, 1) if n else 0.0,
        "pct_tie":   round(n_tie  / n * 100, 1) if n else 0.0,
        "pct_loss":  round(n_loss / n * 100, 1) if n else 0.0,
    }

def judge_file(path: Path, judge_fn, max_prompts: int | None = None,
               output_dir: Path | None = None) -> dict:
    raw      = json.loads(path.read_text(encoding="utf-8"))
    items    = raw.get("data", raw) if isinstance(raw, dict) else raw
    exp_vars = raw.get("experiment_variables", {}) if isinstance(raw, dict) else {}
    generator = exp_vars.get("defender", path.stem)

    if max_prompts is not None:
        items = items[:max_prompts]

    judgments    = []
    judged_items = []

    for item in tqdm(items, desc=f"  judging {generator}", leave=True):
        instruction = item.get("instruction", "")
        response_a  = item.get("output", "")
        response_b  = item.get("reference_output", "")

        if not instruction or not response_a or not response_b:
            continue

        verdict = judge_fn(instruction, response_a, response_b)
        judgments.append(verdict)
        judged_items.append({
            "instruction":      instruction,
            "model_output":     response_a,
            "reference_output": response_b,
            "verdict":          verdict,
        })

    stats = compute_win_rate(judgments)
    stats["generator"] = generator
    stats["exp_vars"]  = exp_vars

    out_dir  = output_dir if output_dir is not None else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (path.stem + "_winrate.json")
    out_path.write_text(
        json.dumps({"stats": stats, "judgments": judged_items},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  → win rate: {stats['win_rate']:.1f}%  "
          f"(win {stats['pct_win']:.1f}%  tie {stats['pct_tie']:.1f}%  "
          f"loss {stats['pct_loss']:.1f}%)")
    print(f"  Saved → {out_path.name}")
    return stats

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True,
                   help="Path(s) / glob patterns to output JSON files from "
                        "run_alpaca_eval.py.")
    p.add_argument("--judge", default="gpt-4",
                   choices=["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4.1", "local"],
                   help="Judge backend.")
    p.add_argument("--gpt_api",     default=None,
                   help="OpenAI API key (required for GPT judges).")
    p.add_argument("--judge_model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                   help="HuggingFace model ID for the local judge.")
    p.add_argument("--max_prompts", type=int, default=None,
                   help="Limit number of judgments per file (for quick tests).")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Directory to write *_winrate.json files. "
                        "Defaults to the same folder as each input.")
    return p.parse_args()

def main():
    args = get_args()

    paths = []
    for pattern in args.input:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            paths.extend(Path(p) for p in expanded)
        else:
            paths.append(Path(pattern))

    paths = [p for p in paths
             if p.exists()
             and not p.name.endswith("_winrate.json")]

    if not paths:
        print("No valid input files found.")
        sys.exit(1)

    print(f"Found {len(paths)} file(s) to judge.")

    if args.judge in ("gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4.1"):
        if args.gpt_api is None:
            raise ValueError("--gpt_api is required for GPT judges.")
        import openai
        client   = openai.OpenAI(api_key=args.gpt_api)
        model_id = args.judge
        judge_fn = lambda inst, a, b: _gpt_judge(inst, a, b, client, model_id)

    else:
        judge_model, judge_tok = load_local_judge(args.judge_model)
        judge_fn = lambda inst, a, b: _local_judge(inst, a, b, judge_model, judge_tok)

    all_stats = []
    for path in paths:
        print(f"\nJudging: {path.name}")
        stats = judge_file(path, judge_fn, args.max_prompts, args.output_dir)
        all_stats.append(stats)

    col = 38
    print(f"\n{'='*(col+36)}")
    print(f"  AlpacaEval Win Rate  (model vs. text-davinci-003)")
    print(f"  Judge: {args.judge}")
    print(f"{'='*(col+36)}")
    print(f"  {'Defender':<{col}} {'WinRate':>9} {'Win%':>6} {'Tie%':>6} {'Loss%':>6}")
    print(f"  {'─'*col} {'─'*9} {'─'*6} {'─'*6} {'─'*6}")
    for s in all_stats:
        print(f"  {s['generator']:<{col}} {s['win_rate']:>8.1f}%"
              f" {s['pct_win']:>5.1f}% {s['pct_tie']:>5.1f}% {s['pct_loss']:>5.1f}%")
    print(f"{'='*(col+36)}")
    print(f"  Win rate = (1/|P|) * Σ [ I[win] + 0.5 * I[tie] ]\n")

if __name__ == "__main__":
    main()
