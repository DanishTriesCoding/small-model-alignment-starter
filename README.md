# Small-Model Alignment Starter

This branch is the student starter package for the SFT + DPO alignment assignment. It is intentionally minimal: it provides the tested environment metadata, assignment PDF, fixed evaluation/data files, and no completed training implementation.

## What Is Included

```text
requirements.txt
pyproject.toml
submission_adapter.py
tests/test_public_adapter_smoke.py
tests/toy_lm.py
docs/assignment_manual.pdf
data/persona_lora/ramsay_style_preferences.json
data/persona_lora/README.md
eval_sets/hh_rlhf_eval_300_seed42.json
```

## Recommended Kaggle Setup

Use a Kaggle notebook with the `GPU T4 x2` accelerator. After cloning this repository/branch, run:

```bash
python -m pip uninstall -y -q torchao || true
python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors pytest trl peft
python -m pip install -q -e . --no-deps
```

This keeps Kaggle's preinstalled CUDA-compatible PyTorch intact, avoids known stale `torchao` / PEFT conflicts, and makes the local repository importable without replacing the GPU stack.

## What You Need To Build

You will add your own training and evaluation code on top of this starter. At minimum, your project should implement:

- SFT training on 30,000 examples.
- From-scratch DPO on HH-RLHF-style preference data.
- From-scratch DPO on UltraFeedback-style preference data.
- TRL `DPOTrainer` + LoRA training on the provided persona preference data.
- Generation and DPO-vs-SFT evaluation outputs.
- A short final report with plots and analysis.

Read `docs/assignment_manual.pdf` for the full requirements and suggested workflow.

## Submission Adapter And Public Tests

Keep the six function signatures in `submission_adapter.py` unchanged. Implement
each function as a small wrapper around your own SFT/DPO code. The grader uses
this stable interface so your internal file and class organization can remain
your choice.

Run the public tests from the repository root:

```bash
python -m pip install -q pytest
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
```

The tests use a tiny local tokenizer and model, so they do not download a model,
load the assignment datasets, or require a GPU. The starter adapter raises
`NotImplementedError` until you connect it to your implementation. Passing the
public tests checks the interface and a few basic invariants; the private grader
also checks numerical correctness and harder edge cases described in the manual.

For `compute_sft_loss`, the assignment accepts either the mean of per-example
summed response negative log probabilities or the mean over all unmasked response
tokens. Use one reduction consistently and document it in the report.

## Fixed Files

- `eval_sets/hh_rlhf_eval_300_seed42.json` is the fixed synthetic helpful/harmless evaluation set.
- For AlpacaEval-style evaluation, use 300 prompts with seed `42` so comparisons are consistent.
- `eval_sets/alpaca_eval_300_seed42.json` is the exact AlpacaEval subset (300 prompts, seed `42`) used for every Alpaca generation.
- `eval_sets/persona_style_eval_seed42.json` is the fixed 12-prompt style set (cooking, coding, general help, safety-sensitive) for the persona evaluation.
- `data/persona_lora/ramsay_style_preferences.json` is the small preference dataset for the LoRA-DPO persona task.

## Saving Work On Kaggle

Kaggle sessions are temporary. Write important outputs under `/kaggle/working`, save notebook versions, and archive checkpoints/generations as zip files or Kaggle output datasets before ending a session.

## Results

Checkpoints stay on Kaggle (too large for git). Everything needed to rebuild the report tables is in `results/`:

| Path | Contents | Made by |
|---|---|---|
| `results/plots/` | SFT loss, DPO (HH + UF) loss / reward accuracy / reward margin, persona LoRA-DPO curves (`.png` + `.pdf`) | `scripts/plot_training.py` |
| `results/plots/training_summary.json` | best checkpoint step and best/last validation metrics per run | `scripts/plot_training.py` |
| `results/generations/` | greedy generations (`max_new_tokens=256`, `repetition_penalty=1.1`, `no_repeat_ngram_size=4`): base / SFT / DPO-UF on Alpaca, SFT / DPO-HH on HH, base / LoRA SmolLM2 on the persona set | `scripts/generate.py` |
| `results/eval/*_pairs.jsonl`, `*_key.jsonl` | blind A/B pairs and the separate A/B to model key | `scripts/build_pairs.py` |
| `results/eval/*_judged.jsonl` | blind judge verdicts (judge: `google/gemini-3.6-flash`, 5 pairs per call, 5 s sleep, up to 3 attempts) | `scripts/judge_pairwise_kaggle.py` |
| `results/eval/*_judged_summary.json` | wins / losses / ties / unknown / failures, win rates, per-category counts | `scripts/judge_pairwise_kaggle.py` |
| `results/eval/*_judged_resolved.jsonl` | verdicts mapped back to model names, with the prompt and both responses | `scripts/judge_pairwise_kaggle.py` |

Comparisons: `hh_dpo_vs_sft` (DPO-HH vs SFT, HH set), `uf_dpo_vs_sft` (DPO-UF vs SFT, Alpaca subset), and `persona_lora_vs_base` judged twice: `persona_style` (persona faithfulness) and `persona_quality` (usefulness and safety, tone ignored).
