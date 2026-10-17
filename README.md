# Small-Model Alignment (CS5326 PA2)

This is my completed SFT and DPO assignment, built on top of the course starter package.
It has my SFT and from-scratch DPO training code, the TRL LoRA-DPO persona run, the generation and Kaggle judging pipeline, and all evaluation results.
The final report is `report/report.pdf` and its LaTeX source is in `report/`.
The results it cites are listed in the [Results](#results) section below.

## Repository Layout

```text
alignment/            batching, masking and loss code (SFT and DPO)
scripts/              training, generation, pair building, judging and plotting
submission_adapter.py wrapper functions used by the grader
tests/                public adapter tests and the toy model they use
data/persona_lora/    persona preference data
eval_sets/            the fixed prompt sets used for every comparison
results/              generations, judge verdicts, summaries and training plots
report/               final report (PDF and LaTeX source)
docs/                 assignment manual
```

## Build Requirements

The assignment asked for the pieces below. Each one is done and the table shows where it lives.

| Requirement | Where |
|---|---|
| SFT on 30,000 examples | `scripts/train_sft.py` |
| From-scratch DPO on HH-RLHF pairs | `scripts/train_dpo.py --dataset hh` |
| From-scratch DPO on UltraFeedback pairs | `scripts/train_dpo.py --dataset uf` |
| TRL `DPOTrainer` with LoRA on the persona data | `scripts/train_persona_dpo.py` |
| Generations and DPO vs SFT evaluation | `scripts/generate.py`, `scripts/build_pairs.py`, `scripts/judge_pairwise_kaggle.py` |
| Training plots | `scripts/plot_training.py` |
| Final report | `report/report.pdf` |

The full requirements are in `docs/assignment_manual.pdf`.

## Kaggle Setup

I ran everything in Kaggle notebooks with the `GPU T4 x2` accelerator. After cloning the repository I ran these commands.

```bash
python -m pip uninstall -y -q torchao || true
python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors pytest trl peft
python -m pip install -q -e . --no-deps
```

This keeps Kaggle's own CUDA build of PyTorch, avoids an old `torchao` conflict with PEFT, and makes the repository importable.
Each stage wrote its outputs to `/kaggle/working` and I saved a notebook version after every stage.
Training, generation and judging each ran in their own notebook, with the previous notebook's output attached as input.

## Submission Adapter and Tests

`submission_adapter.py` keeps the six required function signatures and calls into my code in `alignment/`.
The public tests run on a tiny local model, so they need no GPU and no downloads.

```bash
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
```

All 7 public tests pass.
For `compute_sft_loss` I use the mean over all unmasked response tokens in the batch.

## Data and Evaluation Sets

| File | What it is |
|---|---|
| `eval_sets/hh_rlhf_eval_300_seed42.json` | Synthetic helpful and harmless set given with the assignment. Used for HH DPO vs SFT |
| `eval_sets/alpaca_eval_300_seed42.json` | The exact 300 AlpacaEval prompts sampled with seed 42. Used for UF DPO vs SFT |
| `eval_sets/persona_style_eval_seed42.json` | My 12-prompt style set covering cooking, coding, general help and safety-sensitive requests |
| `data/persona_lora/ramsay_style_preferences.json` | Persona preference data for the LoRA-DPO run |

## Results

Checkpoints are too large for git, so they stay on Kaggle. Everything needed to rebuild the report tables is in `results/`.

| Path | Contents | Made by |
|---|---|---|
| `results/plots/` | SFT loss, DPO loss, reward accuracy and reward margin for HH and UF, and the persona LoRA-DPO curves | `scripts/plot_training.py` |
| `results/plots/training_summary.json` | Best checkpoint step and best and last validation metrics for each run | `scripts/plot_training.py` |
| `results/generations/` | Greedy generations with 256 new tokens, repetition penalty 1.1 and no repeated 4-grams. Base, SFT and UF DPO on Alpaca. SFT and HH DPO on the HH set. Base and LoRA SmolLM2 on the persona set | `scripts/generate.py` |
| `results/eval/*_pairs.jsonl` and `*_key.jsonl` | Blind A/B pairs and the separate key that maps positions to models | `scripts/build_pairs.py` |
| `results/eval/*_judged.jsonl` | Blind verdicts from `google/gemini-3.6-flash` with 5 pairs per call, a 5 second wait and up to 3 attempts | `scripts/judge_pairwise_kaggle.py` |
| `results/eval/*_judged_summary.json` | Wins, losses, ties, unknowns, failures, win rates and per-category counts | `scripts/judge_pairwise_kaggle.py` |
| `results/eval/*_judged_resolved.jsonl` | Verdicts mapped back to model names, with the prompt and both answers | `scripts/judge_pairwise_kaggle.py` |

The comparisons are `hh_dpo_vs_sft` for HH DPO against SFT on the HH set and `uf_dpo_vs_sft` for UF DPO against SFT on the Alpaca subset.
The persona pairs in `persona_lora_vs_base` were judged twice.
`persona_style` checks how well each answer uses the persona and `persona_quality` checks usefulness and safety while ignoring tone.
