# Pruning Layers in Whisper's Encoder

Code for the paper *"Pruning Layers in Whisper's Encoder"*.

The six least important encoder layers are identified via leave-one-out ΔWER ranking and removed, reducing the encoder stack by 18.5%. Label-free knowledge distillation on unlabelled speech recovers multilingual performance, raising mean WER by only +1.9 pp across Danish, English, German, and French.

The pruned and distilled model is available on Hugging Face: [rasgaard/whisper-large-v3-turbo-encoder-pruned](https://huggingface.co/rasgaard/whisper-large-v3-turbo-encoder-pruned)

## Repository structure

```
├── compute_layer_metrics.py        # Layer importance ranking and pruning sweep
├── distill_pruned_encoder.py       # Label-free MSE distillation
├── paper_figures.ipynb             # Interactive notebook for all paper figures
├── scripts/
│   ├── make_paper_figures.py       # Script version of paper figures
│   ├── k7_zero_shot_sweep.py       # k=7 layer sensitivity analysis
│   └── k6_random_baseline_sweep.py # Random layer selection baseline
└── results/
    ├── baseline_wers.json                       # Full model WER on FLEURS test sets
    ├── delta_wers.json                          # Per-layer ΔWER for all 32 layers
    ├── prune_sweep_delta_wer.json               # Greedy pruning sweep k=1..14
    ├── distillation_log_5_6_7_9_11_12.json     # Distillation training log
    ├── k7_zero_shot_sweep.json                  # k=7 candidate layer sweep
    ├── k6_random_baseline_sweep.json            # Random baseline (n=50)
    └── paper/                                   # Paper figures (PDF + PNG)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/rasgaard/whisper-encoder-layer-prune
cd whisper-encoder-layer-prune
uv sync
```

## Reproducing the results

All steps below run on a single GPU. The full pipeline takes approximately 2–3 hours on an A100.

### Step 1 — Layer importance ranking

Measure ΔWER for each of the 32 encoder layers in isolation across Danish, English, German, and French:

```bash
uv run python compute_layer_metrics.py --phase 1
```

Output: `results/delta_wers.json`, `results/baseline_wers.json`

### Step 2 — Greedy pruning sweep

Sweep k=1 to k=14 layers removed, always taking the k least important by ΔWER:

```bash
uv run python compute_layer_metrics.py --phase 3
```

Output: `results/prune_sweep_delta_wer.json`

### Step 3 — Distillation

Remove the six least important layers and distill using unlabelled People's Speech audio:

```bash
uv run python distill_pruned_encoder.py --layers 5 6 7 9 11 12 --steps 2000
```

Output: `results/distillation_log_5_6_7_9_11_12.json`, `results/distilled_pruned_model_5_6_7_9_11_12/`

Key options: `--lr 1e-5`, `--batch-size 8`, `--eval-every 500`

### Step 4 — k=7 sensitivity analysis

Test ten candidate layers as a seventh removed layer (zero-shot):

```bash
uv run python scripts/k7_zero_shot_sweep.py
```

Output: `results/k7_zero_shot_sweep.json`

### Step 5 — Random layer selection baseline

Sample 50 random k=6 layer sets from the first half of the encoder and evaluate zero-shot:

```bash
uv run python scripts/k6_random_baseline_sweep.py --n-random 50 --seed 42
```

Output: `results/k6_random_baseline_sweep.json`

### Step 6 — Paper figures

Interactively via the notebook:

```bash
uv run jupyter lab paper_figures.ipynb
```

Or non-interactively:

```bash
uv run python scripts/make_paper_figures.py
```

Output: `results/paper/`

