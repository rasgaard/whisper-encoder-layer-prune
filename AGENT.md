# Agent Handoff — Whisper Encoder Layer Pruning

This document is for a new Claude instance picking up this project. It covers current state, key findings, what's running, and what to do next.

---

## Project in one sentence

Empirically characterise which encoder layers in Whisper-large-v3-turbo are redundant (via proxy metrics and single-layer ΔWER), then attempt to exploit that redundancy through label-free representation distillation — evaluated multilingually across Danish, English, Italian, German, and Swedish.

---

## What has been done

### Phase 1 — Proxy metrics
`compute_layer_metrics.py --phase 1`

Computed per-layer cosine similarity, MSE, outlier dimension fraction (α=6.0), outlier delta, and activation amplification ratio for all 32 layers × 5 languages. Results in `results/proxy_metrics.json`.

**New field**: `max_act_per_dim` — shape [32, 1280] per language — stores the running max absolute activation per hidden dimension. Used for the alpha sensitivity analysis.

### Phase 2 — Per-layer ΔWER
`compute_layer_metrics.py --phase 2`

Removed each layer individually and measured WER change. Results in `results/delta_wers.json`. Key structure:
```json
{ "baseline_wers": {...}, "layer_wers": {...}, "delta_wers": {...} }
```

### Phase 3 — Multi-layer pruning sweep
`compute_layer_metrics.py --phase 3 --ranking delta_wer`

Removes layers incrementally in ΔWER rank order. Results in `results/prune_sweep_delta_wer.json`. The sweep ran through k=7 (catastrophic failure).

### Analysis notebook
`analysis.ipynb` — run with `uv run jupyter nbconvert --to notebook --execute --inplace analysis.ipynb`

Contains 6 sections:
1. ΔWER per layer (heatmap + bar chart)
2. Proxy metrics heatmaps
3. Proxy vs ΔWER Spearman correlation (RQ1)
4. Cross-language consistency (RQ2)
5. Layer importance ranking
6. Alpha sensitivity analysis (NEW)

### Distillation (just started / may still be running)
`distill_pruned_encoder.py`

Label-free representation distillation. Pruned student (layers {5,6,7,9,10,11} removed) trained to match full teacher's encoder output using MSE loss on People's Speech audio. Results will appear in `results/distillation_log.json` and `results/distilled_pruned_model/`.

---

## Key findings

### 1. Italian WER improves with layer removal
Removing layers 5–11 individually gives Italian a ~10–13% relative WER *reduction*. Stable across all individual-layer removals. Unexplained — likely related to over-processing of features or interference between layers for Romance languages.

### 2. Danish is the most fragile language
Danish exceeds the 5% relative WER threshold at k=3 (layers {7,9,11} removed). English and German stay well below 5% through k=5. Hypothesis: fragility correlates with lower training data representation in Whisper.

### 3. Non-additive catastrophic failure at k=6
Sweep results:
- k=5 ({5,6,7,9,11}): DA +14.7%, EN +4.3%, IT -10%, DE +5.7%
- k=6 ({5,6,7,9,10,11}): DA +58.8%, EN +9.2%, IT -7.4%, DE +9.0%

Layer 10 has individual ΔWER ≈ +0.002 for Danish (seemingly safe), but causes catastrophic failure in combination. Individual importance scores cannot predict multi-layer removal outcomes.

### 4. Outlier fraction outperforms cosine similarity as proxy
Spearman ρ vs ΔWER (layers 1–30, excluding boundary layers 0 & 31):
- cosine similarity: ρ ≈ 0.475
- MSE: ρ ≈ 0.494
- outlier_frac: ρ ≈ 0.720

### 5. Optimal alpha is ~7.0, not 6.0
The LLM.int8() threshold of α=6.0 slightly undershoots for Whisper's encoder. Peak Spearman ρ is at α≈7.0 (ρ=0.777 vs 0.711 at α=6.0). p99 of max activations is ~5.1, so α=6.0 catches <1% of dimensions.

### 6. Swedish does not behave like Danish
Despite both being North Germanic, Swedish baseline WER is 0.174 (vs Danish 0.239) and its per-layer ΔWER profile differs. The fragility hypothesis based on language family alone is too simple.

---

## Baseline WERs
```
Danish:  0.2392
English: 0.1541
Italian: 0.1474
German:  0.1708
Swedish: 0.1740
```

---

## Layer importance (top 10 least important by mean ΔWER)
Rank 1: layer 7  (mean ΔWER = -0.0045)
Rank 2: layer 9  (mean ΔWER = -0.0043)
Rank 3: layer 11 (mean ΔWER = -0.0041)
Rank 4: layer 6  (mean ΔWER = -0.0036)
Rank 5: layer 5  (mean ΔWER = -0.0036)
Rank 6: layer 10 (mean ΔWER = -0.0031)

All 6 cluster in layers 5–11. Boundary layers 0 and 31 are excluded from pruning (architecturally critical, ΔWER ~+19 and ~+2.9 respectively).

---

## Distillation rationale

Zero-shot layer removal is limited by representational drift: the pruned encoder produces out-of-distribution representations for the frozen decoder. The distillation objective is:

```
minimize MSE(student_encoder(audio), teacher_encoder(audio))
```

No labels used. People's Speech validation split (18,622 English examples, already cached at `~/.cache/huggingface/datasets/MLCommons___peoples_speech`) is the distillation corpus. English-only data is a deliberate choice — the representation alignment objective is language-agnostic. Whether this preserves multilingual performance uniformly is an open empirical question and part of what the experiment tests.

---

## How to run things

```bash
# Environment
source .venv/bin/activate   # or use `uv run` prefix

# Phase 1 (proxy metrics, ~15 min on H100)
uv run python compute_layer_metrics.py --phase 1

# Phase 2 (per-layer ΔWER, ~4 hours on H100)
uv run python compute_layer_metrics.py --phase 2

# Phase 3 (multi-layer sweep, hours)
uv run python compute_layer_metrics.py --phase 3 --ranking delta_wer

# Add a new language (merges into existing results files)
uv run python compute_layer_metrics.py --add-language fi_fi finnish

# Distillation (run on GPU 1 if GPU 0 is busy)
CUDA_VISIBLE_DEVICES=1 uv run python distill_pruned_encoder.py

# Analysis notebook
uv run jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```

---

## File map

| File | Purpose |
|------|---------|
| `compute_layer_metrics.py` | Phases 1–3 + `--add-language`. Main computation script. |
| `distill_pruned_encoder.py` | Label-free encoder distillation script. |
| `analysis.ipynb` | All figures and correlation analysis. |
| `DESIGN.md` | Original experiment design document (phases 1–4). |
| `results/delta_wers.json` | Per-layer ΔWER for all 5 languages. |
| `results/proxy_metrics.json` | Proxy metrics incl. `max_act_per_dim` for alpha sweep. |
| `results/prune_sweep_delta_wer.json` | Multi-layer sweep results (k=0 to k=7). |
| `results/distillation_log.json` | Distillation training log (created when script runs). |
| `results/distilled_pruned_model/` | Saved best checkpoint from distillation. |
| `results/fig1–4_*.pdf/png` | Paper figures. |
| `results/alpha_sensitivity.png` | Alpha threshold sensitivity plot. |

---

## Open questions / next steps

1. **Did distillation work?** Check `results/distillation_log.json`. Compare zero-shot vs post-distillation WER across all 5 languages. Key question: does Danish recover? Does Italian's improvement persist?

2. **If distillation works**: this is the paper result. The narrative is: individual-layer metrics underestimate safe pruning capacity; label-free distillation recovers the latent redundancy with no task-specific data. The multilingual fragility (Danish) and anomaly (Italian) are the stress tests.

3. **If distillation doesn't work well**: try (a) more steps, (b) lower LR, (c) intermediate layer matching instead of just final layer, (d) LoRA instead of full encoder fine-tuning.

4. **Statistical validity concern**: FLEURS test sets are ~350 samples per language. Small ΔWER differences (< 0.005) are likely within noise. Consider bootstrap confidence intervals before reporting fine-grained rankings.

5. **Finnish (`fi_fi`)**: Identified as a potentially interesting addition (Uralic, different family from all current languages). Available in `rasgaard/fleurs_test`. Would test whether fragility is about resource level or language family.

---

## Compute environment

- Cluster with LSF scheduler (bsub). Queues: `p1`, `gpul40s`, `gpua100`
- Interactive node: 2× NVIDIA H100 PCIe (80GB each)
- Python env managed with `uv` (`uv run python ...`)
- HuggingFace cache: `~/.cache/huggingface/`
