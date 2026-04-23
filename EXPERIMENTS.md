# Distillation Experiments

Label-free representation distillation for pruned Whisper-large-v3-turbo encoder.
All experiments use the same setup unless noted: 2000 steps, lr=1e-5, batch=8, AdamW wd=0.01, bfloat16, People's Speech validation split (18,622 examples).

**Distillation objective:** MSE between pruned student encoder output and frozen full teacher encoder output. No transcriptions used. Only the pruned encoder's weights are updated; decoder is frozen throughout.

**Baseline WERs (full model, zero-shot):**

| Language | WER   |
|----------|-------|
| Danish   | 0.239 |
| English  | 0.154 |
| Italian  | 0.147 |
| German   | 0.171 |
| Swedish  | 0.174 |

---

## Experiment 1 — Optimal pruning set (layers 5,6,7,9,10,11)

**Hypothesis:** The 6 least important encoder layers by mean ΔWER can be removed and partially recovered via distillation.

**Layers removed:** 5, 6, 7, 9, 10, 11 (ranked 1–6 least important across 5 languages)

**LSF job:** 28264069 — completed in ~26 min

| Language | Baseline | Zero-shot pruned | After distillation | Δ recovered |
|----------|----------|------------------|--------------------|-------------|
| Danish   | 0.239    | 0.380 (+59%)     | 0.278 (+16%)       | −43pp       |
| English  | 0.154    | 0.168 (+9%)      | 0.163 (+6%)        | −3pp        |
| Italian  | 0.147    | 0.137 (−7%)      | 0.136 (−7%)        | —           |
| German   | 0.171    | 0.186 (+9%)      | 0.183 (+7%)        | −2pp        |
| Swedish  | 0.174    | 0.239 (+37%)     | 0.201 (+16%)       | −21pp       |
| **Mean** |          | **+0.209**       | **+0.075**         |             |

**Convergence:** Loss dropped from 0.062 to ~0.027 over 2000 steps. WER gains were front-loaded — most recovery happened in the first 500 steps, with diminishing returns thereafter. Best checkpoint was the final step (2000).

**Findings:**
- Distillation recovered a large fraction of the zero-shot degradation (mean rel_Δ: +0.209 → +0.075, ~64% reduction)
- Danish and Swedish, which degraded most without distillation, also recovered the most in absolute terms
- Italian remained below baseline throughout — those pruned layers appear to have been mildly harmful for Italian
- English and German were only marginally above the 5% threshold after distillation
- Training was fast (~26 min on a single A100-class GPU), suggesting this approach is practical

---

## Experiment 2 — Next-tier pruning set (layers 2,4,8,13,14,17)

**Hypothesis:** The *next* 6 least important layers (ranks 7–12) can be pruned and recovered similarly, testing whether the rank ordering generalises.

**Layers removed:** 2, 4, 8, 13, 14, 17

**LSF job:** 28265085

| Language | Baseline | Zero-shot pruned | After distillation |
|----------|----------|------------------|--------------------|
| Danish   | 0.239    | +8.5× (>1.0)     | +3.26×             |
| English  | 0.154    | large            | +0.130             |
| Italian  | 0.147    | large            | +0.026             |
| German   | 0.171    | large            | +0.244             |
| Swedish  | 0.174    | large            | +1.128             |
| **Mean** |          | **+8.48**        | **+0.958**         |

**Findings:** Dramatically worse zero-shot degradation than Experiment 1 (+8.5 vs +0.21), confirming that rank position matters — these layers are meaningfully more important. Distillation recovers English and Italian to near-acceptable levels but Danish and Swedish remain severely degraded. The rank ordering generalises: the ΔWER-optimal set is genuinely special, not just any 6 low-scoring layers.

---

## Experiment 3 — Contiguous early block (layers 2,3,4,5,6,7)

**Hypothesis:** Removing a contiguous early block is structurally cleaner and may be easier to recover via distillation, despite including some moderately important layers (layer 3, mean ΔWER +0.0075).

**Layers removed:** 2, 3, 4, 5, 6, 7

**LSF job:** 28264827

| Language | Baseline | Zero-shot pruned | After distillation |
|----------|----------|------------------|--------------------|
| Danish   | 0.239    | +3.9×            | +14.33×            |
| English  | 0.154    | +8.4×            | +1.41×             |
| Italian  | 0.147    | +6.8×            | +8.19×             |
| German   | 0.171    | +12.5×           | +14.61×            |
| Swedish  | 0.174    | +9.3×            | +14.07×            |
| **Mean** |          | **+8.17**        | **+10.52**         |

**Findings:** Catastrophic — WERs above 1.0 (model produces near-random output). Distillation made things *worse* over training, with loss decreasing while WER increased. This suggests the encoder is learning to mimic teacher activations in a way that is incoherent for decoding — the damage from removing layers 2–7 is irreversible with this approach. Contiguous early-block removal is far more harmful than scattered removal of the same number of layers.

---

---

## Experiment 4 — Proxy-guided pruning (layers 1,2,3,4,5,6)

**Hypothesis:** The outlier fraction proxy metric can identify which layers to prune *without requiring labeled data or ΔWER sweeps*, and post-distillation performance should approximate the ground-truth-guided result.

**Why this is interesting:** In a practical deployment scenario, running a full ΔWER sweep (Experiment 1) requires held-out labeled audio per language — expensive and language-specific. The outlier fraction proxy is computed from unlabelled activations in a single forward pass. If proxy-guided pruning recovers similarly after distillation, it validates the entire proxy metric approach as a practical tool.

The proxy metric (mean outlier fraction across languages) assigns exactly 0.0 to layers 0–6 — it cannot differentiate within this tier. Selecting the 6 lowest-scoring layers (excluding layer 0, which is the input embedding and untouchable by definition) yields {1, 2, 3, 4, 5, 6}. Crucially, **layer 1 has a mean ΔWER of +0.014** — the proxy would select it as "safe to prune", but the ground-truth ranking places it as the 2nd most important layer overall. This makes Experiment 4 a direct stress test of where the proxy metric breaks down.

**Layers removed:** 1, 2, 3, 4, 5, 6

**LSF job:** 28264828

| Language | Baseline | Zero-shot pruned | After distillation (best ckpt, step 500) |
|----------|----------|------------------|------------------------------------------|
| Danish   | 0.239    | +3.17×           | +3.17×                                   |
| English  | 0.154    | +5.31×           | +5.31×                                   |
| Italian  | 0.147    | +5.72×           | +5.72×                                   |
| German   | 0.171    | +4.85×           | +4.85×                                   |
| Swedish  | 0.174    | +4.74×           | +4.74×                                   |
| **Mean** |          | **+4.88**        | **+4.76** (best ckpt = step 500)         |

**Findings:** Catastrophic — WERs near 1.0 throughout. The best checkpoint (step 500) was essentially no better than zero-shot. By step 2000 the model had degraded further (mean +6.3). The proxy metric assigns outlier_frac=0.0 to layer 1 — identical to truly unimportant layers — but ΔWER reveals it as the 2nd most important layer in the network. This is a clean illustration of the proxy's resolution limit: it cannot differentiate within the zero-outfrac early-layer tier, and blindly following it selects a critical layer. Distillation cannot recover from removing layer 1.

---

## Summary table (to be updated)

| Experiment | Layers removed          | Zero-shot mean rel_Δ | Post-distillation mean rel_Δ |
|------------|-------------------------|----------------------|------------------------------|
| 1 — optimal      | 5,6,7,9,10,11     | +0.209               | +0.075                       |
| 2 — next tier    | 2,4,8,13,14,17    | —                    | —                            |
| 3 — early block  | 2,3,4,5,6,7       | —                    | —                            |
| 4 — proxy-guided | 1,2,3,4,5,6       | +4.88                | +4.76 (unrecoverable)        |

---

## Experiment 5 — LoRA distillation (layers 5,6,7,9,10,11, rank 16)

**Hypothesis:** LoRA (rank 16) applied to encoder attention layers can recover pruning degradation with far fewer trainable parameters than full fine-tuning, potentially with a useful accuracy/efficiency tradeoff.

**Layers removed:** 5, 6, 7, 9, 10, 11 (same as Experiment 1)
**Trainable parameters:** 2.79M (0.40% of total) vs 519M for full fine-tuning

**Results at step 2000 (still converging):**

| Language | Baseline | Zero-shot | Full FT (Exp 1) | LoRA r=16 |
|----------|----------|-----------|-----------------|-----------|
| Danish   | 0.239    | +59%      | +16%            | +34%      |
| English  | 0.154    | +9%       | +6%             | +11%      |
| Italian  | 0.147    | -7%       | -7%             | -7%       |
| German   | 0.171    | +9%       | +7%             | +10%      |
| Swedish  | 0.174    | +37%      | +16%            | +25%      |
| **Mean** |          | **+0.209**| **+0.075**      | **+0.145**|

**Findings:** LoRA recovers roughly half as much as full fine-tuning (mean rel_Δ +0.145 vs +0.075) despite using 186× fewer trainable parameters. Crucially, the loss was still decreasing at step 2000 — the run had not converged, suggesting more steps or a higher rank could close the gap further. Training time was ~12 min vs ~26 min for full fine-tuning. The rank-16 constraint limits how much of the representational shift can be corrected, particularly for Danish and Swedish which require the largest adjustments.
