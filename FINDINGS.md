# Encoder Layer Redundancy and Distillation Recovery in Whisper-large-v3-turbo

A summary of all experimental findings from this project.

---

## Motivation

Whisper-large-v3-turbo has a 32-layer transformer encoder. Not all layers contribute equally to transcription quality. If the least useful layers can be identified and removed, inference becomes faster and the model smaller — without retraining from scratch. This project investigates which layers are redundant, how well a cheap proxy metric can predict redundancy, and whether label-free distillation can recover the quality lost by pruning.

---

## Phase 1 — Layer Importance via ΔWER

Each encoder layer was removed in isolation and WER was measured on FLEURS test sets for five languages: Danish, English, Italian, German, and Swedish. The change in WER relative to the full model (ΔWER) measures each layer's contribution.

**Key findings:**

- Layer importance is highly non-uniform. Layers 0 and 31 are catastrophically important (removing either causes WER to increase by 20–4× respectively). The middle encoder layers (roughly 5–14) are largely redundant.
- The six least important layers by mean ΔWER across all five languages are **5, 6, 7, 9, 10, 11** — all clustered in the middle of the encoder.
- Importance profiles are broadly consistent across languages, but not identical. Italian is a consistent outlier: several mid-layers appear to slightly *hurt* Italian performance, so removing them yields a small WER improvement.
- Danish is consistently the most sensitive language — its WER degrades more than any other language when important layers are removed.
- Swedish was added as a North Germanic language similar to Danish, with the hypothesis that they would behave alike. They do not — Swedish is substantially less sensitive, suggesting the similarity is linguistic rather than acoustic/representational.

**Layer importance ranking (mean ΔWER, ascending = least important):**

| Rank | Layer | Mean ΔWER |
|------|-------|-----------|
| 1    | 7     | −0.00346  |
| 2    | 9     | −0.00326  |
| 3    | 11    | −0.00291  |
| 4    | 5     | −0.00278  |
| 5    | 6     | −0.00254  |
| 6    | 10    | −0.00205  |
| 7    | 4     | −0.00176  |
| 8    | 13    | −0.00093  |
| …    | …     | …         |
| —    | 1     | +0.01412  |
| —    | 0     | +19.843   |

---

## Phase 2 — Proxy Metric: Outlier Dimension Fraction

Running a full ΔWER sweep requires labeled audio per language. A practical deployment scenario needs a cheaper signal. We evaluated the **outlier dimension fraction** — the fraction of hidden dimensions with maximum absolute activation exceeding a threshold α — as a proxy for layer importance.

**Key findings:**

- The proxy correlates meaningfully with ΔWER (Spearman ρ = 0.777 at the optimal threshold α = 7.0), better than the LLM-literature default of α = 6.0 (ρ = 0.711).
- The proxy has a hard resolution limit in the early-layer tier: layers 0–6 all score exactly 0.0 outlier fraction, making them indistinguishable from one another. The proxy can identify which later layers are important but cannot differentiate within the early cluster.
- This resolution failure has a concrete consequence: a naive proxy-guided pruning selection includes layer 1 (which scores 0.0 but has mean ΔWER = +0.014, the 2nd most important layer overall). Experiment 4 below shows this is unrecoverable.
- The peak α sensitivity (α = 7.0) is slightly higher than the LLM default, consistent with Whisper's encoder producing somewhat larger activation magnitudes than typical language model activations.

---

## Phase 3 — Pruning Capacity

Removing multiple layers simultaneously was swept from k=1 to k=16 (always removing the k least important by ΔWER). Results across five languages:

- WER degrades gracefully up to **k=6** (removing layers 5,6,7,9,10,11), with mean rel_Δ of +0.174 across four languages.
- At k=7 there is a sharp cliff: adding layer 4 to the removed set pushes mean rel_Δ to +3.124 — an **18× jump** in a single step.
- The full sweep (4 languages; Swedish added later):

| k | Layers removed | Mean rel_Δ |
|---|---------------|------------|
| 4 | 5,7,9,11 | +0.006 |
| 5 | 5,6,7,9,11 | +0.037 |
| **6** | **5,6,7,9,10,11** | **+0.174** |
| **7** | **4,5,6,7,9,10,11** | **+3.124** |

- **k=6 (18.75% of encoder layers) is the practical pruning limit** before quality deteriorates unacceptably.

---

## Phase 4 — Label-Free Representation Distillation

Removing encoder layers causes *representational drift*: the pruned encoder's output distribution shifts away from what the frozen decoder expects, degrading decoding quality. To correct this without task-specific data, we fine-tune only the pruned encoder to minimise MSE between its output hidden states and those of the full teacher encoder, using People's Speech (English, unlabelled audio) as the distillation corpus.

**Setup:** 2000 steps, lr=1e-5, batch=8, AdamW, bfloat16. ~26 minutes on a single A100-class GPU.

**Results (layers 5,6,7,9,10,11 removed):**

| Language | Baseline WER | Zero-shot pruned | After distillation |
|----------|-------------|------------------|--------------------|
| Danish   | 0.239       | 0.380 (+59%)     | 0.278 (+16%)       |
| English  | 0.154       | 0.168 (+9%)      | 0.163 (+6%)        |
| Italian  | 0.147       | 0.137 (−7%)      | 0.136 (−7%)        |
| German   | 0.171       | 0.186 (+9%)      | 0.183 (+7%)        |
| Swedish  | 0.174       | 0.239 (+37%)     | 0.201 (+16%)       |
| **Mean** |             | **+0.209**       | **+0.075**         |

Distillation recovered ~64% of the zero-shot degradation. Most gains occurred in the first 500 steps; the curve flattened significantly after that. Danish and Swedish — the most degraded languages — recovered the most in absolute terms. The distillation corpus is English-only, yet multilingual recovery is strong, suggesting the re-alignment happens at a representational level rather than a language-specific one.

---

## Phase 5 — Comparative Pruning Experiments

To understand whether the optimal pruning set is genuinely special or whether any 6 low-importance layers behave similarly, three additional configurations were tested with identical distillation setup:

### Next-tier layers (2,4,8,13,14,17)
The 6 layers ranked 7–12 by ΔWER. Zero-shot degradation: mean +8.48 (40× worse than Exp 1). After distillation: mean +0.958. English and Italian partially recover; Danish and Swedish remain severely degraded. **Conclusion:** rank position matters — these layers are an order of magnitude more important than the optimal set, and distillation cannot bridge that gap.

### Contiguous early block (2,3,4,5,6,7)
Removes a clean contiguous block from the early encoder. Zero-shot: mean +8.17. After distillation: mean +10.52 — *worse than zero-shot*. WERs exceed 1.0 (model outputs near-random text). Training loss decreases while WER worsens, indicating the encoder is learning activations that satisfy the MSE objective but are incoherent for decoding. **Conclusion:** contiguous early-block removal destroys the residual stream structure in a way the MSE objective cannot repair.

### Proxy-guided selection (1,2,3,4,5,6)
Layers selected by lowest outlier fraction score, excluding layer 0. Includes layer 1, which the proxy cannot distinguish from unimportant layers but which has ΔWER = +0.014. Zero-shot: mean +4.88. Best checkpoint (step 500): mean +4.76. No meaningful recovery at any point. **Conclusion:** direct illustration of the proxy's resolution failure. The method can identify safe layers in the later encoder but is blind to which early layers are critical.

**Summary of all experiments:**

| Configuration         | Layers removed    | Zero-shot | Post-distillation |
|-----------------------|-------------------|-----------|-------------------|
| Optimal (Exp 1)       | 5,6,7,9,10,11     | +0.209    | **+0.075**        |
| Next tier             | 2,4,8,13,14,17    | +8.48     | +0.958            |
| Contiguous early      | 2,3,4,5,6,7       | +8.17     | +10.52 (diverged) |
| Proxy-guided          | 1,2,3,4,5,6       | +4.88     | +4.76 (failed)    |

The optimal set is genuinely special. Distillation has a repair budget: it can correct modest drift but cannot reconstruct a broken representation.

---

## Phase 6 — LoRA Distillation

As a parameter-efficiency experiment, LoRA (rank 16) was applied to the encoder's attention projections (q_proj, v_proj) instead of full fine-tuning, using the same optimal pruning set.

- **Trainable parameters:** 2.79M (0.40% of total) vs 519M for full fine-tuning
- **Training time:** ~12 minutes vs ~26 minutes
- **Result at 2000 steps:** mean rel_Δ = +0.145 (vs +0.075 for full fine-tuning)
- The run had not converged at 2000 steps — loss was still decreasing

LoRA recovers roughly half as much as full fine-tuning with 186× fewer trainable parameters and half the training time. The rank-16 constraint limits recovery, particularly for Danish and Swedish. A higher rank or more steps may close the gap. Useful as a fast approximation when full fine-tuning is too expensive.

---

## Phase 7 — Efficiency Gains

Removing 6 of 32 encoder layers (18.75%) yields measurable runtime and memory benefits:

| Metric                         | Full model | Pruned model | Gain     |
|-------------------------------|------------|--------------|----------|
| Total parameters               | 808.9M     | 690.8M       | −14.6%   |
| Encoder parameters             | 637.0M     | 518.9M       | −18.5%   |
| Model size (bfloat16)          | 1543 MB    | 1318 MB      | −225 MB  |
| Encoder forward pass (batch=8) | 81.4 ms    | 67.1 ms      | −17.6%   |
| Full transcription (batch=8)   | 327 ms     | 212 ms       | **−35.3%** |
| Throughput (per sample)        | 40.9 ms    | 26.4 ms      | **1.55×** |

The transcription speedup (1.55×) substantially exceeds the encoder-only speedup (1.21×) because the encoder runs once per sequence while the autoregressive decoder dominates wall time — a faster encoder pass disproportionately reduces total latency at practical batch sizes.

---

## Summary

A 6-layer (18.75%) pruning of Whisper-large-v3-turbo's encoder, followed by 26 minutes of label-free representation distillation, yields a model that is **1.55× faster at transcription**, **225 MB smaller**, and degrades mean WER by only +7.5% relative across five languages. The pipeline requires no transcriptions — only unlabelled audio for distillation.

The outlier fraction proxy metric (α = 7.0) identifies layer importance with Spearman ρ = 0.777 against ground-truth ΔWER but has a resolution failure in the early encoder layers, where all layers score identically despite having very different importance. This failure mode is not benign: blindly following the proxy selects a critical layer and produces a model that cannot be recovered by distillation.

The complete pipeline is:
1. Sweep ΔWER per layer on a small held-out set (ground truth, or use proxy for later layers)
2. Remove the k least important layers (k ≤ 6 for this model)
3. Run 2000-step MSE distillation on unlabelled audio (~26 min)
4. Deploy at 1.55× throughput with modest WER cost
