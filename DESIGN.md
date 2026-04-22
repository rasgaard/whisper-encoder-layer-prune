# Encoder Layer Redundancy in Whisper: A Multilingual Analysis
## Experiment Design Document

---

## 1. Motivation and Observation

Whisper-large-v3-turbo contains 32 transformer encoder layers, each consisting of self-attention and feed-forward sub-layers connected by residual connections. Preliminary analysis on English (LibriSpeech) reveals that many of these layers exhibit near-identity behavior: the cosine similarity between a layer's input and output is close to 1, and the mean squared error between them is close to 0.

This is not incidental. The residual connection in each transformer layer means the output is:

```
y = x + F(x)
```

If `F(x)` is small relative to `x`, the layer effectively passes its input through unchanged. This creates the conditions for structural redundancy: layers that are nominally present but contribute negligibly to the transformation of representations.

This phenomenon has been documented in large language models (LLMs), but its presence in speech encoder models—and its interaction with multilingual representations—remains underexplored.

---

## 2. Related Work

**Layer pruning in LLMs.** A growing body of work has shown that large proportions of LLM layers can be removed with minimal degradation. ShortGPT (Men et al., 2024) introduces the Block Influence (BI) metric, defined as 1 minus the cosine similarity between a layer's input and output, and uses it to rank and prune layers. They find that over 25% of layers can be removed from models like LLaMA-2 and Mistral with negligible perplexity increase.

**Limitations of cosine similarity as an importance metric.** "Rethinking Layer Relevance in Large Language Models Beyond Cosine Similarity" argues that cosine similarity is a poor proxy for layer importance because it conflates magnitude changes with directional changes in representation space. They propose using accuracy drop (ΔACC) upon layer removal as the ground-truth importance signal, and show that cosine similarity rankings frequently disagree with accuracy-based rankings. This motivates using WER difference as the primary metric in our work.

**Structured pruning for speech models.** While there is prior work on pruning Whisper via knowledge distillation (e.g., distil-whisper) or attention head pruning, direct encoder layer removal based on functional importance has not been systematically studied—particularly not through a multilingual lens.

---

## 3. Research Questions

**RQ1: Are cosine similarity and MSE reliable proxies for encoder layer importance in Whisper?**

Specifically, does the ranking of layers by cosine similarity (or MSE) agree with their ranking by ΔWER? This directly tests the claim from the "Rethinking Layer Relevance" paper in the context of a speech encoder rather than an LLM.

**RQ2: Is encoder layer importance consistent across languages?**

Do the same layers emerge as redundant regardless of the input language, or does importance vary by language? If importance is highly consistent, it supports the existence of a universal prunable set.

**RQ3: How much of the encoder can be removed with negligible WER impact?**

What is the maximum number of layers that can be dropped—using individual ΔWER ranking—before WER degrades beyond an acceptable threshold (e.g., <5% relative WER increase on any language)?

**RQ4: Does a language-agnostic prunable set exist?**

Is there a subset of layers that is simultaneously unimportant across all evaluated languages? This would be the practical target for compression: a single pruned model that retains multilingual capability.

---

## 4. Experimental Design

### 4.1 Model and Data

- **Model**: `openai/whisper-large-v3-turbo` (32 encoder layers)
- **Languages**: Danish (`da_dk`), English (`en_us`), Italian (`it_it`), German (`de_de`) from `rasgaard/fleurs_test`
- These four cover two Germanic languages (one with limited training data relative to English), one Romance language, and the dominant high-resource language in Whisper's training set.

### 4.2 Phase 1 — Proxy Metric Analysis

**Goal**: Establish a structured baseline of layer-level activity.

For each encoder layer `i` (0–31) and each language:

1. Run the encoder with forward hooks to collect input and output hidden states for layer `i` across all test samples.
2. Compute per-sample:
   - **Cosine similarity**: `cos_sim(h_in, h_out)` averaged over the sequence dimension
   - **MSE**: `mean((h_in - h_out)^2)` averaged over the sequence and hidden dimension
3. Aggregate across samples to get a per-layer, per-language distribution.

This produces a `[32 × 4]` heatmap of proxy importance scores and allows visual inspection of which layers are candidates for removal.

### 4.3 Phase 2 — ΔWER-Based Importance Scoring

**Goal**: Measure the true functional importance of each layer.

For each layer `i` and each language:

1. Create a pruned model with layer `i` removed using `prune_encoder_layers(model, [i])`.
2. Run full transcription on the FLEURS test split for that language.
3. Compute WER for the pruned model.
4. **Layer importance score**: `ΔWER_i = WER_pruned_i - WER_baseline`

A positive ΔWER means removing the layer hurt performance; a score near 0 means the layer is functionally redundant.

This produces a `[32 × 4]` matrix of ΔWER scores.

### 4.4 Phase 3 — Cross-Lingual Analysis

1. **Correlation analysis**: Compute Spearman rank correlation between cosine similarity rankings and ΔWER rankings, per language and overall. This answers RQ1.
2. **Consistency analysis**: Compute rank correlation of ΔWER scores across language pairs. A high correlation implies consistent importance ordering (RQ2).
3. **Aggregated importance ranking**: Average ΔWER across languages to get a single per-layer importance score. Identify the bottom-k layers by this score.
4. **Language-agnostic prunable set**: Find layers with ΔWER < threshold on *all* languages simultaneously (RQ4).

### 4.5 Phase 4 — Pruning Capacity Sweep

**Goal**: Identify how many layers can be dropped before multilingual performance degrades meaningfully.

**Significance threshold**: A WER increase is considered significant if any language exceeds **5% relative WER increase** over the baseline (e.g., baseline WER of 10% → threshold at 10.5%). This is a conventional threshold in ASR compression work and can be revisited.

**Procedure**:

1. Sort all 32 layers by aggregated ΔWER (ascending) — least important first.
2. For k = 1, 2, 3, ..., 31 (every possible removal count):
   - Remove the bottom-k layers simultaneously from the model.
   - Evaluate WER on all four languages.
   - Record max relative WER increase across languages.
3. Plot **WER vs. k** (one curve per language, plus max-across-languages) to find the "elbow" — the point at which removing one more layer causes a sharp WER increase.
4. Report **k\*** as the maximum number of layers that can be removed while keeping all languages within the 5% relative threshold.

This answers the central applied question directly: "Whisper-large-v3-turbo can have X of its 32 encoder layers removed with less than 5% relative WER degradation across Danish, English, Italian, and German."

The sweep also reveals whether the degradation is gradual (smooth elbow) or abrupt (cliff), which has implications for how safely further pruning could be pushed in future work.

---

## 5. Evaluation and Reporting

For each experiment, report:

| Metric | Description |
|--------|-------------|
| Baseline WER | Per-language WER of the unmodified model |
| Per-layer ΔWER | Heatmap across layers and languages |
| Cos-sim / ΔWER correlation | Spearman ρ, per language and overall |
| Cross-language importance correlation | Spearman ρ between language pairs |
| Pruned model WER | WER after multi-layer removal, per language |
| Compression ratio | Layers retained / 32 |

---

## 6. Expected Contributions

1. **Empirical evidence** that a substantial fraction of Whisper's encoder layers are functionally near-redundant, measured by a task-level metric (WER) rather than a representational proxy.
2. **A direct test** of the cosine similarity critique in a speech encoder setting, extending the "Rethinking Layer Relevance" findings beyond LLMs.
3. **A multilingual importance profile** of Whisper's encoder layers, identifying whether redundancy is language-agnostic or language-specific.
4. **A practical pruning recipe**: a specific subset of layers that can be removed to yield a smaller model with negligible multilingual WER degradation.

---

## 7. Proxy Metric Formulations

Let layer $i$ have input hidden states $\mathbf{H}^{(i)} \in \mathbb{R}^{B \times T \times D}$ and output $\mathbf{H}^{(i+1)} \in \mathbb{R}^{B \times T \times D}$, where $B$ is batch size, $T$ is sequence length, and $D$ is the hidden dimension (1280 for Whisper-large-v3-turbo). All metrics are averaged over samples.

### Cosine Similarity (baseline)

$$\text{CosSim}(i) = \frac{1}{BT} \sum_{b=1}^{B} \sum_{t=1}^{T} \frac{\mathbf{h}_{b,t}^{(i)} \cdot \mathbf{h}_{b,t}^{(i+1)}}{\|\mathbf{h}_{b,t}^{(i)}\|\ \|\mathbf{h}_{b,t}^{(i+1)}\|}$$

A value near 1 indicates the layer's output is nearly parallel to its input — the hallmark of near-identity behaviour. Used as an importance proxy by ShortGPT (Men et al., 2024).

### Mean Squared Error

$$\text{MSE}(i) = \frac{1}{BTD} \sum_{b,t,d} \left( H_{b,t,d}^{(i+1)} - H_{b,t,d}^{(i)} \right)^2$$

Measures absolute magnitude of change rather than angular difference. Sensitive to scale but complementary to cosine similarity.

### Outlier Dimension Fraction (proposed)

Let $\alpha > 0$ be an outlier threshold (we use $\alpha = 6.0$, following Dettmers et al., 2022). Define the maximum absolute activation per hidden dimension:

$$m_d^{(i+1)} = \max_{b,\, t}\ \left| H_{b,t,d}^{(i+1)} \right|, \quad d = 1, \ldots, D$$

The **outlier dimension fraction** of layer $i$'s output is:

$$\text{OutlierFrac}(i) = \frac{1}{D} \sum_{d=1}^{D} \mathbf{1}\!\left[ m_d^{(i+1)} > \alpha \right]$$

Layers that produce many high-magnitude outlier dimensions are hypothesised to be constructing distinctive, task-relevant features and therefore be more important.

### Outlier Delta

$$\Delta\text{Outlier}(i) = \text{OutlierFrac}(i) - \frac{1}{D}\sum_{d=1}^{D} \mathbf{1}\!\left[ m_d^{(i)} > \alpha \right]$$

Positive values indicate the layer *creates* new outlier dimensions; negative values indicate it suppresses them.

### Activation Amplification Ratio

$$\text{AmpRatio}(i) = \frac{1}{D} \sum_{d=1}^{D} \frac{m_d^{(i+1)}}{\max\!\left(m_d^{(i)},\, \varepsilon\right)}$$

where $\varepsilon = 10^{-6}$ prevents division by zero. Values near 1 indicate the layer preserves activation magnitudes (near-identity); values substantially above 1 indicate amplification.

### Ground-Truth Importance: ΔWER

$$\Delta\text{WER}(i) = \text{WER}(\mathcal{M}_{-i}) - \text{WER}(\mathcal{M})$$

where $\mathcal{M}_{-i}$ is the model with encoder layer $i$ removed and $\mathcal{M}$ is the unmodified model. Higher $\Delta\text{WER}$ indicates a more important layer.

---

## 8. Open Questions and Risks

- **Non-additive effects**: ΔWER from individual layer removal may not predict multi-layer removal well. Phase 4 is specifically designed to check this, but it may require revisiting the selection strategy.
- **FLEURS test set size**: If the per-language test sets are small, WER estimates will be noisy. May need to report confidence intervals or bootstrap estimates.
- **Tokenization and decoding**: WER depends on the decoding configuration (greedy vs. beam search, language token forcing vs. language detection). These must be held constant across all evaluations.
- **Layer interaction**: Removing a single layer may expose interactions with adjacent layers that are not apparent from single-layer ΔWER scores. Worth noting as a limitation.
