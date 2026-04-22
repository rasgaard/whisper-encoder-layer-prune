#!/usr/bin/env python3
"""
Compute per-layer proxy metrics (cosine similarity, MSE, outlier activations)
and ΔWER for Whisper-large-v3-turbo encoder layers across multiple languages.

Results are saved to results/ as JSON for subsequent analysis.

Usage:
    uv run python compute_layer_metrics.py            # both phases
    uv run python compute_layer_metrics.py --phase 1  # proxy metrics only
    uv run python compute_layer_metrics.py --phase 2  # ΔWER only
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from jiwer import wer as compute_wer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "openai/whisper-large-v3-turbo"
NUM_LAYERS = 32
BATCH_SIZE = 16
OUTLIER_THRESHOLD = 6.0  # standard threshold from LLM.int8()
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Maps FLEURS config name → Whisper language name for forced decoding
LANGUAGES = {
    "da_dk": "danish",
    "en_us": "english",
    "it_it": "italian",
    "de_de": "german",
}


# ---------------------------------------------------------------------------
# Data / model loading
# ---------------------------------------------------------------------------

def load_datasets():
    datasets = {}
    for lang_code in LANGUAGES:
        print(f"Loading {lang_code}...", flush=True)
        datasets[lang_code] = load_dataset("rasgaard/fleurs_test", lang_code, split="train")
    return datasets


def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, dtype=torch.float16)
    model = model.to(DEVICE).eval()
    return model, processor


def prune_encoder_layers(model, layers_to_remove: list[int]):
    pruned = copy.deepcopy(model)
    layers = pruned.model.encoder.layers
    for idx in sorted(set(layers_to_remove), reverse=True):
        del layers[idx]
    return pruned


# ---------------------------------------------------------------------------
# Phase 1 — Proxy metrics
# ---------------------------------------------------------------------------

def _outlier_dim_fraction(h: torch.Tensor, threshold: float) -> float:
    """
    Fraction of hidden dimensions whose max absolute activation across all
    positions exceeds `threshold`. Shape: [batch, seq_len, hidden_dim].
    """
    max_per_dim = h.abs().amax(dim=(0, 1))  # [hidden_dim]
    return (max_per_dim > threshold).float().mean().item()


def _amplification_ratio(h_in: torch.Tensor, h_out: torch.Tensor) -> float:
    """
    Mean ratio of max absolute activation per hidden dimension between output
    and input. Values > 1 indicate the layer amplifies activations.
    """
    max_in  = h_in.abs().amax(dim=(0, 1)).clamp(min=1e-6)  # avoid div/0
    max_out = h_out.abs().amax(dim=(0, 1))
    return (max_out / max_in).mean().item()


def compute_proxy_metrics(model, processor, datasets: dict) -> dict:
    """
    For each encoder layer compute, averaged over all samples:
      - cosine similarity between input and output hidden states
      - MSE between input and output hidden states
      - outlier dimension fraction in the output (dims with max |act| > threshold)
      - outlier count delta (output fraction minus input fraction)
      - activation amplification ratio (mean max|h_out| / mean max|h_in|)
    One encoder forward pass per batch captures all 32 layers simultaneously.
    """
    results = {}

    for lang_code in LANGUAGES:
        dataset = datasets[lang_code]
        cos_sims    = np.zeros(NUM_LAYERS)
        mses        = np.zeros(NUM_LAYERS)
        outlier_out = np.zeros(NUM_LAYERS)  # outlier fraction in output
        outlier_delta = np.zeros(NUM_LAYERS)  # output fraction − input fraction
        amp_ratios  = np.zeros(NUM_LAYERS)
        n_batches   = 0

        audios = [s["audio"]["array"] for s in dataset]
        sr = dataset[0]["audio"]["sampling_rate"]

        for batch_start in range(0, len(audios), BATCH_SIZE):
            batch_audio = audios[batch_start : batch_start + BATCH_SIZE]
            inputs = processor(
                batch_audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding="max_length",
            ).to(device=DEVICE, dtype=model.dtype)

            layer_stats: dict[int, tuple] = {}

            def make_hook(i):
                def hook(module, inp, out):
                    h_in = inp[0].detach().float()
                    # In transformers 5.x the encoder layer may return a plain
                    # tensor rather than a tuple — guard against indexing the
                    # batch dimension accidentally.
                    h_out = (out[0] if isinstance(out, (tuple, list)) else out).detach().float()

                    cos_sim = F.cosine_similarity(h_in, h_out, dim=-1).mean().item()
                    mse     = F.mse_loss(h_in, h_out).item()
                    out_frac  = _outlier_dim_fraction(h_out, OUTLIER_THRESHOLD)
                    in_frac   = _outlier_dim_fraction(h_in,  OUTLIER_THRESHOLD)
                    amp       = _amplification_ratio(h_in, h_out)

                    layer_stats[i] = (cos_sim, mse, out_frac, out_frac - in_frac, amp)
                return hook

            hooks = [
                model.model.encoder.layers[i].register_forward_hook(make_hook(i))
                for i in range(NUM_LAYERS)
            ]

            with torch.no_grad():
                model.model.encoder(inputs.input_features)

            for h in hooks:
                h.remove()

            for i, (cs, mse, o_out, o_delta, amp) in layer_stats.items():
                cos_sims[i]      += cs
                mses[i]          += mse
                outlier_out[i]   += o_out
                outlier_delta[i] += o_delta
                amp_ratios[i]    += amp
            n_batches += 1

            print(
                f"  [{lang_code}] proxy metrics: batch {n_batches}"
                f" / {(len(audios) + BATCH_SIZE - 1) // BATCH_SIZE}",
                end="\r",
                flush=True,
            )

        print()
        results[lang_code] = {
            "cos_sim":        (cos_sims      / n_batches).tolist(),
            "mse":            (mses          / n_batches).tolist(),
            "outlier_frac":   (outlier_out   / n_batches).tolist(),
            "outlier_delta":  (outlier_delta / n_batches).tolist(),
            "amp_ratio":      (amp_ratios    / n_batches).tolist(),
        }
        cs = cos_sims / n_batches
        amp = amp_ratios / n_batches
        print(f"  [{lang_code}] cos_sim  min={cs.min():.4f}  max={cs.max():.4f}")
        print(f"  [{lang_code}] amp_ratio min={amp.min():.4f}  max={amp.max():.4f}")

    return results


# ---------------------------------------------------------------------------
# Phase 2 — ΔWER
# ---------------------------------------------------------------------------

def transcribe_dataset(model, processor, dataset, lang_name: str) -> tuple[list, list]:
    audios = [s["audio"]["array"] for s in dataset]
    sr = dataset[0]["audio"]["sampling_rate"]
    references = [s["transcription"] for s in dataset]  # already normalised
    hypotheses = []

    for batch_start in range(0, len(audios), BATCH_SIZE):
        batch_audio = audios[batch_start : batch_start + BATCH_SIZE]
        inputs = processor(
            batch_audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding="max_length",
        ).to(device=DEVICE, dtype=model.dtype)

        with torch.no_grad():
            predicted_ids = model.generate(
                inputs.input_features,
                language=lang_name,
                task="transcribe",
            )

        batch_hyps = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        hypotheses.extend(h.lower().strip() for h in batch_hyps)

    return hypotheses, references


def compute_baseline_wers(model, processor, datasets: dict) -> dict:
    baseline = {}
    for lang_code, lang_name in LANGUAGES.items():
        print(f"  Baseline [{lang_code}]...", flush=True)
        hyps, refs = transcribe_dataset(model, processor, datasets[lang_code], lang_name)
        baseline[lang_code] = compute_wer(refs, hyps)
        print(f"  Baseline WER [{lang_code}]: {baseline[lang_code]:.4f}")
    return baseline


def compute_delta_wers(model, processor, datasets: dict, baseline_wers: dict) -> dict:
    """
    For each layer index, remove that single layer, run transcription on all
    languages, and record ΔWER = WER_pruned − WER_baseline.
    Results are saved incrementally after each layer.
    """
    delta_wers = {lang: [] for lang in LANGUAGES}
    layer_wers = {lang: [] for lang in LANGUAGES}

    for layer_idx in range(NUM_LAYERS):
        print(f"\nLayer {layer_idx:2d} / {NUM_LAYERS - 1}", flush=True)
        pruned = prune_encoder_layers(model, [layer_idx])
        pruned = pruned.to(DEVICE).eval()

        for lang_code, lang_name in LANGUAGES.items():
            hyps, refs = transcribe_dataset(pruned, processor, datasets[lang_code], lang_name)
            wer_score = compute_wer(refs, hyps)
            delta = wer_score - baseline_wers[lang_code]
            layer_wers[lang_code].append(wer_score)
            delta_wers[lang_code].append(delta)
            print(f"  [{lang_code}]  WER={wer_score:.4f}  ΔWER={delta:+.4f}")

        del pruned
        torch.cuda.empty_cache()

        # Incremental save so a crash doesn't lose everything
        with open(RESULTS_DIR / "delta_wers_partial.json", "w") as f:
            json.dump(
                {
                    "completed_layers": layer_idx + 1,
                    "baseline_wers": baseline_wers,
                    "layer_wers": layer_wers,
                    "delta_wers": delta_wers,
                },
                f,
                indent=2,
            )

    return delta_wers, layer_wers


# ---------------------------------------------------------------------------
# Phase 3 — Outlier-fraction-ranked multi-layer pruning sweep
# ---------------------------------------------------------------------------

BOUNDARY_LAYERS = {0, 31}  # excluded from pruning candidates
THRESHOLD = 0.05            # 5% relative WER increase


def outlier_rank(proxy_path: Path, delta_wer_path: Path) -> list[int]:
    """
    Rank interior layers (1–30) by mean outlier_frac ascending.
    Ties broken by mean ΔWER ascending (least harmful first).
    """
    with open(proxy_path) as f:
        proxy = json.load(f)
    with open(delta_wer_path) as f:
        d = json.load(f)

    langs = list(LANGUAGES.keys())
    outlier_frac = np.array([proxy[l]["outlier_frac"] for l in langs])   # [4, 32]
    delta_wers   = np.array([d["delta_wers"][l]       for l in langs])   # [4, 32]

    mean_of = outlier_frac.mean(axis=0)
    mean_dw = delta_wers.mean(axis=0)

    candidates = [i for i in range(NUM_LAYERS) if i not in BOUNDARY_LAYERS]
    return sorted(candidates, key=lambda i: (mean_of[i], mean_dw[i]))


def run_phase3(model, processor, datasets):
    print("\n=== Phase 3: Outlier-fraction-ranked pruning sweep ===")

    proxy_path    = RESULTS_DIR / "proxy_metrics.json"
    delta_path    = RESULTS_DIR / "delta_wers.json"
    baseline_path = RESULTS_DIR / "baseline_wers.json"

    for p in (proxy_path, delta_path, baseline_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}. Run phases 1 and 2 first.")

    with open(baseline_path) as f:
        baseline_wers = json.load(f)

    rank_order   = outlier_rank(proxy_path, delta_path)
    baseline_arr = np.array([baseline_wers[l] for l in LANGUAGES])

    print(f"\nPruning order (outlier_frac rank, boundary layers 0 & 31 fixed):")
    print(f"  {rank_order}\n")

    results  = []
    out_path = RESULTS_DIR / "prune_sweep.json"

    # k=0: sanity-check baseline (no layers removed)
    print("k=0  [baseline — no layers removed]")
    step = {"k": 0, "layers_removed": [], "wers": {}, "rel_delta": {}}
    for lang_code, lang_name in LANGUAGES.items():
        hyps, refs = transcribe_dataset(model, processor, datasets[lang_code], lang_name)
        wer_score  = compute_wer(refs, hyps)
        rel_delta  = (wer_score - baseline_wers[lang_code]) / baseline_wers[lang_code]
        step["wers"][lang_code]      = wer_score
        step["rel_delta"][lang_code] = rel_delta
        print(f"  [{lang_code}]  WER={wer_score:.4f}  rel_Δ={rel_delta:+.3f}")
    results.append(step)

    for k in range(1, len(rank_order) + 1):
        layers_to_remove = sorted(rank_order[:k])
        print(f"\nk={k}  removing layers {layers_to_remove}")

        pruned = prune_encoder_layers(model, layers_to_remove)
        pruned = pruned.to(DEVICE).eval()

        step    = {"k": k, "layers_removed": layers_to_remove, "wers": {}, "rel_delta": {}}
        max_rel = 0.0

        for lang_code, lang_name in LANGUAGES.items():
            hyps, refs = transcribe_dataset(pruned, processor, datasets[lang_code], lang_name)
            wer_score  = compute_wer(refs, hyps)
            rel_delta  = (wer_score - baseline_wers[lang_code]) / baseline_wers[lang_code]
            step["wers"][lang_code]      = wer_score
            step["rel_delta"][lang_code] = rel_delta
            max_rel = max(max_rel, rel_delta)
            flag = "  *** EXCEEDS 5%" if rel_delta > THRESHOLD else ""
            print(f"  [{lang_code}]  WER={wer_score:.4f}  rel_Δ={rel_delta:+.3f}{flag}")

        del pruned
        torch.cuda.empty_cache()

        results.append(step)
        with open(out_path, "w") as f:
            json.dump(
                {"baseline_wers": baseline_wers, "rank_order": rank_order, "steps": results},
                f, indent=2,
            )

        if max_rel > THRESHOLD:
            print(f"\n  → 5% threshold first exceeded at k={k}. Continuing to build full picture...")

    print(f"\nSaved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run phase 1 (proxy metrics), 2 (ΔWER), or 3 (pruning sweep). Omit to run 1+2.",
    )
    return parser.parse_args()


def run_phase1(model, processor, datasets):
    print("\n=== Phase 1: Proxy Metrics ===")
    proxy_results = compute_proxy_metrics(model, processor, datasets)
    out_path = RESULTS_DIR / "proxy_metrics.json"
    with open(out_path, "w") as f:
        json.dump(proxy_results, f, indent=2)
    print(f"Saved → {out_path}")


def run_phase2(model, processor, datasets):
    print("\n=== Phase 2: Baseline WER ===")
    baseline_wers = compute_baseline_wers(model, processor, datasets)
    with open(RESULTS_DIR / "baseline_wers.json", "w") as f:
        json.dump(baseline_wers, f, indent=2)

    print("\n=== Phase 2: ΔWER Sweep (32 layers × 4 languages) ===")
    delta_wers, layer_wers = compute_delta_wers(model, processor, datasets, baseline_wers)

    out_path = RESULTS_DIR / "delta_wers.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "baseline_wers": baseline_wers,
                "layer_wers": layer_wers,
                "delta_wers": delta_wers,
            },
            f,
            indent=2,
        )
    print(f"\nSaved → {out_path}")


def main():
    args = parse_args()

    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}\n")

    datasets = load_datasets()
    model, processor = load_model()

    if args.phase == 1:
        run_phase1(model, processor, datasets)
    elif args.phase == 2:
        run_phase2(model, processor, datasets)
    elif args.phase == 3:
        run_phase3(model, processor, datasets)
    else:
        run_phase1(model, processor, datasets)
        run_phase2(model, processor, datasets)


if __name__ == "__main__":
    main()
