#!/usr/bin/env python3
"""
Compute per-layer ΔWER for Whisper-large-v3-turbo encoder layers and run a
greedy pruning sweep to identify the optimal layer removal set.

Results are saved to results/ as JSON for subsequent analysis.

Usage:
    uv run python compute_layer_metrics.py --phase 1  # ΔWER sweep (32 layers × 4 languages)
    uv run python compute_layer_metrics.py --phase 2  # greedy pruning sweep
    uv run python compute_layer_metrics.py --add-language fr_fr french
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from jiwer import wer as compute_wer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "openai/whisper-large-v3-turbo"
NUM_LAYERS = 32
BATCH_SIZE = 16
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Maps FLEURS config name → Whisper language name for forced decoding
LANGUAGES = {
    "da_dk": "danish",
    "en_us": "english",
    "de_de": "german",
    "fr_fr": "french",
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
# Phase 1 — ΔWER sweep
# ---------------------------------------------------------------------------

def transcribe_dataset(model, processor, dataset, lang_name: str) -> tuple[list, list]:
    audios = [s["audio"]["array"] for s in dataset]
    sr = dataset[0]["audio"]["sampling_rate"]
    references = [s["transcription"] for s in dataset]
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


def run_phase1(model, processor, datasets):
    print("\n=== Phase 1: Baseline WER ===")
    baseline_wers = compute_baseline_wers(model, processor, datasets)
    with open(RESULTS_DIR / "baseline_wers.json", "w") as f:
        json.dump(baseline_wers, f, indent=2)

    print("\n=== Phase 1: ΔWER Sweep (32 layers × 4 languages) ===")
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


# ---------------------------------------------------------------------------
# Phase 2 — Greedy pruning sweep
# ---------------------------------------------------------------------------

BOUNDARY_LAYERS = {0, 31}  # excluded from pruning candidates
THRESHOLD = 0.05            # 5% relative WER increase


def delta_wer_rank(delta_wer_path: Path) -> list[int]:
    """Rank interior layers by mean ΔWER ascending (least harmful first)."""
    with open(delta_wer_path) as f:
        d = json.load(f)

    langs = list(LANGUAGES.keys())
    delta_wers = np.array([d["delta_wers"][l] for l in langs])
    mean_dw    = delta_wers.mean(axis=0)

    candidates = [i for i in range(NUM_LAYERS) if i not in BOUNDARY_LAYERS]
    return sorted(candidates, key=lambda i: mean_dw[i])


def run_phase2(model, processor, datasets):
    print("\n=== Phase 2: Greedy pruning sweep ===")

    delta_path    = RESULTS_DIR / "delta_wers.json"
    baseline_path = RESULTS_DIR / "baseline_wers.json"

    for p in (delta_path, baseline_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}. Run phase 1 first.")

    with open(baseline_path) as f:
        baseline_wers = json.load(f)

    rank_order = delta_wer_rank(delta_path)

    print(f"\nPruning order (ΔWER rank, boundary layers 0 & 31 fixed):")
    print(f"  {rank_order}\n")

    results  = []
    out_path = RESULTS_DIR / "prune_sweep_delta_wer.json"

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
# Add a new language to existing results
# ---------------------------------------------------------------------------

def run_add_language(model, processor, lang_code: str, lang_name: str):
    """
    Compute baseline WER and per-layer ΔWER for a single new language and
    merge the results into the existing JSON files produced by phase 1.
    """
    print(f"\n=== Adding language: {lang_code} ({lang_name}) ===")

    for p in (RESULTS_DIR / "delta_wers.json", RESULTS_DIR / "baseline_wers.json"):
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}. Run phase 1 first.")

    print(f"Loading dataset for {lang_code}...", flush=True)
    dataset = load_dataset("rasgaard/fleurs_test", lang_code, split="train")

    # baseline WER
    print(f"Computing baseline WER for {lang_code}...", flush=True)
    hyps, refs = transcribe_dataset(model, processor, dataset, lang_name)
    baseline_wer = compute_wer(refs, hyps)
    print(f"  Baseline WER [{lang_code}]: {baseline_wer:.4f}")

    with open(RESULTS_DIR / "baseline_wers.json") as f:
        baseline_wers = json.load(f)
    baseline_wers[lang_code] = baseline_wer
    with open(RESULTS_DIR / "baseline_wers.json", "w") as f:
        json.dump(baseline_wers, f, indent=2)

    # per-layer ΔWER
    print(f"\nComputing per-layer ΔWER for {lang_code}...", flush=True)
    layer_wers_lang = []
    delta_wers_lang = []

    for layer_idx in range(NUM_LAYERS):
        print(f"  Layer {layer_idx:2d} / {NUM_LAYERS - 1}", flush=True)
        pruned = prune_encoder_layers(model, [layer_idx])
        pruned = pruned.to(DEVICE).eval()

        hyps, refs = transcribe_dataset(pruned, processor, dataset, lang_name)
        wer_score = compute_wer(refs, hyps)
        delta = wer_score - baseline_wer
        layer_wers_lang.append(wer_score)
        delta_wers_lang.append(delta)
        print(f"    WER={wer_score:.4f}  ΔWER={delta:+.4f}")

        del pruned
        torch.cuda.empty_cache()

    with open(RESULTS_DIR / "delta_wers.json") as f:
        existing = json.load(f)
    existing["baseline_wers"][lang_code] = baseline_wer
    existing["layer_wers"][lang_code]    = layer_wers_lang
    existing["delta_wers"][lang_code]    = delta_wers_lang
    with open(RESULTS_DIR / "delta_wers.json", "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\nDone. Merged {lang_code} into results/ files.")
    print("You can now re-run phase 2 to include this language in the sweep.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=None,
        help="Run phase 1 (ΔWER sweep) or phase 2 (greedy pruning sweep).",
    )
    parser.add_argument(
        "--add-language",
        nargs=2,
        metavar=("LANG_CODE", "LANG_NAME"),
        help="Compute and merge results for one new language, e.g. --add-language sv_se swedish",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}\n")

    if args.add_language:
        lang_code, lang_name = args.add_language
        model, processor = load_model()
        run_add_language(model, processor, lang_code, lang_name)
        return

    datasets = load_datasets()
    model, processor = load_model()

    if args.phase == 1:
        run_phase1(model, processor, datasets)
    elif args.phase == 2:
        run_phase2(model, processor, datasets)
    else:
        run_phase1(model, processor, datasets)
        run_phase2(model, processor, datasets)


if __name__ == "__main__":
    main()
