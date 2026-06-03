#!/usr/bin/env python3
"""
Zero-shot WER sweep: k=6 base set + each of N candidate 7th layers.

Loads the full model once, then for each candidate prunes [BASE + candidate]
and evaluates WER on FLEURS (da/en/de/fr) without any distillation.

Results saved to results/k7_zero_shot_sweep.json.

Run:
    uv run python tools/k7_zero_shot_sweep.py
"""

import argparse
import copy
import json
from pathlib import Path

import torch
from datasets import load_dataset
from jiwer import wer as compute_wer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID  = "openai/whisper-large-v3-turbo"
RESULTS   = Path("results")
OUT_FILE  = RESULTS / "k7_zero_shot_sweep.json"

BASE_LAYERS = [5, 6, 7, 9, 11, 12]   # optimal k=6 set

# 10 candidates spanning the importance spectrum:
#   10, 4, 13  — just outside k=6, tier 7-9
#   14, 8      — mid tier (14 tested in variants: ~+0.170)
#   15, 20, 25 — later layers, should be safe
#   2, 1       — high-importance early layers, should be catastrophic
CANDIDATES = [10, 4, 13, 14, 8, 15, 20, 25, 2, 1]

LANGUAGES = {
    "da_dk": "danish",
    "en_us": "english",
    "de_de": "german",
    "fr_fr": "french",
}

EVAL_BATCH = 16


def prune(model, layers_to_remove: list[int]):
    m = copy.deepcopy(model)
    for idx in sorted(set(layers_to_remove), reverse=True):
        del m.model.encoder.layers[idx]
    return m


def transcribe(model, processor, dataset, lang_name: str):
    audios = [s["audio"]["array"] for s in dataset]
    sr     = dataset[0]["audio"]["sampling_rate"]
    refs   = [s["transcription"] for s in dataset]
    hyps   = []
    for start in range(0, len(audios), EVAL_BATCH):
        batch = audios[start : start + EVAL_BATCH]
        inp = processor(
            batch, sampling_rate=sr, return_tensors="pt", padding="max_length",
        ).to(device=DEVICE, dtype=torch.bfloat16)
        with torch.no_grad():
            ids = model.generate(inp.input_features, language=lang_name, task="transcribe")
        hyps.extend(h.lower().strip() for h in processor.batch_decode(ids, skip_special_tokens=True))
    return hyps, refs


def eval_config(model, processor, fleurs, baselines, label):
    print(f"\n--- {label} ---")
    out = {}
    for lc, ln in LANGUAGES.items():
        hyps, refs = transcribe(model, processor, fleurs[lc], ln)
        w = compute_wer(refs, hyps)
        delta_pp  = (w - baselines[lc]) * 100
        rel_delta = (w - baselines[lc]) / baselines[lc]
        out[lc] = {"wer": w, "delta_pp": delta_pp, "rel_delta": rel_delta}
        print(f"  {lc}: WER={w:.4f}  Δ={delta_pp:+.2f}pp  rel={rel_delta:+.3f}")
    mean_pp  = sum(v["delta_pp"]  for v in out.values()) / len(out)
    mean_rel = sum(v["rel_delta"] for v in out.values()) / len(out)
    print(f"  mean: Δ={mean_pp:+.2f}pp  rel={mean_rel:+.3f}")
    return out, mean_pp, mean_rel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None)
    args = p.parse_args()

    global DEVICE
    if args.device:
        DEVICE = args.device

    print(f"Device: {DEVICE}")
    print(f"Base layers (k=6): {BASE_LAYERS}")
    print(f"Candidate 7th layers: {CANDIDATES}\n")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Loading full model...")
    full_model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    for param in full_model.parameters():
        param.requires_grad_(False)
    full_model = full_model.to(DEVICE).eval()

    print("Loading FLEURS test sets...")
    fleurs = {lc: load_dataset("rasgaard/fleurs_test", lc, split="train") for lc in LANGUAGES}

    with open(RESULTS / "baseline_wers.json") as f:
        baselines = json.load(f)

    results = {"base_layers": BASE_LAYERS, "candidates": [], "k6_reference": None}

    # k=6 reference (no 7th layer)
    print("\n=== k=6 reference ===")
    m6 = prune(full_model, BASE_LAYERS)
    wer6, mean_pp6, mean_rel6 = eval_config(m6, processor, fleurs, baselines, f"k=6 {BASE_LAYERS}")
    results["k6_reference"] = {"layers": BASE_LAYERS, "wer": wer6, "mean_delta_pp": mean_pp6, "mean_rel_delta": mean_rel6}
    del m6
    torch.cuda.empty_cache()

    # k=7 candidates
    for cand in CANDIDATES:
        layers = sorted(BASE_LAYERS + [cand])
        label  = f"k=7 (7th = layer {cand}) → {layers}"
        print(f"\n=== {label} ===")
        m7 = prune(full_model, layers)
        wer7, mean_pp7, mean_rel7 = eval_config(m7, processor, fleurs, baselines, label)
        results["candidates"].append({
            "seventh_layer": cand,
            "layers": layers,
            "wer": wer7,
            "mean_delta_pp": mean_pp7,
            "mean_rel_delta": mean_rel7,
        })
        del m7
        torch.cuda.empty_cache()

        with open(OUT_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  → saved to {OUT_FILE}")

    print(f"\nDone. Results in {OUT_FILE}")

    # Summary table
    print("\n=== Summary ===")
    print(f"{'Config':<30} {'Mean Δpp':>10} {'Mean rel':>10}")
    print(f"{'k=6 (base)':<30} {mean_pp6:>+10.2f} {mean_rel6:>+10.3f}")
    for r in results["candidates"]:
        lbl = f"k=7 (7th=L{r['seventh_layer']})"
        print(f"{lbl:<30} {r['mean_delta_pp']:>+10.2f} {r['mean_rel_delta']:>+10.3f}")


if __name__ == "__main__":
    main()
