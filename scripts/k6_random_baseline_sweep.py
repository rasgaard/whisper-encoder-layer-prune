#!/usr/bin/env python3
"""
Zero-shot WER for random k=6 layer selections from the first half of the encoder
(layers 1–15), compared against the optimised set [5,6,7,9,11,12].

Layer 0 is excluded from the pool: it is known to be catastrophically important
(removing it alone causes WER > 20×) and would make every sample containing it
uninterpretable.

Results saved to results/k6_random_baseline_sweep.json.

Run:
    uv run python tools/k6_random_baseline_sweep.py [--n-random 10] [--seed 42]
"""

import argparse
import copy
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from jiwer import wer as compute_wer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "openai/whisper-large-v3-turbo"
RESULTS  = Path("results")
OUT_FILE = RESULTS / "k6_random_baseline_sweep.json"

OPTIMAL  = [5, 6, 7, 9, 11, 12]
POOL     = list(range(1, 16))   # layers 1–15 (first half, layer 0 excluded)

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
        out[lc] = {
            "wer":       w,
            "delta_pp":  (w - baselines[lc]) * 100,
            "rel_delta": (w - baselines[lc]) / baselines[lc],
        }
        print(f"  {lc}: WER={w:.4f}  Δ={out[lc]['delta_pp']:+.2f}pp")
    mean_pp = sum(v["delta_pp"] for v in out.values()) / len(out)
    print(f"  mean Δ = {mean_pp:+.2f}pp")
    return out, mean_pp


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-random", type=int, default=10,
                   help="Number of random layer sets to evaluate (default: 10)")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--device",   default=None)
    args = p.parse_args()

    global DEVICE
    if args.device:
        DEVICE = args.device

    random.seed(args.seed)

    # Sample the full n_random set deterministically from the seed
    random_sets = []
    while len(random_sets) < args.n_random:
        candidate = sorted(random.sample(POOL, 6))
        if candidate not in random_sets and candidate != OPTIMAL:
            random_sets.append(candidate)

    # Resume: load existing results and skip already-evaluated sets
    already_done = set()
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            results = json.load(f)
        for r in results["random_sets"]:
            already_done.add(tuple(r["layers"]))
        if results["optimal"] is not None:
            already_done.add(tuple(OPTIMAL))
        print(f"Resuming — {len(results['random_sets'])} random sets already done, "
              f"{len(random_sets) - len(results['random_sets'])} remaining.")
    else:
        results = {"pool": POOL, "seed": args.seed, "optimal": None, "random_sets": []}

    print(f"Device: {DEVICE}")
    print(f"Optimal set: {OPTIMAL}")
    print(f"Random pool: layers {POOL[0]}–{POOL[-1]}")
    to_run = sum(1 for s in random_sets if tuple(s) not in already_done)
    print(f"Target n={args.n_random}, to run: {to_run} configs"
          f"{' + optimal' if tuple(OPTIMAL) not in already_done else ''}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("\nLoading full model...")
    full_model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    for param in full_model.parameters():
        param.requires_grad_(False)
    full_model = full_model.to(DEVICE).eval()

    print("Loading FLEURS test sets...")
    fleurs = {lc: load_dataset("rasgaard/fleurs_test", lc, split="train")
              for lc in LANGUAGES}

    with open(RESULTS / "baseline_wers.json") as f:
        baselines = json.load(f)

    # Optimal reference (skip if already done)
    if tuple(OPTIMAL) not in already_done:
        print("\n=== Optimal k=6 reference ===")
        m = prune(full_model, OPTIMAL)
        wer_opt, mean_opt = eval_config(m, processor, fleurs, baselines,
                                        f"optimal {OPTIMAL}")
        results["optimal"] = {"layers": OPTIMAL, "wer": wer_opt, "mean_delta_pp": mean_opt}
        del m; torch.cuda.empty_cache()
        with open(OUT_FILE, "w") as f:
            json.dump(results, f, indent=2)
    else:
        mean_opt = results["optimal"]["mean_delta_pp"]
        print(f"\nOptimal already evaluated: {mean_opt:+.2f} pp (skipping)")

    # Random sets
    for i, layers in enumerate(random_sets):
        if tuple(layers) in already_done:
            print(f"  skipping {layers} (already done)")
            continue
        label = f"random {i+1}/{args.n_random}: {layers}"
        print(f"\n=== {label} ===")
        m = prune(full_model, layers)
        wer_r, mean_r = eval_config(m, processor, fleurs, baselines, label)
        results["random_sets"].append({
            "layers": layers, "wer": wer_r, "mean_delta_pp": mean_r
        })
        del m; torch.cuda.empty_cache()

        with open(OUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {OUT_FILE}")

    # Summary
    random_means = [r["mean_delta_pp"] for r in results["random_sets"]]
    print(f"\n=== Summary ===")
    print(f"Optimal:       {mean_opt:+.2f} pp")
    print(f"Random min:    {min(random_means):+.2f} pp")
    print(f"Random mean:   {sum(random_means)/len(random_means):+.2f} pp")
    print(f"Random max:    {max(random_means):+.2f} pp")


if __name__ == "__main__":
    main()
