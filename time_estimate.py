#!/usr/bin/env python3
"""Quick timing benchmark to estimate total compute_layer_metrics.py runtime."""

import time
import torch
from datasets import load_dataset
from compute_layer_metrics import (
    BATCH_SIZE, DEVICE, LANGUAGES, NUM_LAYERS,
    load_model, prune_encoder_layers, transcribe_dataset,
)
import torch.nn.functional as F

N_PROBE_LAYERS = 3   # layers to time for the ΔWER sweep estimate
N_PROXY_BATCHES = 3  # batches to time for proxy metrics estimate

def main():
    print(f"Device: {DEVICE}  |  Batch size: {BATCH_SIZE}\n")

    # Load one language only (en_us) for timing
    lang_code, lang_name = "en_us", "english"
    dataset = load_dataset("rasgaard/fleurs_test", lang_code, split="train")
    n_samples = len(dataset)
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE
    n_languages = len(LANGUAGES)

    model, processor = load_model()

    # ------------------------------------------------------------------
    # 1. Proxy metrics — time N_PROXY_BATCHES batches, extrapolate
    # ------------------------------------------------------------------
    from compute_layer_metrics import compute_proxy_metrics
    import numpy as np

    audios = [s["audio"]["array"] for s in dataset]
    sr = dataset[0]["audio"]["sampling_rate"]

    # Warm up
    inputs = processor(audios[:BATCH_SIZE], sampling_rate=sr, return_tensors="pt", padding="max_length")
    inputs = inputs.to(device=DEVICE, dtype=model.dtype)
    with torch.no_grad():
        model.model.encoder(inputs.input_features)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for batch_start in range(0, N_PROXY_BATCHES * BATCH_SIZE, BATCH_SIZE):
        batch_audio = audios[batch_start : batch_start + BATCH_SIZE]
        inp = processor(batch_audio, sampling_rate=sr, return_tensors="pt", padding="max_length")
        inp = inp.to(device=DEVICE, dtype=model.dtype)
        layer_stats = {}

        def make_hook(i):
            def hook(module, inp_, out):
                h_in = inp_[0].detach().float()
                h_out = out[0].detach().float()
                layer_stats[i] = (
                    F.cosine_similarity(h_in, h_out, dim=-1).mean().item(),
                    F.mse_loss(h_in, h_out).item(),
                )
            return hook

        hooks = [model.model.encoder.layers[i].register_forward_hook(make_hook(i)) for i in range(NUM_LAYERS)]
        with torch.no_grad():
            model.model.encoder(inp.input_features)
        for h in hooks:
            h.remove()

    torch.cuda.synchronize()
    proxy_batch_time = (time.perf_counter() - t0) / N_PROXY_BATCHES
    proxy_total = proxy_batch_time * n_batches * n_languages
    print(f"=== Proxy metrics ===")
    print(f"  {proxy_batch_time:.2f}s per batch  →  {proxy_total/60:.1f} min total ({n_batches} batches × {n_languages} languages)")

    # ------------------------------------------------------------------
    # 2. ΔWER sweep — time deepcopy + transcription for N_PROBE_LAYERS
    # ------------------------------------------------------------------
    print(f"\n=== ΔWER sweep (timing {N_PROBE_LAYERS} layers) ===")
    layer_times = []
    for layer_idx in range(N_PROBE_LAYERS):
        t0 = time.perf_counter()
        pruned = prune_encoder_layers(model, [layer_idx])
        pruned = pruned.to(DEVICE).eval()
        hyps, refs = transcribe_dataset(pruned, processor, dataset, lang_name)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        layer_times.append(elapsed)
        del pruned
        torch.cuda.empty_cache()
        print(f"  Layer {layer_idx}: {elapsed:.1f}s (1 language, {n_samples} samples)")

    avg_layer_time = sum(layer_times) / len(layer_times)
    # Total: NUM_LAYERS layers × n_languages languages × avg time per (layer, language)
    # avg_layer_time already includes deepcopy (done once per layer) + 1 language
    # For n_languages: deepcopy once, then eval n_languages times
    # Approximate: (deepcopy_time + n_languages * eval_time) * NUM_LAYERS
    # Simpler: avg_layer_time / 1 language * n_languages * NUM_LAYERS
    delta_wer_total = avg_layer_time * n_languages * NUM_LAYERS
    print(f"\n  ~{avg_layer_time:.1f}s per (layer × language)  →  {delta_wer_total/60:.1f} min total")
    print(f"  (Note: deepcopy is amortised across {n_languages} languages in the actual script)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    grand_total = (proxy_total + delta_wer_total) / 60
    print(f"\n{'='*45}")
    print(f"  Proxy metrics:  {proxy_total/60:.1f} min")
    print(f"  ΔWER sweep:     {delta_wer_total/60:.1f} min")
    print(f"  Grand total:    {grand_total:.0f} min  (~{grand_total/60:.1f} h)")
    print(f"{'='*45}")

if __name__ == "__main__":
    main()
