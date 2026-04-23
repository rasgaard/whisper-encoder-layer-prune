#!/usr/bin/env python3
"""Benchmark memory and runtime: full model vs pruned+distilled model."""

import copy
import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import load_dataset

DEVICE = "cuda:1"
MODEL_ID = "openai/whisper-large-v3-turbo"
PRUNED_DIR = "results/distilled_pruned_model"
LAYERS_TO_REMOVE = [5, 6, 7, 9, 10, 11]
N_WARMUP = 5
N_RUNS = 20

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def model_size_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

def time_encoder(model, inputs, n_runs):
    torch.cuda.synchronize(DEVICE)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model.model.encoder(inputs.input_features)
    torch.cuda.synchronize(DEVICE)
    return (time.perf_counter() - t0) / n_runs

def time_transcribe(model, processor, inputs, n_runs):
    torch.cuda.synchronize(DEVICE)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            ids = model.generate(inputs.input_features, language="english", task="transcribe")
    torch.cuda.synchronize(DEVICE)
    return (time.perf_counter() - t0) / n_runs

print("Loading models...")
processor  = AutoProcessor.from_pretrained(MODEL_ID)
full_model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, dtype=torch.bfloat16).eval()

# Reconstruct pruned architecture on CPU, load distilled weights, then move to GPU
pruned_model = copy.deepcopy(full_model)
for idx in sorted(LAYERS_TO_REMOVE, reverse=True):
    del pruned_model.model.encoder.layers[idx]
pruned_state = torch.load(f"{PRUNED_DIR}/model.safetensors", weights_only=True) if False else None

# Use from_pretrained with the pruned architecture by loading state dict directly
import safetensors.torch
pruned_state = safetensors.torch.load_file(f"{PRUNED_DIR}/model.safetensors")
pruned_model.load_state_dict(pruned_state, strict=False)

full_model   = full_model.to(DEVICE).eval()
pruned_model = pruned_model.to(DEVICE).eval()

print("\n--- Parameter counts ---")
full_total   = count_params(full_model)
pruned_total = count_params(pruned_model)
full_enc     = count_params(full_model.model.encoder)
pruned_enc   = count_params(pruned_model.model.encoder)
print(f"Full model:   {full_total/1e6:.1f}M params  ({model_size_mb(full_model):.0f} MB in bfloat16)")
print(f"Pruned model: {pruned_total/1e6:.1f}M params  ({model_size_mb(pruned_model):.0f} MB in bfloat16)")
print(f"Reduction:    {(1 - pruned_total/full_total)*100:.1f}% fewer params  ({(full_total-pruned_total)/1e6:.1f}M removed)")
print(f"Encoder only: {full_enc/1e6:.1f}M → {pruned_enc/1e6:.1f}M  ({(1-pruned_enc/full_enc)*100:.1f}% reduction)")

print("\nLoading sample audio...")
dataset = load_dataset("rasgaard/fleurs_test", "en_us", split="train")
audios = [s["audio"]["array"] for s in dataset.select(range(8))]
sr = dataset[0]["audio"]["sampling_rate"]
inputs = processor(audios, sampling_rate=sr, return_tensors="pt", padding="max_length").to(device=DEVICE, dtype=torch.bfloat16)

print(f"\n--- Encoder forward pass ({N_RUNS} runs, batch=8) ---")
for _ in range(N_WARMUP):
    with torch.no_grad():
        full_model.model.encoder(inputs.input_features)
        pruned_model.model.encoder(inputs.input_features)

full_enc_time   = time_encoder(full_model, inputs, N_RUNS)
pruned_enc_time = time_encoder(pruned_model, inputs, N_RUNS)
print(f"Full model:   {full_enc_time*1000:.1f} ms/batch")
print(f"Pruned model: {pruned_enc_time*1000:.1f} ms/batch")
print(f"Speedup:      {full_enc_time/pruned_enc_time:.2f}x  ({(1-pruned_enc_time/full_enc_time)*100:.1f}% faster)")

print(f"\n--- Full transcription ({N_RUNS} runs, batch=8) ---")
full_trans_time   = time_transcribe(full_model, processor, inputs, N_RUNS)
pruned_trans_time = time_transcribe(pruned_model, processor, inputs, N_RUNS)
print(f"Full model:   {full_trans_time*1000:.1f} ms/batch  ({full_trans_time/8*1000:.1f} ms/sample)")
print(f"Pruned model: {pruned_trans_time*1000:.1f} ms/batch  ({pruned_trans_time/8*1000:.1f} ms/sample)")
print(f"Speedup:      {full_trans_time/pruned_trans_time:.2f}x  ({(1-pruned_trans_time/full_trans_time)*100:.1f}% faster)")

print("\nDone.")
