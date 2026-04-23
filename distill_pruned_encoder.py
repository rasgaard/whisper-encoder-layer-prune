#!/usr/bin/env python3
"""
Label-free representation distillation for a pruned Whisper encoder.

Removing encoder layers shifts the encoder output distribution away from what
the (frozen) decoder expects. This script re-aligns the pruned encoder by
minimising MSE between its final hidden states and those of the full model,
using People's Speech (English, unlabelled) as the distillation corpus.

No transcriptions are used. The full model is the teacher; only the pruned
encoder's parameters are updated.

Usage:
    uv run python distill_pruned_encoder.py
    uv run python distill_pruned_encoder.py --steps 2000 --lr 1e-5 --batch-size 8
"""

import argparse
import copy
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from jiwer import wer as compute_wer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

try:
    from peft import LoraConfig, get_peft_model
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID   = "openai/whisper-large-v3-turbo"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

LAYERS_TO_REMOVE = [5, 6, 7, 9, 10, 11]  # default: 6 least important by mean ΔWER

LANGUAGES = {
    "da_dk": "danish",
    "en_us": "english",
    "it_it": "italian",
    "de_de": "german",
    "sv_se": "swedish",
}

EVAL_BATCH_SIZE = 16  # for WER evaluation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prune_encoder_layers(model, layers_to_remove: list[int]):
    pruned = copy.deepcopy(model)
    layers = pruned.model.encoder.layers
    for idx in sorted(set(layers_to_remove), reverse=True):
        del layers[idx]
    return pruned


def transcribe_dataset(model, processor, dataset, lang_name: str):
    audios     = [s["audio"]["array"] for s in dataset]
    sr         = dataset[0]["audio"]["sampling_rate"]
    references = [s["transcription"] for s in dataset]
    hypotheses = []

    for start in range(0, len(audios), EVAL_BATCH_SIZE):
        batch = audios[start : start + EVAL_BATCH_SIZE]
        inputs = processor(
            batch, sampling_rate=sr, return_tensors="pt", padding="max_length",
        ).to(device=DEVICE, dtype=torch.bfloat16)
        with torch.no_grad():
            ids = model.generate(inputs.input_features, language=lang_name, task="transcribe")
        hypotheses.extend(h.lower().strip() for h in processor.batch_decode(ids, skip_special_tokens=True))

    return hypotheses, references


def evaluate_wer(model, processor, fleurs_datasets: dict, baseline_wers: dict, tag: str) -> dict:
    print(f"\n--- WER evaluation: {tag} ---")
    results = {}
    for lang_code, lang_name in LANGUAGES.items():
        hyps, refs = transcribe_dataset(model, processor, fleurs_datasets[lang_code], lang_name)
        wer        = compute_wer(refs, hyps)
        rel_delta  = (wer - baseline_wers[lang_code]) / baseline_wers[lang_code]
        results[lang_code] = {"wer": wer, "rel_delta": rel_delta}
        flag = "  *** EXCEEDS 5%" if rel_delta > 0.05 else ""
        print(f"  [{lang_code}]  WER={wer:.4f}  rel_Δ={rel_delta:+.3f}{flag}")
    mean_rel = sum(r["rel_delta"] for r in results.values()) / len(results)
    print(f"  mean rel_Δ = {mean_rel:+.3f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps",      type=int,   default=2000,  help="Distillation steps (default: 2000)")
    p.add_argument("--lr",         type=float, default=1e-5,  help="Learning rate (default: 1e-5)")
    p.add_argument("--batch-size", type=int,   default=8,     help="Distillation batch size (default: 8)")
    p.add_argument("--eval-every", type=int,   default=500,   help="Evaluate on FLEURS every N steps (default: 500)")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     type=str,   default=None,  help="Device (e.g. cuda:1). Defaults to cuda if available.")
    p.add_argument("--layers",     type=int,   nargs="+",     default=None,
                   help="Encoder layer indices to remove (default: 5 6 7 9 10 11)")
    p.add_argument("--lora",       type=int,   default=None,  metavar="RANK",
                   help="Use LoRA with given rank instead of full fine-tuning (e.g. --lora 16)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    global DEVICE
    if args.device:
        DEVICE = args.device

    layers_to_remove = sorted(args.layers) if args.layers is not None else LAYERS_TO_REMOVE
    tag = "_".join(str(l) for l in layers_to_remove)
    if args.lora:
        tag += f"_lora{args.lora}"
    log_path    = RESULTS_DIR / f"distillation_log_{tag}.json"
    output_dir  = RESULTS_DIR / f"distilled_pruned_model_{tag}"

    print(f"Device: {DEVICE}")
    print(f"Layers to remove: {layers_to_remove}")
    print(f"LoRA rank: {args.lora if args.lora else 'disabled (full fine-tuning)'}")
    print(f"Steps: {args.steps}  LR: {args.lr}  Batch: {args.batch_size}\n")

    # --- load models ---
    print("Loading processor and models...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    teacher = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Prune on CPU before moving to GPU to avoid needing two full model copies in VRAM
    student = prune_encoder_layers(teacher, layers_to_remove)

    teacher = teacher.to(DEVICE).eval()
    student = student.to(DEVICE)

    if args.lora:
        if not _PEFT_AVAILABLE:
            raise ImportError("peft is required for --lora. Run: uv add peft")
        for p in student.parameters():
            p.requires_grad_(False)
        lora_cfg = LoraConfig(
            r=args.lora,
            lora_alpha=args.lora * 2,
            target_modules=["q_proj", "v_proj"],
            layers_to_transform=list(range(len(student.model.encoder.layers))),
        )
        student = get_peft_model(student, lora_cfg)
        encoder_fn = lambda x: student.base_model.model.model.encoder(x).last_hidden_state
        trainable_params = [p for p in student.parameters() if p.requires_grad]
    else:
        for p in student.model.encoder.parameters():
            p.requires_grad_(True)
        for p in student.model.decoder.parameters():
            p.requires_grad_(False)
        encoder_fn = lambda x: student.model.encoder(x).last_hidden_state
        trainable_params = list(student.model.encoder.parameters())

    student.train()
    n_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable encoder parameters: {n_params / 1e6:.2f}M")

    # --- load FLEURS for evaluation ---
    print("\nLoading FLEURS test sets...")
    fleurs = {
        lang: load_dataset("rasgaard/fleurs_test", lang, split="train")
        for lang in LANGUAGES
    }
    with open(RESULTS_DIR / "baseline_wers.json") as f:
        baseline_wers = json.load(f)

    # --- zero-shot baseline (skip if already saved) ---
    if log_path.exists():
        with open(log_path) as f:
            existing = json.load(f)
        if "zero_shot" in existing:
            print("\nLoading cached zero-shot baseline from distillation_log.json")
            zero_shot = existing["zero_shot"]
        else:
            student.eval()
            zero_shot = evaluate_wer(student, processor, fleurs, baseline_wers, "zero-shot pruned (before distillation)")
            student.train()
    else:
        student.eval()
        zero_shot = evaluate_wer(student, processor, fleurs, baseline_wers, "zero-shot pruned (before distillation)")
        student.train()

    # --- load distillation data ---
    print("\nLoading People's Speech (distillation corpus)...")
    distill_data = load_dataset("MLCommons/peoples_speech", "validation", split="validation")
    distill_data = distill_data.shuffle(seed=args.seed)
    audios = [s["audio"]["array"] for s in distill_data]
    sr     = distill_data[0]["audio"]["sampling_rate"]
    print(f"  {len(audios)} examples at {sr}Hz")

    # --- optimizer ---
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=0.01,
    )

    # --- distillation loop ---
    print(f"\nStarting distillation ({args.steps} steps)...")
    results_log = {"zero_shot": zero_shot, "steps": []}
    best_mean_rel = float("inf")
    best_state   = None

    step       = 0
    total_loss = 0.0
    data_idx   = 0

    while step < args.steps:
        # build batch — wrap around if needed
        batch_audio = []
        while len(batch_audio) < args.batch_size:
            batch_audio.append(audios[data_idx % len(audios)])
            data_idx += 1

        inputs = processor(
            batch_audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding="max_length",
        ).to(device=DEVICE, dtype=torch.bfloat16)

        with torch.no_grad():
            teacher_out = teacher.model.encoder(inputs.input_features).last_hidden_state

        student_out = encoder_fn(inputs.input_features)

        loss = F.mse_loss(student_out.float(), teacher_out.float())

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss ({loss.item()}) at step {step + 1}. "
                               "Check for activation overflow — try a lower learning rate or bfloat16.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        step       += 1

        if step % 50 == 0:
            print(f"  step {step:4d}/{args.steps}  loss={total_loss/50:.6f}", flush=True)
            total_loss = 0.0

        if step % args.eval_every == 0 or step == args.steps:
            student.eval()
            wer_results = evaluate_wer(student, processor, fleurs, baseline_wers, f"step {step}")
            student.train()

            mean_rel = sum(r["rel_delta"] for r in wer_results.values()) / len(wer_results)
            results_log["steps"].append({"step": step, "wer": wer_results})

            if mean_rel < best_mean_rel:
                best_mean_rel = mean_rel
                best_state    = copy.deepcopy(student.state_dict())
                print(f"  → new best (mean rel_Δ={mean_rel:+.3f}), saving checkpoint")

            with open(log_path, "w") as f:
                json.dump(results_log, f, indent=2)

    # --- save best model ---
    print("\nSaving best checkpoint...")
    student.load_state_dict(best_state)
    student.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved → {output_dir}")

    # --- final evaluation ---
    student.eval()
    final = evaluate_wer(student, processor, fleurs, baseline_wers, "final (best checkpoint)")
    results_log["final"] = final
    with open(log_path, "w") as f:
        json.dump(results_log, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
