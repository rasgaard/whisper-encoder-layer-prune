#!/bin/bash
#BSUB -q p1
#BSUB -J distill_pruned_encoder
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/distill_%J.out
#BSUB -e logs/distill_%J.err

cd /work3/roraa/home/repos/whisper-encoder-layer-prune
mkdir -p logs

uv run python distill_pruned_encoder.py
