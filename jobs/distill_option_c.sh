#!/bin/bash
#BSUB -q gpua10
#BSUB -J distill_option_c
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -o logs/distill_option_c_%J.out
#BSUB -e logs/distill_option_c_%J.err

cd /work3/roraa/home/repos/whisper-encoder-layer-prune
mkdir -p logs

# Option C: proxy-guided selection — 6 lowest outlier_frac layers (excluding layer 0)
uv run python distill_pruned_encoder.py --layers 1 2 3 4 5 6
