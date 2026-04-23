#!/bin/bash
#BSUB -q gpul40s
#BSUB -J distill_option_a
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/distill_option_a_%J.out
#BSUB -e logs/distill_option_a_%J.err

cd /work3/roraa/home/repos/whisper-encoder-layer-prune
mkdir -p logs

# Option A: next 6 least important layers by mean ΔWER (ranks 7-12)
uv run python distill_pruned_encoder.py --layers 2 4 8 13 14 17
