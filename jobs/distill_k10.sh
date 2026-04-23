#!/bin/bash
#BSUB -q p1
#BSUB -J distill_k10
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 6:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -o logs/distill_k10_%J.out
#BSUB -e logs/distill_k10_%J.err

cd /work3/roraa/home/repos/whisper-encoder-layer-prune
mkdir -p logs

# k=10: 10 least important layers by mean ΔWER
uv run python distill_pruned_encoder.py --layers 2 4 5 6 7 8 9 10 11 13 --steps 4000 --eval-every 1000
