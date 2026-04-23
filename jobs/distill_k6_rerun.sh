#!/bin/bash
#BSUB -q p1
#BSUB -J distill_k6_rerun
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -o logs/distill_k6_rerun_%J.out
#BSUB -e logs/distill_k6_rerun_%J.err

cd /work3/roraa/home/repos/whisper-encoder-layer-prune
mkdir -p logs

# k=6 rerun: optimal pruning set, fresh log (seed=1 to avoid cached zero-shot)
uv run python distill_pruned_encoder.py --layers 5 6 7 9 10 11 --steps 2000 --seed 1
