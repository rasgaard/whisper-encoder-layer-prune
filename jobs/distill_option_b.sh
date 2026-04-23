#!/bin/bash
#BSUB -q gpua10
#BSUB -J distill_option_b
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -o logs/distill_option_b_%J.out
#BSUB -e logs/distill_option_b_%J.err

cd /work3/roraa/home/repos/whisper-encoder-layer-prune
mkdir -p logs

# Option B: contiguous early block (layers 2-7)
uv run python distill_pruned_encoder.py --layers 2 3 4 5 6 7
