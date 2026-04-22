#!/bin/sh
### General options
### -- specify queue --
#BSUB -q p1
### -- set the job Name --
#BSUB -J whisper_layer_metrics
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --
#BSUB -W 4:00
### -- request memory --
#BSUB -R "rusage[mem=40GB]"
### -- set the email address --
#BSUB -u Rasmus.Aagaard@laerdal.com
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o logs/gpu_%J.out
#BSUB -e logs/gpu_%J.err
# -- end of LSF options --

echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Job name: $LSB_JOBNAME"
echo "Started at: $(date)"
echo "Running on: $(hostname)"
echo "=========================================="
echo ""

echo "GPU Information:"
nvidia-smi
echo ""

cd /work3/roraa/home/repos/whisper-encoder-layer-prune/

echo "Git commit:"
git log -1 --oneline
echo ""

echo "Starting at $(date)"
START_TIME=$(date +%s)

uv run python compute_layer_metrics.py
EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_HOURS=$((ELAPSED / 3600))
ELAPSED_MINS=$(((ELAPSED % 3600) / 60))

echo ""
echo "=========================================="
echo "Completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Total runtime: ${ELAPSED_HOURS}h ${ELAPSED_MINS}m"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Results saved to:"
    ls -lh results/
fi

exit $EXIT_CODE
