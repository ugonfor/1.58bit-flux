#!/bin/bash
# Wait for dataset generation to finish, then auto-launch V7 pipeline
set -e
WD=/home/ugonfor/DiT-bits-vs-quality/DiT-bits-vs-quality
DATASET_OUT=$WD/output/teacher_dataset_new826.pt
LOG=$WD/output/generate_new826.log

echo "[wait_and_run_v7] Waiting for $DATASET_OUT ..."
while true; do
    if [ -f "$DATASET_OUT" ]; then
        # Also verify the generation process finished (look for "Saved" in log)
        if grep -q "^Saved " "$LOG" 2>/dev/null; then
            echo "[wait_and_run_v7] Dataset found and generation complete. Launching V7 pipeline."
            break
        fi
    fi
    sleep 60
done

cd "$WD"
PYTHONUNBUFFERED=1 bash run_v7.sh 2>&1 | tee output/run_v7_full.log
echo "[wait_and_run_v7] V7 pipeline finished."
