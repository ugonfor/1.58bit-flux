#!/bin/bash
# V10b: Rank-128 LoRA — extended training (12000 steps) from V10 checkpoint
# V10 (6000 steps, cold-start) got 87.7% OOD CLIP — under-converged.
# Evidence: high per-prompt variance (p14=98% but p15=42%). More steps needed.
# Warm-start from V10 checkpoint to continue convergence.
set -e
PYTHON=/home/jovyan/conda/dit-bits-vs-quality/bin/python
WD=/home/ugonfor/DiT-bits-vs-quality/DiT-bits-vs-quality
cd "$WD"

DATASET=output/teacher_dataset_v9b_combined.pt
RANK=128
V10_CKPT=output/ternary_distilled_r128_res1024_s6000_fm_lpips1e-01.pt

echo "=== V10b: Rank-128 LoRA — extended training ==="
echo "  Dataset: $DATASET (2132 unique prompts)"
echo "  Rank: $RANK"
echo "  Init: Warm-start from V10 ($V10_CKPT)"
echo "  Steps: 12000 (2x V10)"

echo ""
echo "=== Step 1: Train V10b (12000 steps offline FM, warm-start from V10) ==="
# 12000 steps × ~3.3s = ~11h
PYTHONUNBUFFERED=1 $PYTHON train_ternary.py \
    --steps 12000 \
    --rank $RANK \
    --res 1024 \
    --lr-lora 1e-4 \
    --lr-scale 3e-4 \
    --grad-checkpointing \
    --grad-accum 4 \
    --loss-type fm \
    --lpips-weight 0.1 \
    --t-dist logit-normal \
    --dataset "$DATASET" \
    --init-ckpt "$V10_CKPT" \
    2>&1 | tee output/train_v10b.log

echo ""
echo "=== Step 2: CLIP eval on 4 fixed prompts ==="
V10B_CKPT=output/ternary_distilled_r${RANK}_res1024_s12000_fm_lpips1e-01.pt
echo "  Using checkpoint: $V10B_CKPT"
PYTHONUNBUFFERED=1 $PYTHON eval_ternary_clip.py \
    --ckpt "$V10B_CKPT" \
    --rank $RANK \
    --save-dir output/eval_fm_clip_v10b \
    2>&1 | tee output/eval_v10b_clip.log

echo ""
echo "=== Step 3: Diverse prompt eval (20 unseen prompts) ==="
V9B_CKPT=output/ternary_distilled_r64_res1024_s6000_fm_lpips1e-01.pt
HF_HOME=/home/jovyan/.cache/huggingface \
PYTHONUNBUFFERED=1 $PYTHON eval_diverse.py \
    --models "V9b:${V9B_CKPT}:64" "V10b:${V10B_CKPT}:${RANK}" \
    --rank $RANK \
    2>&1 | tee output/eval_v10b_diverse.log

echo ""
echo "=== V10b pipeline complete ==="
