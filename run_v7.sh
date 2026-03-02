#!/bin/bash
# V7 pipeline: merge datasets → train V7 → evaluate
# Run after generate_teacher_dataset.py --n-images 826 ... finishes
set -e
PYTHON=/home/jovyan/conda/dit-bits-vs-quality/bin/python
WD=/home/ugonfor/DiT-bits-vs-quality/DiT-bits-vs-quality
cd "$WD"

echo "=== Step 1: Merge datasets ==="
PYTHONUNBUFFERED=1 $PYTHON merge_datasets.py \
    --datasets output/teacher_dataset_combined.pt output/teacher_dataset_new826.pt \
    --out output/teacher_dataset_v7.pt

echo ""
echo "=== Step 2: Backup V6 checkpoint ==="
cp output/ternary_distilled_r64_res1024_s2000_fm_lpips1e-01.pt \
   output/ternary_distilled_r64_res1024_s2000_fm_lpips1e-01_v6backup.pt
echo "  Backed up V6 → v6backup.pt"

echo ""
echo "=== Step 3: Train V7 (warm-start from V6, 3000 steps) ==="
PYTHONUNBUFFERED=1 $PYTHON train_ternary.py \
    --steps 3000 \
    --rank 64 \
    --res 1024 \
    --lr-lora 1e-4 \
    --lr-scale 3e-4 \
    --grad-checkpointing \
    --grad-accum 4 \
    --loss-type fm \
    --lpips-weight 0.1 \
    --dataset output/teacher_dataset_v7.pt \
    --init-ckpt output/ternary_distilled_r64_res1024_s2000_fm_lpips1e-01.pt \
    2>&1 | tee output/train_v7.log

echo ""
echo "=== Step 4: CLIP eval on 4 fixed prompts ==="
V7_CKPT=$(ls -t output/ternary_distilled_r64_res1024_s3000_fm_lpips*.pt 2>/dev/null | head -1)
echo "  Using checkpoint: $V7_CKPT"
PYTHONUNBUFFERED=1 $PYTHON eval_ternary_clip.py \
    --ckpt "$V7_CKPT" \
    --rank 64 \
    --save-dir output/eval_fm_clip_v7 \
    2>&1 | tee output/eval_v7_clip.log

echo ""
echo "=== Step 5: Quality eval (LPIPS + aesthetic) ==="
PYTHONUNBUFFERED=1 $PYTHON eval_quality.py \
    --models BF16:output/bf16_reference V6:output/eval_fm_clip_v6 V7:output/eval_fm_clip_v7 \
    2>&1 | tee output/eval_v7_quality.log

echo ""
echo "=== Step 6: Diverse prompt eval (20 unseen prompts) ==="
PYTHONUNBUFFERED=1 $PYTHON eval_diverse.py \
    --ckpt "$V7_CKPT" \
    --rank 64 \
    2>&1 | tee output/eval_v7_diverse.log

echo ""
echo "=== V7 pipeline complete ==="
