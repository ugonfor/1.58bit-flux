#!/bin/bash
# V5 pipeline: merge dataset → train → eval
# Run after teacher_dataset_350.pt is generated.
set -e

cd /home/ugonfor/DiT-bits-vs-quality/DiT-bits-vs-quality
PYTHON=/home/jovyan/conda/dit-bits-vs-quality/bin/python
export HF_HOME=/home/jovyan/.cache/huggingface
export PYTHONUNBUFFERED=1

echo "=== V5 Pipeline ==="
echo "Step 1: Merge datasets (V2 200-img + new 348-img = 548 combined)"
$PYTHON merge_datasets.py \
  --datasets output/teacher_dataset_200.pt output/teacher_dataset_350.pt \
  --out output/teacher_dataset_combined.pt

echo ""
echo "Step 2: Backup V2 checkpoint and eval directory"
cp output/ternary_distilled_r64_res1024_s3000_fm.pt \
   output/ternary_distilled_r64_res1024_s3000_fm_v2_backup.pt
echo "  Backed up V2 checkpoint → ...fm_v2_backup.pt"

# Rename V2 mid-training eval dir so V5 doesn't overwrite it
if [ -d output/eval_ternary_r64_res1024_s3000_fm ]; then
  mv output/eval_ternary_r64_res1024_s3000_fm \
     output/eval_ternary_r64_res1024_s3000_fm_v2_backup
  echo "  Renamed V2 eval dir → ...fm_v2_backup/"
fi

echo ""
echo "Step 3: V5 offline FM training (548 images, 174 prompts, warm-start from V2)"
$PYTHON train_ternary.py \
  --steps 3000 --rank 64 --res 1024 \
  --lr-lora 1e-4 --lr-scale 3e-4 \
  --grad-checkpointing --grad-accum 4 \
  --loss-type fm \
  --dataset output/teacher_dataset_combined.pt \
  --init-ckpt output/ternary_distilled_r64_res1024_s3000_fm.pt \
  2>&1 | tee output/train_r64_res1024_s3000_fm_v5.log

echo ""
echo "Step 4: CLIP evaluation"
$PYTHON eval_ternary_clip.py \
  --ckpt output/ternary_distilled_r64_res1024_s3000_fm.pt \
  --rank 64 --steps 30 --seed 42 \
  --save-dir output/eval_fm_clip_v5 \
  2>&1 | tee output/eval_fm_clip_v5.log

echo ""
echo "=== V5 Pipeline Complete ==="
