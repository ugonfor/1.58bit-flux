# Post 003: Ternary FLUX — Flow-Matching Distillation (Correct Approach)

**Date**: 2026-02-28
**Status**: Results + analysis

---

## Background

[Post 002](./002-ternary-flux-distillation.md) introduced self-supervised distillation using
layer-wise MSE losses (rank-8 LoRA, 512px). While CLIP improved from 0.178 (naive PTQ) to 0.203,
a deep flaw remained: the training distribution did not match inference.

This post documents the correct flow-matching (FM) distillation approach, which eliminates
distribution shift and achieves dramatically better image quality.

---

## The Distribution Shift Problem

Every failed approach shared the same root cause:

**Training**: student sees `z_t = pure_noise` (sampled from N(0,I)) for all t values
**Inference**: model denoises along trajectory `z_t = (1-t)·z_0 + t·ε`

The model never encountered intermediate states `z_t` during training, so inference along
the proper trajectory was never learned. Result: both layer-wise MSE and output-velocity
loss at random noise produced pure noise during inference.

### Failed Approaches

| Approach | Loss at step 500 | Eval image |
|---|---|---|
| Layer-wise MSE (post-002) | ~450 (decreasing) | Pure noise |
| Output velocity at random noise | ~0.07 (converged) | Pure noise |
| FM distillation (this post) | ~0.23 (converging) | Real structure |

The "output-velocity at random noise" approach was particularly deceptive: loss went to
~0.07 within 500 steps because matching velocities at `z_t=pure_noise` is easy — but the
model learned nothing about the actual flow trajectory.

---

## Correct Approach: Flow-Matching Distillation

### Step 1: Pre-Generate Teacher Latents

```bash
python generate_teacher_dataset.py --n-images 50 --steps 28 --seed 42
# → output/teacher_dataset.pt (126 MB, 50 items)
```

For each prompt, the BF16 teacher generates a complete image (28 denoising steps, guidance 3.5)
and saves the packed latent `z_0 ∈ ℝ^{1×4096×64}` along with pre-encoded text embeddings.

**Key CUDA fix**: Add `del result; torch.cuda.empty_cache()` after each generation. Without
this, the CUDA caching allocator accumulates 3 GB/image and OOMs at image 22 on the A100 80GB.

### Step 2: Flow-Matching Training

For each training step:
1. Sample random item from teacher dataset, random `t ∈ U(0,1)`
2. Compute `z_t = (1-t)·z_0 + t·ε` where `ε ~ N(0,I)`
3. Compute teacher velocity: `v_teacher = teacher(z_t, t, c)` (frozen)
4. Compute student velocity: `v_student = student(z_t, t, c)` (trainable)
5. Loss: `MSE(v_student, v_teacher)`

This trains the student to match the teacher's velocity at *every point along the actual
flow trajectory*, not at random noise. At inference, the student naturally produces the
same trajectory as the teacher.

```bash
python train_ternary.py \
  --steps 3000 --rank 64 --res 1024 \
  --grad-checkpointing --grad-accum 4 \
  --loss-type fm \
  --dataset output/teacher_dataset.pt
```

**Memory**: 43.2 GB VRAM with gradient checkpointing (without: ~78 GB → OOM).
**Trainable**: 350M params (3M scale + 347M LoRA-64 × 504 layers).
**Speed**: ~2.56s/step → 128 min for 3000 steps.

---

## Results

### Loss Curve

| Steps | Avg Loss | Min Loss | Max Loss |
|---|---|---|---|
| 10–1000 | 0.248 | 0.052 | 1.28 |
| 1010–1820 | 0.153 | 0.033 | 1.56 |

Average loss dropped 38% from first to second half of training. High max values reflect
large-t timesteps (near pure noise) which are intrinsically harder to denoise.

### Visual Quality Progression

| Step | Quality |
|---|---|
| 500 | Abstract, warm-toned — structure emerging but no semantic content |
| 1000 | Photorealistic scenes — text conditioning beginning to work |
| 1500 | Prompt-aligned, cinematic quality — cyberpunk atmosphere, landscape features |
| 2000 | High quality photorealism — near-BF16 aesthetic quality |
| 2500 | Converged — woman with glowing hands in cyberpunk plaza; mountain valley with river |
| 3000 | Final — CLIP 0.2783 average (-13.6% vs BF16) |

Step 1500 prompt alignment for "Cyberpunk samurai on a neon-lit rooftop at dusk":
dark armored figures in an orange-glowing post-apocalyptic cityscape — genuinely captures
the aesthetic even if the exact subject (samurai on rooftop) differs slightly.

Step 2000 prompt alignment for "A fantasy landscape with mountains and a river":
aerial mountain canyon with river, golden sunlight, lush vegetation — excellent match.

### CLIP Scores

| Model | CLIP Score | Notes |
|---|---|---|
| BF16 (reference) | 0.322 | Full-precision baseline |
| INT4 + LoRA-64 | 0.318 | Near-lossless |
| Ternary PTQ (no training) | 0.178 | Pure noise |
| Distilled (rank-8, 512px, 800 steps) | 0.203 | Post-002 result |
| **FM distilled (rank-64, 1024px, 3000 steps)** | **0.2783** | This post — only -13.6% vs BF16 |

### Final CLIP Scores (step 3000, 30 inference steps, seed=42)

| Prompt | CLIP | % of BF16 |
|---|---|---|
| Cyberpunk samurai on a neon-lit rooftop | 0.2929 | 90.8% |
| A fantasy landscape with mountains and a river | 0.3002 | 93.2% |
| Portrait of a young woman with wild curly hair | 0.2719 | 84.5% |
| Aerial view of a coastal city at sunset | 0.2483 | 77.0% |
| **Average** | **0.2783** | **86.4%** |

**Gap vs BF16: only -13.6%** — a 3× improvement over naive PTQ's -44.7% gap.

Progress vs previous approaches:
- Naive ternary PTQ: CLIP 0.178 (pure noise, gap -44.7%)
- Rank-8 layer-wise distillation (512px, 800 steps): CLIP 0.203 (gap -37.0%)
- **FM rank-64 distillation (1024px, 3000 steps): CLIP 0.2783 (gap -13.6%)**

The fantasy landscape and cyberpunk prompts reach 90–93% of BF16 quality. Portrait and
aerial scenes are weaker (77–84%), likely due to limited coverage in the 50-image teacher
dataset. Expanding the dataset to 200+ images with more portrait/aerial diversity would
likely close this gap further.

---

## Key Technical Insights

### Why FM Distillation Works

Flow-matching is a generative model framework where images are points `z_0` and noise is
`z_1 = ε`. The model learns the velocity field `v(z_t, t)` that transforms noise into images
via `dz/dt = v(z_t, t)`. During inference, numerical integration of this ODE from `z_1` (noise)
to `z_0` (image) recovers the clean image.

Training on the actual trajectory `z_t = (1-t)·z_0 + t·ε` means the student learns to
predict the same velocity as the teacher at every point the inference trajectory visits.

### Architecture Constraints

- 504 `nn.Linear` layers, all 100% linear (no conv/norm trainable params)
- `TernaryLinear` = frozen int8 {-1,0,+1} weights + trainable per-channel scale + LoRA
- `W_eff = weight_q * scale + lora_A @ lora_B`
- SVD init from ternary residual `E = W - W_q * scale` avoids catastrophic initial loss

### Memory Budget (A100 80GB)

| Component | Memory |
|---|---|
| Ternary student (rank-64) | 12,021 MB |
| BF16 teacher transformer | 22,700 MB |
| Other pipeline components | ~9,000 MB |
| **Total static** | ~43,700 MB |
| Peak (with grad checkpointing) | ~43,700 MB |
| Peak (without grad checkpointing) | ~78,300 MB → OOM |

---

## Reproducibility

```bash
# Step 1: Generate teacher dataset (12 min on A100)
HF_HOME=/home/jovyan/.cache/huggingface PYTHONUNBUFFERED=1 \
python generate_teacher_dataset.py --n-images 50 --steps 28 --seed 42
# → output/teacher_dataset.pt

# Step 2: FM distillation (128 min on A100)
HF_HOME=/home/jovyan/.cache/huggingface PYTHONUNBUFFERED=1 \
python train_ternary.py \
  --steps 3000 --rank 64 --res 1024 \
  --grad-checkpointing --grad-accum 4 \
  --loss-type fm \
  --dataset output/teacher_dataset.pt
# → output/ternary_distilled_r64_res1024_s3000_fm.pt
# → output/eval_ternary_r64_res1024_s3000_fm/step{0500,1000,1500,2000,2500,3000}_p*.png

# Step 3: CLIP evaluation
python eval_ternary_clip.py \
  --ckpt output/ternary_distilled_r64_res1024_s3000_fm.pt \
  --rank 64 --steps 30
```

GPU: NVIDIA A100-SXM4-80GB
Python env: diffusers==0.34.0, torch==2.5.0+cu124
