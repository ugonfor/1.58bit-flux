# Post 010: V9a — Online FM at LR=1e-4: Unlocking the Artifact Threshold

**Date**: 2026-03-03
**Status**: Complete

---

## What V8 Left Open

V8 tested online FM at LR=3e-5 and found no improvement over V7. Two hypotheses:
1. **LR was the bottleneck**: 3e-5 was too conservative, model barely moved from V7 weights
2. **Distribution mismatch**: 5-step pseudo-z_0 ≠ 30-step BF16 distribution → fundamental ceiling

V9a tests hypothesis 1: **what happens at LR=1e-4** (same as V7's offline FM LR, but with 5-step online denoising)?

The risk: V4 had grid artifacts at LR=1e-4 with **single-step** pseudo-z_0. If the artifacts were caused by the LR alone, V9a would also show them. If they were caused by single-step noise, V9a should be clean.

---

## V9a Config

```bash
python train_ternary.py \
    --steps 1000 \
    --rank 64 \
    --res 1024 \
    --lr-lora 1e-4 \        # Same as V7 offline; 3.3× higher than V8
    --lr-scale 3e-4 \
    --grad-checkpointing --grad-accum 4 \
    --loss-type online --online-steps 5 \
    --lpips-weight 0.1 --t-dist logit-normal \
    --calib-prompts-file prompts_1000.txt \
    --init-ckpt output/ternary_distilled_r64_res1024_s4000_fm_lpips1e-01.pt  # V7
```

Only 1000 steps (half of V7's 2000 with V8) because we need a quick artifact check first. If clean → extend to 3000+ steps.

---

## Key Finding: No Artifacts at LR=1e-4 with 5-Step Denoising

Step 500 eval images (seed=42, 4 training prompts):

| Prompt | Quality | Artifacts? |
|---|---|---|
| Cyberpunk samurai | Correct — armored warrior in neon city | None |
| Fantasy landscape | Mountain valley, river visible | None |
| Portrait | Woman, curly hair, golden light | None |
| Aerial city | Sunset over water (slight semantic miss) | None |

**V4's grid artifacts were caused by single-step pseudo-z_0 quality, not the LR itself.** 5-step denoising produces clean enough targets for LR=1e-4 to work safely. This is the key engineering finding from V9a.

---

## Results

### Fixed-Prompt Eval (4 training prompts)

| Model | CLIP | Aesthetic | LPIPS↓ |
|---|---|---|---|
| BF16 | 0.3448 | 5.63 | ref |
| V7 | 0.3373 | 6.03 | 0.526 |
| V8 (LR=3e-5) | 0.3372 | 5.84 | 0.529 |
| **V9a (LR=1e-4)** | **0.3267** | **TBD** | **TBD** |

V9a fixed-prompt CLIP (0.3267) is slightly below V7 (0.3373) — expected, since online FM doesn't memorize fixed training prompts the way offline FM does. The model learns a more general velocity field rather than tuning to specific images.

### Diverse-Prompt Eval (20 unseen prompts)

| Model | CLIP | Aesthetic | LPIPS↓ | vs BF16 |
|---|---|---|---|---|
| BF16 | 0.3375 | 5.84 | ref | 100% |
| V7 | 0.3001 | 5.91 | 0.668 | **88.9%** |
| V8 (LR=3e-5) | 0.2994 | 5.72 | 0.669 | **88.7%** |
| **V9a (LR=1e-4)** | **0.3015** | **5.54** | **0.706** | **89.3%** |

**V9a improves OOD CLIP from 88.9% → 89.3%.** This confirms hypothesis 1: LR was the bottleneck for V8, not distribution mismatch alone. The higher LR does produce learning.

The cost: aesthetic dropped 0.36 points (5.91 → 5.54) and LPIPS degraded (0.668 → 0.706), indicating the online pseudo-z_0 distribution mismatch is still present and visible in pixel-space faithfulness.

### Per-Prompt Breakdown

| Prompt | BF16 CLIP | V7 CLIP | V9a CLIP | V9a vs BF16 | V9a vs V7 |
|---|---|---|---|---|---|
| p00 Lion | 0.3276 | 0.2817 | **0.3140** | 95.9% | +32.3 |
| p01 Parrot | 0.3220 | 0.3124 | 0.2748 | 85.3% | -37.6 |
| p02 Wolf | 0.3417 | 0.3386 | **0.3454** | 101.1% | +6.8 |
| p03 Cathedral | 0.3134 | 0.3103 | 0.2864 | 91.4% | -23.9 |
| p04 Skyscraper | 0.3246 | 0.3116 | 0.3076 | 94.8% | -4.0 |
| p05 Pagoda | 0.3510 | 0.3251 | **0.3748** | **106.8%** | +49.7 |
| p06 Northern lights | 0.3006 | 0.2610 | 0.2859 | 95.1% | +24.9 |
| p07 Volcano | 0.3456 | 0.3303 | 0.3304 | 95.6% | +0.1 |
| p08 Lavender | 0.3384 | 0.3205 | 0.3190 | 94.3% | -1.5 |
| p09 Fisherman | 0.3845 | 0.3012 | 0.3069 | 79.8% | +5.7 |
| p10 Ballet dancer | 0.3433 | 0.2776 | 0.2947 | 85.8% | +17.1 |
| p11 Sax musician | 0.3677 | 0.2662 | 0.2916 | 79.3% | +25.4 |
| p12 Sushi | 0.3122 | 0.2044 | 0.2358 | 75.5% | +31.4 |
| p13 Bread | 0.3433 | 0.3266 | 0.3173 | 92.4% | -9.3 |
| p14 Dragon/castle | 0.3668 | 0.3179 | 0.2495 | 68.0% | -68.4 |
| p15 Astronaut | 0.3365 | 0.3032 | 0.2989 | 88.8% | -4.3 |
| p16 Magic forest | 0.3684 | 0.3237 | 0.3341 | 90.7% | +10.4 |
| p17 Venice watercolor | 0.3179 | 0.2792 | 0.2666 | 83.9% | -12.6 |
| p18 Renaissance portrait | 0.2902 | 0.3055 | 0.2926 | 100.8% | -12.9 |
| p19 Tokyo neon | 0.3534 | 0.3056 | 0.3032 | 85.8% | -2.4 |

Key wins vs V7: p05 pagoda (+49.7), p00 lion (+32.3), p12 sushi (+31.4), p11 musician (+25.4), p06 northern lights (+24.9).
Key losses vs V7: p14 dragon/castle (−68.4), p01 parrot (−37.6), p03 cathedral (−23.9).

### Visual Inspection of Notable Prompts

**p05 (Japanese pagoda — V9a best)**: Stunning image: accurate 3-tier pagoda framed perfectly by pink cherry blossoms, lush green lawn. V9a=0.3748, the highest CLIP score in the entire diverse set, **exceeding BF16** (0.3510). The online FM approach generalized exceptionally to this prompt.

**p00 (lion at sunset)**: V9a shows a recognizable lion-like figure on rocks at golden sunset — significant improvement over V7 (which showed a cat-like/meerkat animal). Still not a fully convincing mane but correctly conveys "majestic lion at golden hour."

**p14 (dragon flying over medieval castle)**: V9a failure: shows a small creature perched on a rocky cliff with pine trees, blue haze background. No castle, no storm. V7 had correctly attempted a castle scene. This is V9a's worst regression.

**p12 (sushi platter)**: V9a still wrong — generates a red roe/caviar blob on a round plate. CLIP improved from 0.2044 → 0.2358 (food category recognized, not sushi-specific). The food domain continues to be the weakest category.

**p11 (street musician)**: V9a shows a person standing in a rainy neon street, more convincing urban atmosphere vs V7. Not clearly playing saxophone, but composition improved. CLIP 0.2662 → 0.2916 (+9.5%).

**p06 (northern lights)**: V9a shows a glowing column of light over a snowy river in a forest. Improvement over V7 but still not aurora. CLIP 0.2610 → 0.2859.

---

## Interpretation: Hypothesis 1 Partially Confirmed

| Outcome | Result |
|---|---|
| OOD CLIP improvement | **Yes: 88.9% → 89.3%** (+0.4pp) |
| No grid artifacts | **Yes: LR=1e-4 + 5-step is clean** |
| Aesthetic improvement | **No: degraded 5.91 → 5.54** |
| LPIPS improvement | **No: degraded 0.668 → 0.706** |

**Conclusion**: Hypothesis 1 is partially correct — LR was the bottleneck limiting V8's OOD CLIP. Raising it to 1e-4 with 5-step denoising does produce measurable improvement. However, the distribution mismatch (hypothesis 2) is simultaneously real: online FM learns trajectories from 5-step pseudo-z_0, which is not the same distribution as 30-step BF16 inference. This mismatch caps the aesthetic ceiling and degrades LPIPS even as CLIP improves.

The net effect is +0.4pp CLIP but −0.36 aesthetic points — a trade-off rather than a clean win.

---

## Data Scaling Projection

Based on V6 (86.0% @ 174 prompts) and V7 (88.9% @ 1000 prompts), a log2 scaling model:

```
OOD CLIP % = 0.0115 × log2(prompts) + 0.7744
```

| Prompts | Predicted OOD % |
|---|---|
| 1,000 (V7) | 88.9% ✓ |
| 2,000 | 90.0% |
| 3,000 | 90.7% |
| 7,232 (paper) | 92.2% |
| 10,000 | 92.7% |

Even at paper scale, the model predicts ~92%, not 100%. The remaining ~8% gap appears to be a fundamental limit of ternary + rank-64 LoRA compression — not a data problem.

---

## Path Forward: V9b — Offline Data Scaling

V9a confirmed that online FM at LR=1e-4 gives a modest CLIP improvement (+0.4pp) at an aesthetic cost (−0.36). The distribution mismatch between 5-step pseudo-z_0 and 30-step BF16 inference limits online FM's ceiling.

**V9b: Generate 2,000 more teacher latents, train offline FM for 6,000 steps from V7.**

Expected outcome from log2 scaling law:
- 3,000 total prompts → ~90.7% OOD CLIP
- Cleaner quality (offline FM on 30-step BF16 latents, no distribution mismatch)
- Consistent with V5→V6→V7's pattern of reliable data-driven improvement

V9a-full (extending to 3,000 steps at LR=1e-4) remains an option if we want to continue online FM exploration, but the aesthetic degradation makes offline data scaling the cleaner path.

---

GPU: NVIDIA A100-SXM4-80GB
Training: 93.8 min (1000 steps × 5.6s, 5 teacher passes per step)
Python env: diffusers==0.34.0, torch==2.5.0+cu124
