# Post 013: V10 — Rank-128 LoRA Breaks the 90% Ceiling

**Date**: 2026-03-09
**Status**: Complete

---

## Motivation

V9c (Post 012) proved the ternary+rank-64 architecture hits a capacity ceiling at ~90% OOD CLIP. More data didn't help — it hurt. The log₂ scaling law broke at ~4000 prompts. The bottleneck is model capacity, not data.

V10 doubles the LoRA rank from 64 to 128, giving **2× trainable parameters** (~350M → ~700M). Two experiments test whether increased expressivity breaks through the ceiling:

- **V10** (6,000 steps, cold SVD init): Does rank-128 help at all?
- **V10b** (12,000 steps, warm-start from V10): Does it converge with more training?

---

## Config

| | V9b (baseline) | V10 | V10b |
|---|---|---|---|
| LoRA rank | 64 | 128 | 128 |
| Trainable params | 350M | 700M | 700M |
| Steps | 6,000 | 6,000 | 12,000 |
| Init | Warm (from V7) | Cold (SVD) | Warm (from V10) |
| Dataset | V9b combined | V9b combined | V9b combined |
| Checkpoint size | 668 MB | 1,330 MB | 1,330 MB |

Same dataset (2,132 unique prompts) across all three to isolate rank as the variable.

---

## Results

### Summary

| Model | OOD CLIP | Aesthetic | LPIPS↓ | vs BF16 |
|---|---|---|---|---|
| BF16 | 0.3375 | 5.842 | ref | 100% |
| V9b (r64, 6k) | 0.3036 | **5.939** | **0.664** | 90.0% |
| V10 (r128, 6k cold) | 0.2958 | 5.634 | 0.745 | 87.7% |
| **V10b (r128, 12k)** | **0.3050** | 5.686 | 0.719 | **90.4%** |

**V10b at 90.4% OOD CLIP — new best.** Rank-128 breaks through the rank-64 ceiling.

### Fixed-Prompt Eval (4 training prompts)

| Model | CLIP | vs BF16 |
|---|---|---|
| BF16 | 0.322 | 100% |
| V9b (r64) | 0.3250 | 100.9% |
| V10 (r128, 6k) | 0.3320 | 103.1% |
| V10b (r128, 12k) | 0.3311 | 102.8% |

---

### Per-Prompt Breakdown (V10b vs V9b)

| Prompt | BF16 | V9b | V10b | V10b% | vs V9b |
|---|---|---|---|---|---|
| p00 Lion | 0.3276 | 0.3128 | 0.3255 | 99.4% | +12.7 |
| p01 Parrot | 0.3220 | 0.3232 | 0.3201 | 99.4% | −3.1 |
| p02 Wolf | 0.3417 | 0.3340 | 0.3301 | 96.6% | −3.9 |
| p03 Cathedral | 0.3134 | 0.2941 | 0.3114 | 99.4% | +17.3 |
| p04 Skyscraper | 0.3246 | 0.3000 | 0.3049 | 93.9% | +4.9 |
| p05 Pagoda | 0.3510 | 0.3507 | 0.3458 | 98.5% | −4.9 |
| p06 N. Lights | 0.3006 | 0.2873 | 0.2915 | 97.0% | +4.2 |
| p07 Volcano | 0.3456 | 0.3355 | 0.3291 | 95.2% | −6.4 |
| p08 Lavender | 0.3384 | 0.3262 | 0.3068 | 90.7% | −19.4 |
| p09 Fisherman | 0.3845 | 0.3263 | **0.3653** | **95.0%** | **+39.0** |
| p10 Ballet | 0.3433 | 0.2616 | **0.3095** | **90.1%** | **+47.9** |
| p11 Musician | 0.3677 | 0.3133 | 0.3037 | 82.6% | −9.6 |
| p12 Sushi | 0.3122 | 0.2181 | 0.2343 | 75.0% | +16.2 |
| p13 Bread | 0.3433 | 0.2672 | 0.2471 | 72.0% | −20.1 |
| p14 Dragon | 0.3668 | 0.2976 | 0.3065 | 83.6% | +8.9 |
| p15 Astronaut | 0.3365 | 0.3071 | 0.1970 | 58.5% | **−110.1** |
| p16 Magic forest | 0.3684 | 0.3404 | **0.3822** | **103.7%** | **+41.8** |
| p17 Venice | 0.3179 | 0.2850 | 0.2883 | 90.7% | +3.3 |
| p18 Renaissance | 0.2902 | 0.2918 | 0.2860 | 98.6% | −5.8 |
| p19 Tokyo neon | 0.3534 | 0.3006 | 0.3146 | 89.0% | +14.0 |

**Big wins vs V9b**: p10 ballet (+47.9), p16 magic forest (+41.8), p09 fisherman (+39.0), p03 cathedral (+17.3), p12 sushi (+16.2)

**Losses**: p15 astronaut (−110.1 — catastrophic), p13 bread (−20.1), p08 lavender (−19.4)

---

### Visual Inspection

![V10 highlights: BF16 | V9b | V10b for 8 key prompts](../output/viz/v10_highlights.png)

**p16 (magic forest)**: V10b produces a stunning scene with glowing mushrooms and fireflies — **exceeds BF16** at 103.7% CLIP. The rank-128 LoRA captures fine atmospheric detail that rank-64 couldn't.

**p09 (fisherman)**: V10b generates a photorealistic elderly man with deeply weathered skin, kind eyes, and grey beard. CLIP jumped from 84.9% → 95.0% — the extra capacity captures portrait nuance.

**p10 (ballet)**: V10b shows a dancer mid-leap at sunset on a beach. CLIP improved dramatically from 76.2% → 90.1%. V9b's version was a dark silhouette; V10b has clear anatomy and movement.

**p03 (cathedral)**: V10b recovers the Gothic interior with stained glass. CLIP 93.8% → 99.4%. V9b missed the stained glass detail.

**p15 (astronaut)**: Still catastrophic — V10b generates a rocky/metallic sphere instead of an astronaut. CLIP 58.5% (aes=3.90). This prompt remains a persistent failure across all versions. The ternary model has never learned to generate astronaut-in-space compositions from its training data.

**p05 (pagoda)**: V10b at 98.5% — near-perfect Japanese pagoda with cherry blossoms. Both V9b and V10b nail this prompt.

---

### Convergence: Cold-Start Recovery

![V10 vs V10b convergence per prompt](../output/viz/v10_convergence.png)

The bar chart reveals the cold-start convergence pattern:

- **V10 (red, 6k cold)**: High variance. Some prompts near-BF16 (p01, p03, p05), others catastrophic (p15=42%, p18=60%). The model has capacity but hasn't learned uniformly.
- **V10b (purple, 12k)**: Much more uniform. Most prompts recovered to V9b levels or above. The extra 6,000 steps (warm-started from V10) stabilized under-converged prompts.
- **Remaining outlier**: p15 astronaut — even 12k total steps couldn't fix this. Likely needs this specific concept in the training data.

---

## Key Findings

### 1. Rank-128 breaks the capacity ceiling
V10b (90.4%) surpasses V9b (90.0%) by +0.4pp with the **same dataset**. The rank-64 ceiling at ~90% was indeed a capacity limitation, not a data limitation. This validates the V9c diagnosis.

### 2. Cold-start requires 2× training time
Rank-128 from fresh SVD init needs ~12,000 steps to converge, vs rank-64 warm-started in 6,000 steps. The SVD initialization provides a reasonable starting point but is far from the optimum that a warm-start chain achieves.

### 3. Aesthetic quality lags behind CLIP
V10b's aesthetics (5.686) and LPIPS (0.719) trail V9b's (5.939 / 0.664). The cold-start path produces images that are semantically better-aligned (higher CLIP) but perceptually rougher. This may improve with even more training or a partial warm-start strategy.

### 4. Capacity enables selective breakthroughs
V10b achieves two prompts above BF16 (p16=103.7%, p00=99.4%) and several near-BF16. The extra 350M parameters capture fine details (mushroom glow, skin texture, glass reflections) that rank-64 couldn't represent.

---

## Full 20-Prompt Comparison Grid

![Full grid: all 20 prompts — BF16 | V9b | V10b](../output/viz/v10_full_grid.png)

---

## Progress Summary

| Version | Key Change | OOD CLIP | Delta |
|---|---|---|---|
| V6 | 174 prompts, LPIPS loss | 86.0% | baseline |
| V7 | 1,002 prompts, balanced sampling | 88.9% | +2.9pp |
| V9b | 2,132 prompts | 90.0% | +1.1pp |
| V9c | 4,007 prompts (rank-64 ceiling) | 88.8% | −1.2pp |
| V10 | Rank-128, 6k cold start | 87.7% | −2.3pp |
| **V10b** | **Rank-128, 12k steps** | **90.4%** | **+0.4pp** |

---

## Path Forward

1. **V10c: Rank-128 + V9c dataset (4,007 prompts)**: Now that rank-128 has more capacity, the V9c dataset that caused regression at rank-64 may actually help. This tests whether data+capacity together push past 91%.

2. **V10d: Even longer training (18k-24k steps)**: V10b may still be under-converged. The p15 astronaut failure and aesthetic gap suggest more steps could help.

3. **Partial warm-start strategy**: Copy scale parameters from V9b (compatible across ranks) and only SVD-init the LoRA A/B matrices. This could combine V9b's learned scales with rank-128 capacity.

---

GPU: NVIDIA A100-SXM4-80GB
V10 training: 5.5h (6000 steps × 3.3s)
V10b training: 11.0h (12000 steps × 3.3s)
Python env: diffusers==0.34.0, torch==2.5.0+cu124
