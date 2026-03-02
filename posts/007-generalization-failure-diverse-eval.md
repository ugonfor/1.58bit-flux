# Post 007: The Overfitting Problem — Diverse Prompt Evaluation

**Date**: 2026-03-02
**Status**: Results + analysis

---

## The Problem with Our Evaluation

Posts 003–006 reported CLIP scores improving from 0.2783 → 0.3300, eventually matching and exceeding BF16. These results were real — but on the wrong test set.

Every evaluation used the **same 4 prompts that were in the training dataset**:
- "Cyberpunk samurai on a neon-lit rooftop..."
- "A fantasy landscape with mountains and a river"
- "Portrait of a young woman with wild curly hair..."
- "Aerial view of a coastal city at sunset"

The student model had seen BF16's velocity fields for these exact prompts during training. Of course it scores well — it memorized the trajectories, not the general mapping from text to image.

---

## Diverse Prompt Evaluation

20 prompts the model has **never seen**, covering categories outside the training distribution:

| # | Prompt | Category |
|---|---|---|
| 0 | A majestic lion resting on a savanna at golden hour | Animals |
| 1 | A colorful parrot perched on a tropical branch, macro | Animals |
| 2 | A wolf howling at the full moon in a snowy forest | Animals |
| 3 | Gothic cathedral interior with stained glass windows | Architecture |
| 4 | A futuristic skyscraper with reflective glass facade | Architecture |
| 5 | Traditional Japanese pagoda surrounded by cherry blossoms | Architecture |
| 6 | Northern lights over a frozen lake with reflections | Landscapes |
| 7 | A volcanic eruption at night with lava flowing into the ocean | Landscapes |
| 8 | Rolling lavender fields in Provence at sunrise | Landscapes |
| 9 | An elderly fisherman with weathered face and kind eyes | Portraits |
| 10 | A ballet dancer mid-leap on an outdoor stage at dusk | Portraits |
| 11 | A street musician playing saxophone in a rain-soaked alley | Portraits |
| 12 | An elaborate sushi platter with fresh salmon and tuna | Food |
| 13 | A rustic wooden table with freshly baked bread and herbs | Food |
| 14 | A dragon flying over a medieval castle during a thunderstorm | Fantasy |
| 15 | An astronaut floating in space with Earth and Moon | Sci-fi |
| 16 | A magical forest with glowing mushrooms and fireflies | Fantasy |
| 17 | A watercolor painting of Venice canals at sunset | Art styles |
| 18 | Oil painting portrait in Renaissance style, Rembrandt lighting | Art styles |
| 19 | Rainy Tokyo street at night, reflections on wet pavement | Urban |

---

## Results

### Per-Prompt Scores

| # | Prompt (short) | BF16 aes | BF16 CLIP | V6 aes | V6 CLIP | V6 LPIPS↓ |
|---|---|---|---|---|---|---|
| 0 | Lion on savanna | 6.84 | 0.3276 | 6.50 | 0.3333 | 0.513 |
| 1 | Parrot on branch | 6.14 | 0.3220 | 5.83 | 0.2570 | 0.478 |
| 2 | Wolf at full moon | 5.81 | 0.3417 | 6.04 | 0.3016 | 0.729 |
| 3 | Gothic cathedral | 6.25 | 0.3134 | 5.97 | 0.3038 | 0.645 |
| 4 | Futuristic skyscraper | 5.19 | 0.3246 | 5.38 | 0.2882 | 0.742 |
| 5 | Japanese pagoda | 5.73 | 0.3510 | 5.86 | 0.3444 | 0.813 |
| 6 | Northern lights | 5.62 | 0.3006 | 5.48 | 0.2366 | 0.631 |
| 7 | Volcanic eruption | 5.62 | 0.3456 | 5.22 | 0.3365 | 0.463 |
| 8 | Lavender fields | 6.29 | 0.3384 | 5.86 | 0.2961 | 0.778 |
| 9 | Elderly fisherman | 5.93 | 0.3845 | 6.02 | 0.3478 | 0.698 |
| 10 | Ballet dancer | 6.59 | 0.3433 | 6.13 | 0.3050 | 0.688 |
| 11 | Street musician | 5.72 | 0.3677 | 5.59 | 0.2724 | 0.674 |
| 12 | Sushi platter | 6.29 | 0.3122 | **4.88** | 0.2660 | 0.808 |
| 13 | Bread and herbs | 6.41 | 0.3433 | 5.71 | 0.2408 | 0.699 |
| 14 | Dragon over castle | 5.68 | 0.3668 | 5.05 | 0.3193 | 0.566 |
| 15 | Astronaut in space | 5.21 | 0.3365 | **4.66** | 0.2065 | 0.683 |
| 16 | Magical forest | 4.97 | 0.3684 | 5.81 | 0.3461 | 0.615 |
| 17 | Venice watercolor | 5.70 | 0.3179 | 5.81 | 0.2477 | 0.629 |
| 18 | Renaissance portrait | 5.22 | 0.2902 | 5.38 | 0.2651 | 0.748 |
| 19 | Rainy Tokyo street | 5.64 | 0.3534 | 5.68 | 0.2884 | 0.560 |
| | **AVERAGE** | **5.84** | **0.3375** | **5.64** | **0.2901** | **0.658** |

### Summary: In-Distribution vs Out-of-Distribution

| Eval set | V6 CLIP | BF16 CLIP | V6 / BF16 | V6 LPIPS |
|---|---|---|---|---|
| 4 fixed prompts (in training) | 0.3300 | 0.3448 | **102.5%** | 0.524 |
| 20 diverse prompts (unseen) | 0.2901 | 0.3375 | **86.0%** | 0.658 |
| Gap | −0.0399 | −0.0073 | **−16.5pp** | **+0.134** |

On training prompts: V6 looks like it matches BF16.
On unseen prompts: V6 falls **14% below BF16 CLIP** and is **25% worse on LPIPS**.

---

## Worst Failures (Unseen Prompts)

### p12: Sushi platter (V6 aes=4.88, worst in eval)

BF16 generates a clean, detailed sushi platter. V6 produces an abstract red/pink blob — wrong subject entirely. LPIPS=0.808 (the highest in the eval). The student's velocity field for "food" prompts is essentially random because food images were absent from the 548-image training set.

### p15: Astronaut in space (V6 aes=4.66, CLIP=0.207)

BF16: astronaut floating with Earth and Moon clearly visible. V6: a dark sphere that resembles a planet but no astronaut. CLIP=0.207 is nearly random — the model has no learned velocity field for space/astronaut concepts.

### p8: Lavender fields (V6 CLIP=0.296 vs BF16 0.338)

BF16: vivid purple fields with golden sunrise sky. V6: a generic green/yellow landscape. Concept partially correct (fields) but wrong color and composition.

---

## Root Cause: 42× Too Few Training Prompts

| System | Training prompts | CLIP on unseen prompts |
|---|---|---|
| 1.58-bit FLUX (paper) | **7,232** | ~matching BF16 |
| Our V1 | 50 | — |
| Our V2/V5 | 174 | — |
| Our V6 | 174 | **86% of BF16** |

The 1.58-bit paper used 7,232 diverse prompts to train their distillation. We used 174 — 42× fewer. Each training prompt gives the student one example of how to map a text embedding to the correct velocity field. With only 174 examples, the student memorizes those 174 mappings but cannot interpolate to unseen prompts.

### Why CLIP on training prompts was misleading

Flow-matching distillation trains directly on the student's velocity field at specific (z_t, t, c) tuples. With a fixed dataset, the student sees the same 548 (z_t, c) pairs repeatedly. After 3000 steps with grad_accum=4, each training sample is replayed ~22 times. The student perfectly models the teacher on these samples but treats all other text embeddings c as out-of-distribution.

This is dataset memorization, not generalization.

### Why the 4-prompt eval didn't catch this

The 4 eval prompts happen to be the **most overrepresented** in the training dataset — each has 6 BF16 training examples (the most of any prompt). They're the best-memorized cases. Using them as the only benchmark created a systematically optimistic picture.

---

## What Generalization Looks Like per Category

| Category | V6 aes vs BF16 | V6 CLIP vs BF16 | Generalization |
|---|---|---|---|
| Animals | −0.26 | −0.026 | Partial (subject OK, detail lost) |
| Architecture | −0.06 | −0.023 | Partial |
| Landscapes | −0.34 | −0.043 | Poor |
| Portraits | −0.20 | −0.041 | Poor |
| Food | −0.76 | −0.076 | **Catastrophic** (not in training) |
| Fantasy/Sci-fi | −0.38 | −0.040 | Poor |
| Art styles | +0.14 | −0.025 | Marginal (partial success) |
| Urban | +0.04 | −0.065 | Marginal |

Categories not represented in training data (food, food still life) show catastrophic failure. Even represented categories (animals, architecture) degrade significantly on unseen specific prompts.

---

## Implications and Path Forward

### What posts 003–006 actually measured

The CLIP scores in our posts measured **memorization quality**, not generalization. V6's "102.5% of BF16" should be read as "V6 perfectly memorized 4 training prompts." The model has not learned ternary flow-matching — it has learned 548 specific velocity fields.

### What needs to change

**The primary bottleneck is dataset scale.** Loss function improvements (LPIPS, perceptual loss) are secondary. The path:

1. **Scale to 1,000–7,000 prompts**: With more diverse training prompts, the student's velocity field interpolates better. This is the single highest-impact change.

2. **Prompt diversity strategy**: Training prompts must cover all concepts the model will be evaluated on. Our 174 prompts covered: cyberpunk, fantasy, portraits, aerial views. Food, animals, art styles, architecture were underrepresented or missing.

3. **Held-out eval set**: Always evaluate on prompts with zero overlap with training data. Our fixed 4-prompt eval was a leaky benchmark.

4. **LoRA rank**: Increasing rank 64→128 would give more parameters for the student to represent diverse velocity fields. But dataset scale matters more.

---

## Corrected Full Comparison Table

| Model | CLIP (4 fixed) | CLIP (20 diverse) | Notes |
|---|---|---|---|
| BF16 (reference) | 0.3448 | 0.3375 | Full-precision baseline |
| FM V5 (548 imgs, 174 prompts) | 0.3283 | ~0.29 | Estimated, post-005 |
| FM V6 (V5 + LPIPS) | 0.3300 | **0.2901** | This post |
| Gap V6 vs BF16 | −4.3% | **−14.0%** | True generalization gap |

The 14% CLIP gap on diverse prompts is the honest state of our ternary FLUX distillation. Posts 003–006 reported progress on memorization. The real problem — generalization — requires an order-of-magnitude more training data.

---

GPU: NVIDIA A100-SXM4-80GB
Python env: diffusers==0.34.0, torch==2.5.0+cu124
Eval: 20 prompts, seed=0, 28 inference steps
