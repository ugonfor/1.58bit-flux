# Post 008: V7 — Scaling to 1,000 Prompts + Three Training Improvements

**Date**: 2026-03-02
**Status**: Training in progress — results TBD

---

## What Post 007 Taught Us

Post 007 exposed the hard truth: our model was **memorizing**, not generalizing.

| Eval set | V6 CLIP | BF16 CLIP | Ratio |
|---|---|---|---|
| 4 training prompts | 0.3300 | 0.3448 | **102.5%** |
| 20 unseen diverse | 0.2901 | 0.3375 | **86.0%** |

On unseen prompts: sushi → abstract red blobs, astronaut → featureless sphere, northern lights → dark winter canyon. The 14% CLIP gap and 25% LPIPS gap were caused by one root problem: **174 training prompts, vs the paper's 7,232**.

V7 attacks this directly.

---

## V7: Four Changes

### Change 1 — Dataset Scale: 174 → 1,000 Unique Prompts

Generated 826 new BF16 teacher latents at 1024px, covering concepts completely absent from previous training:

| New category | Example prompts |
|---|---|
| Science / tech | Particle accelerator, quantum computer, fusion plasma |
| Nature extremes | Supercell thunderstorm, fire whirl, ice caves |
| Macro photography | Mantis shrimp strike, spider web dew, tardigrade SEM |
| Sports / action | Wingsuit flying, free solo climbing, surf barrel ride |
| Music & crafts | Murano glassblower, Venetian glasswork, kintsugi repair |
| Weather | Mammatus clouds, Fata Morgana mirage, firenado |
| Gardens & seasons | Giant water lily, corpse flower, frost patterns |

Merged dataset: **548 existing + 826 new = 1,374 images** from **1,000 unique prompts**.

---

### Change 2 — Balanced Prompt Sampling

**The problem**: the merged dataset has ~3 images per old prompt (174 prompts) vs 1 image per new prompt (826 prompts). With plain `random.choice(dataset)`, old prompts got **3× more gradient signal** than new prompts.

```python
# Before (V5/V6): item-level uniform sampling
item = random.choice(fm_dataset)

# V7: prompt-level uniform sampling
prompt_key = random.choice(unique_prompts)  # 1000 prompts, all equal probability
item = fm_dataset[random.choice(prompt_groups[prompt_key])]
```

This ensures all 1,000 prompts get identical gradient weight, regardless of how many latent samples they have. **This directly attacks the training/OOD gap.**

---

### Change 3 — Logit-Normal Timestep Sampling

**The problem**: uniform `t ~ U[0,1]` wastes training budget at the tails:
- `t ≈ 0`: almost clean signal, student velocity is already near-correct
- `t ≈ 1`: pure noise, velocity prediction is nearly undetermined

**V7**: `t = sigmoid(N(0, 0.5))`, concentrating ~68% of steps in t ∈ [0.38, 0.62].

```python
u = random.gauss(0.0, 0.5)
t_frac = 1.0 / (1.0 + math.exp(-u))  # logit-normal
```

This is the same strategy used by FLUX and Stable Diffusion 3 during BF16 training. At intermediate t, the velocity field carries the most semantic information — this is where matching teacher → student matters most.

---

### Change 4 — More Training Steps: 3,000 → 4,000

With 1,000 unique prompts and grad_accum=4, 3,000 steps gives ~0.75 optimizer updates per unique prompt. At 4,000 steps, each prompt sees ~1 optimizer update on average.

---

## V7 Training Config

| Setting | V6 | V7 | Change |
|---|---|---|---|
| Dataset | 548 imgs, 174 prompts | 1,374 imgs, 1,000 prompts | **+5.7× prompts** |
| Item sampling | uniform | balanced (prompt-level) | **new** |
| t-distribution | uniform | logit-normal σ=0.5 | **new** |
| Steps | 2,000 | 4,000 | **2×** |
| Warm-start | V5 | V6 | — |
| lr-lora | 5e-5 | 1e-4 | slightly higher |
| LPIPS weight | 0.1 | 0.1 | same |

---

## Results

*(Results pending — training in progress)*

### Fixed-Prompt Eval (4 training prompts)

| Model | CLIP | Aesthetic | LPIPS↓ |
|---|---|---|---|
| BF16 | 0.3448 | 5.63 | ref |
| V6 | 0.3300 | 5.73 | 0.524 |
| **V7** | **TBD** | **TBD** | **TBD** |

### Diverse-Prompt Eval (20 unseen prompts) — the honest benchmark

| Model | CLIP | Aesthetic | LPIPS↓ | vs BF16 |
|---|---|---|---|---|
| BF16 | 0.3375 | 5.84 | ref | 100% |
| V6 | 0.2901 | 5.64 | 0.658 | **86%** |
| **V7** | **TBD** | **TBD** | **TBD** | **TBD** |

### Hypothesis

With 5.7× more unique prompts and balanced gradient distribution, V7 should significantly close the 14% OOD gap. The specific failure modes from post-007:
- **Sushi**: should improve — food prompts are now in training data
- **Northern lights**: should improve — weather/aurora prompts added
- **Astronaut**: likely improved — space/sci-fi prompts added

We expect V7 OOD CLIP ≥ 0.30 (vs V6's 0.2901), though matching BF16's 0.3375 would require ~7,000 prompts (like the paper).

---

## Path Forward: V8 — Multi-Step Online FM

If V7 still shows a significant OOD gap, the next lever is **online FM with multi-step denoising**. The original V3/V4 online approach failed because single-step pseudo-z_0 was too noisy. The fix: use N=5-10 Euler denoising steps to get a high-quality pseudo-z_0, then train on that trajectory.

```python
# V8: multi-step online FM (--online-steps 5)
z = gaussian_noise()
for k in range(5):  # 5 Euler steps from t=1.0 to t=0
    t_k = 1.0 - k / 5
    v_k = teacher(z, t=t_k, c=text_embed)
    z = z - (1/5) * v_k
z_0_pseudo = z  # much cleaner than single-step
```

This gives **infinite prompt diversity** (no dataset needed) with quality close to full-sampling BF16. The V4 grid artifacts should disappear at reasonable LR since the pseudo-z_0 is now a proper image rather than a single-step rough approximation.

---

GPU: NVIDIA A100-SXM4-80GB
Dataset generation: ~3.2h (826 images × ~14s/image)
Training: ~3h (4000 steps on A100)
Python env: diffusers==0.34.0, torch==2.5.0+cu124
