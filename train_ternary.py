"""
Self-supervised distillation: BF16 FLUX → Ternary FLUX.
Reproduces: https://chenglin-yang.github.io/1.58bit.flux.github.io/

Four loss modes:

  online (BEST): Online flow-matching distillation. No pre-generated dataset needed.
    Each step: sample random Gaussian z_rand, run teacher 1-step to get pseudo-z_0,
    then FM-train on that trajectory. Infinite diversity → no memorization.
    z_0_pseudo = z_rand - t_large * teacher(z_rand, t_large, c)  [single Euler step]
    z_t = (1-t)*z_0_pseudo + t*eps, loss = MSE(student(z_t,t,c), teacher(z_t,t,c))
    Usage: python train_ternary.py --loss-type online

  fm: Proper flow-matching distillation with pre-generated teacher latents.
    1. Pre-generate teacher latents: python generate_teacher_dataset.py
    2. Train: z_t = (1-t)*z_0 + t*eps, loss = MSE(student(z_t,t,c), teacher(z_t,t,c))
    Limited by dataset size — model may memorize fixed trajectories.
    Usage: python train_ternary.py --loss-type fm --dataset output/teacher_dataset.pt

  output (baseline, wrong distribution): Teacher/student velocity MSE at random noise z_t.
    Trains on z_t=pure_noise for ALL t. Distribution shift → images stay noisy.
    Usage: python train_ternary.py --loss-type output

  layer (legacy): MSE on 29 intermediate block activations.
    Doesn't directly optimize final velocity → images stay noisy even as loss drops.
    Usage: python train_ternary.py --loss-type layer
"""
import os, sys, time, random, argparse, json
os.environ.setdefault("HF_HOME", "/home/jovyan/.cache/huggingface")

import torch
import torch.nn.functional as F
from pathlib import Path
from diffusers import FluxPipeline, FluxTransformer2DModel

from models.ternary import quantize_to_ternary, memory_stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "black-forest-labs/FLUX.1-dev"
OUTPUT_DIR = Path("output")

CALIB_PROMPTS = [
    # People & portraits
    "Cyberpunk samurai on a neon-lit rooftop at dusk, dramatic rim lighting",
    "Portrait of a young woman with wild curly hair in golden light",
    "Elderly fisherman with weathered face and deep-set eyes, black and white portrait",
    "Young musician playing guitar on a stage with dramatic spotlights",
    "Astronaut floating in space with Earth in background",
    "A chef preparing food in a modern open kitchen",
    "Ballet dancer mid-leap in a sunlit studio, motion blur",
    "Street photographer in the rain, Tokyo, neon reflections on wet pavement",
    # Nature & landscapes
    "A fantasy landscape with mountains and a river",
    "A cozy wooden cabin in a snowy forest at night, warm interior glow",
    "Aerial view of a coastal city at sunset",
    "Ancient temple ruins covered in vines in a tropical jungle",
    "Underwater scene with colorful tropical fish and coral reef",
    "Autumn forest path with golden leaves and soft fog",
    "Desert sand dunes at sunrise, long shadow patterns",
    "Volcanic eruption at night with lava flowing into the ocean",
    "Northern lights over a frozen tundra with a lone wolf silhouette",
    "Cherry blossom trees lining a river in Japan at golden hour",
    # Animals & macro
    "Close-up of a red rose with water droplets, macro photography",
    "Macro photography of a butterfly on a purple flower",
    "Majestic lion on a rocky outcrop at sunrise, African savanna",
    "Hummingbird hovering beside tropical flowers, ultra-sharp detail",
    "A pod of dolphins leaping through a turquoise wave",
    # Architecture & urban
    "Minimalist black and white architectural photograph, symmetry",
    "A steam locomotive crossing a mountain bridge, dramatic clouds",
    "Oil painting of a harbor with sailing boats at golden hour",
    "Street art mural on a building wall, vibrant graffiti style",
    "Post-apocalyptic overgrown city with nature reclaiming streets",
    "Spiral staircase in an old European library, warm amber light",
    "Brutalist concrete skyscraper at night, rain-slicked plaza",
    "Gothic cathedral interior with colorful stained glass flooding light",
    "Tokyo street intersection at night, crowds and neon signs",
    # Fantasy & sci-fi
    "A dragon flying over a medieval castle at dusk",
    "A futuristic humanoid robot in a busy marketplace",
    "Abstract geometric art in vivid colors, oil on canvas",
    "Glowing bioluminescent forest at night, ethereal blue and green",
    "Steampunk airship fleet above Victorian city, dramatic storm clouds",
    "Alien planet with two moons, exotic flora in foreground",
    "Crystal cave with prismatic light refractions, underground lake",
    "A wizard casting a spell in a dark enchanted forest",
    # Still life & food
    "A freshly baked sourdough loaf on a rustic wooden table, warm tones",
    "Colorful Indian spices arranged in small bowls, overhead view",
    "Rainy window with a coffee cup and a book, cozy atmosphere",
    # Vehicles & industry
    "Formula 1 racing car blurred at speed on a night circuit",
    "An old rusted ship on a beach, low tide, dramatic sky",
    "Inside a busy forge, molten metal pouring, sparks flying",
    # Art styles
    "Impressionist painting of a Parisian boulevard in the rain",
    "Watercolor illustration of a magical treehouse village at dusk",
    "Charcoal sketch style portrait of a Renaissance nobleman",
    "Pop art comic style superhero action scene",
]

EVAL_PROMPTS = [
    "Cyberpunk samurai on a neon-lit rooftop at dusk, dramatic rim lighting, 32-bit render",
    "A fantasy landscape with mountains and a river",
]


# ---------------------------------------------------------------------------
# Layer-wise distillation helpers (legacy loss-type=layer)
# ---------------------------------------------------------------------------
class ActivationCache:
    """Stores intermediate block activations for layer-wise MSE."""
    def __init__(self):
        self.cache = {}
        self._hooks = []

    def register(self, model, double_every=1, single_every=4):
        for i, block in enumerate(model.transformer_blocks):
            if i % double_every == 0:
                h = block.register_forward_hook(self._make_hook(f"d{i}"))
                self._hooks.append(h)
        for i, block in enumerate(model.single_transformer_blocks):
            if i % single_every == 0:
                h = block.register_forward_hook(self._make_hook(f"s{i}"))
                self._hooks.append(h)
        return self

    def _make_hook(self, key):
        def hook(module, inp, output):
            if isinstance(output, (tuple, list)):
                self.cache[key] = torch.cat(
                    [o for o in output if isinstance(o, torch.Tensor)], dim=1)
            else:
                self.cache[key] = output
        return hook

    def clear(self):
        self.cache.clear()

    def remove(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()


def layer_wise_loss(teacher_cache: dict, student_cache: dict) -> torch.Tensor:
    """Activation-variance-normalized MSE across all matched block outputs."""
    assert teacher_cache.keys() == student_cache.keys()
    losses, weights = [], []
    for key in teacher_cache:
        t = teacher_cache[key].float().detach()
        s = student_cache[key].float()
        mse = F.mse_loss(s, t)
        losses.append(mse)
        weights.append(1.0 / t.var().clamp(min=1e-6))
    losses_t  = torch.stack(losses)
    weights_t = torch.stack(weights)
    return (losses_t * weights_t).sum() / weights_t.sum()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_images(pipe, step: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(EVAL_PROMPTS):
        gen = torch.Generator("cuda").manual_seed(42)
        img = pipe(prompt, generator=gen,
                   num_inference_steps=28, guidance_scale=3.5).images[0]
        img.save(out_dir / f"step{step:04d}_p{i}.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps",      type=int,   default=3000)
    p.add_argument("--rank",       type=int,   default=64)
    p.add_argument("--lr-lora",    type=float, default=3e-4)
    p.add_argument("--lr-scale",   type=float, default=1e-3)
    p.add_argument("--eval-every", type=int,   default=500)
    p.add_argument("--res",        type=int,   default=1024)
    p.add_argument("--loss-type",  type=str,   default="fm",
                   choices=["fm", "online", "output", "layer"],
                   help=("online (best): FM distillation with on-the-fly pseudo-z_0 via single-step "
                         "teacher denoising. Infinite diversity, no memorization. No dataset needed. "
                         "fm: proper FM distillation with pre-generated teacher latents (limited by dataset). "
                         "output: velocity MSE at random noise (wrong distribution, baseline). "
                         "layer: intermediate activation MSE (legacy)."))
    p.add_argument("--dataset",    type=str,   default="output/teacher_dataset.pt",
                   help="Path to teacher latents dataset (required for --loss-type fm)")
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--grad-accum",         type=int,   default=1)
    p.add_argument("--no-svd",             action="store_true")
    p.add_argument("--init-ckpt",          type=str,   default=None,
                   help="Path to a prior checkpoint to warm-start from (loads scale+lora weights)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    device = "cuda"
    dtype  = torch.bfloat16

    ckpt_tag  = f"r{args.rank}_res{args.res}_s{args.steps}_{args.loss_type}"
    ckpt_path = OUTPUT_DIR / f"ternary_distilled_{ckpt_tag}.pt"
    eval_dir  = OUTPUT_DIR / f"eval_ternary_{ckpt_tag}"
    log_path  = OUTPUT_DIR / f"training_log_{ckpt_tag}.json"

    print(f"=== Ternary FLUX Distillation ===")
    print(f"  loss={args.loss_type}, steps={args.steps}, rank={args.rank}, res={args.res}, "
          f"grad_accum={args.grad_accum}, grad_checkpointing={args.grad_checkpointing}")

    # ------------------------------------------------------------------ #
    # 1. Load student pipeline (BF16 → ternary)
    # ------------------------------------------------------------------ #
    print("\n[1] Loading student pipeline...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, local_files_only=True,
    ).to(device)
    print(f"    BF16 transformer: {memory_stats(pipe.transformer)['total_mb']:.0f} MB")

    quantize_to_ternary(pipe.transformer, per_channel=True,
                        lora_rank=args.rank, svd_init=not args.no_svd)
    print(f"    Ternary transformer: {memory_stats(pipe.transformer)['total_mb']:.0f} MB")

    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=True)
        state = {k: v for k, v in pipe.transformer.named_parameters()}
        for n, t in ckpt.items():
            if n in state:
                state[n].data.copy_(t.to(dtype))
        print(f"    Loaded {len(ckpt)} tensors from {args.init_ckpt}")

    if args.grad_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
        print("    Gradient checkpointing: ENABLED")

    # ------------------------------------------------------------------ #
    # 2. Load teacher transformer (frozen BF16)
    # ------------------------------------------------------------------ #
    print("\n[2] Loading teacher transformer...")
    teacher = FluxTransformer2DModel.from_pretrained(
        MODEL_NAME, subfolder="transformer",
        torch_dtype=dtype, local_files_only=True,
    ).to(device)
    teacher.requires_grad_(False)
    teacher.eval()
    print(f"    Teacher: {memory_stats(teacher)['total_mb']:.0f} MB | "
          f"VRAM used: {torch.cuda.memory_allocated()/1024**3:.1f} / "
          f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f} GB")

    # ------------------------------------------------------------------ #
    # 2b. Pre-encode prompts → offload encoders + VAE
    # ------------------------------------------------------------------ #
    print("\n[2b] Pre-encoding calibration prompts...")
    calib_embeds = []
    with torch.no_grad():
        for prompt in CALIB_PROMPTS:
            pe, poe, ti = pipe.encode_prompt(
                prompt=prompt, prompt_2=None, device=device,
                num_images_per_prompt=1, max_sequence_length=256,
            )
            calib_embeds.append({
                "prompt_embeds": pe.cpu(),
                "pooled_embeds": poe.cpu(),
                "text_ids":      ti.cpu(),
            })
    print(f"    Encoded {len(calib_embeds)} prompts.")
    for attr in ("text_encoder", "text_encoder_2", "vae"):
        m = getattr(pipe, attr, None)
        if m is not None: m.to("cpu")
    torch.cuda.empty_cache()
    print(f"    VRAM after offload: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    _lat_h = 2 * (int(args.res) // (pipe.vae_scale_factor * 2))
    _lat_w = _lat_h
    _num_ch = pipe.transformer.config.in_channels // 4
    print(f"    Latent grid: {_lat_h}×{_lat_w}, channels: {_num_ch}")

    # ------------------------------------------------------------------ #
    # 3. Load teacher dataset (fm mode) or hooks (layer mode)
    # ------------------------------------------------------------------ #
    # img_ids are the same for all samples at a given resolution
    _img_ids = FluxPipeline._prepare_latent_image_ids(
        1, _lat_h // 2, _lat_w // 2, device, dtype)

    fm_dataset = None
    if args.loss_type == "fm":
        print(f"\n[3] Loading teacher dataset: {args.dataset}")
        fm_dataset = torch.load(args.dataset, map_location="cpu", weights_only=False)
        print(f"    Loaded {len(fm_dataset)} items. "
              f"Latent shape: {fm_dataset[0]['latent_z0'].shape}")
        print(f"    Flow-matching training: z_t = (1-t)*z_0 + t*eps, "
              f"target = teacher_velocity(z_t, t)")
        teacher_cache = student_cache = None

    elif args.loss_type == "online":
        teacher_cache = student_cache = None
        print(f"\n[3] Online FM distillation: pseudo-z_0 via single-step teacher denoising.")
        print(f"    Each step: z_rand~N(0,I) → teacher 1-step Euler → z_0_pseudo → FM trajectory")
        print(f"    Infinite diversity, no dataset memorization. Using {len(calib_embeds)} prompts.")

    elif args.loss_type == "layer":
        teacher_cache = ActivationCache().register(teacher, double_every=1, single_every=4)
        student_cache = ActivationCache().register(pipe.transformer, double_every=1, single_every=4)
        print(f"\n[3] Layer hooks: {len(teacher_cache._hooks)} teacher "
              f"+ {len(student_cache._hooks)} student")

    else:  # output (random-noise baseline)
        teacher_cache = student_cache = None
        print(f"\n[3] Output-velocity loss at random noise (distribution-shifted baseline).")

    # ------------------------------------------------------------------ #
    # 4. Optimizer
    # ------------------------------------------------------------------ #
    scale_params = [p for n, p in pipe.transformer.named_parameters() if n.endswith(".scale")]
    lora_params  = [p for n, p in pipe.transformer.named_parameters()
                    if n.endswith(".lora_A") or n.endswith(".lora_B")]

    optimizer = torch.optim.AdamW([
        {"params": scale_params, "lr": args.lr_scale, "weight_decay": 0.0},
        {"params": lora_params,  "lr": args.lr_lora,  "weight_decay": 1e-4},
    ], betas=(0.9, 0.999), eps=1e-8)

    # Cosine schedule over total training steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=1e-6)

    n_scale = sum(p.numel() for p in scale_params)
    n_lora  = sum(p.numel() for p in lora_params)
    print(f"    Trainable: {n_scale:,} scale + {n_lora:,} LoRA = {n_scale+n_lora:,} total")

    # ------------------------------------------------------------------ #
    # 5. Training loop
    # ------------------------------------------------------------------ #
    print(f"\n[4] Training for {args.steps} steps...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log = []
    t0  = time.time()
    grad_accum = max(1, args.grad_accum)

    pipe.transformer.train()
    optimizer.zero_grad()

    for step in range(1, args.steps + 1):
        t_frac = random.uniform(0.0, 1.0)

        if args.loss_type == "fm":
            # ---- Correct flow-matching: z_t = (1-t)*z_0 + t*eps ----
            item = random.choice(fm_dataset)
            z_0  = item["latent_z0"].to(device=device, dtype=dtype)   # [1, seq, 64] packed
            emb  = item
            eps  = torch.randn_like(z_0)
            # Linear interpolation along flow-matching trajectory
            z_t  = (1.0 - t_frac) * z_0 + t_frac * eps               # [1, seq, 64]
            inputs = {
                "hidden_states":         z_t,
                "timestep":              torch.tensor([t_frac], device=device, dtype=dtype),
                "guidance":              torch.tensor([3.5],   device=device, dtype=dtype),
                "encoder_hidden_states": emb["prompt_embeds"].to(device=device, dtype=dtype),
                "pooled_projections":    emb["pooled_embeds"].to(device=device, dtype=dtype),
                "txt_ids":               emb["text_ids"].to(device=device, dtype=dtype),
                "img_ids":               _img_ids,
            }
            with torch.no_grad():
                v_teacher = teacher(**inputs, return_dict=False)[0].float().detach()
            v_student = pipe.transformer(**inputs, return_dict=False)[0].float()
            loss = F.mse_loss(v_student, v_teacher)

        elif args.loss_type == "online":
            # ---- Online FM: pseudo-z_0 via single-step teacher Euler denoising ----
            # Reference: EfficientDM (ICLR 2024) + flow-matching adaptation
            emb = random.choice(calib_embeds)
            pe  = emb["prompt_embeds"].to(device=device, dtype=dtype)
            poe = emb["pooled_embeds"].to(device=device, dtype=dtype)
            ti  = emb["text_ids"].to(device=device, dtype=dtype)

            # Step 1: sample random Gaussian at a large timestep
            t_large = random.uniform(0.7, 0.95)
            raw_rand = torch.randn(1, _num_ch, _lat_h, _lat_w, device=device, dtype=dtype)
            z_rand = FluxPipeline._pack_latents(raw_rand, 1, _num_ch, _lat_h, _lat_w)

            # Step 2: teacher single-step Euler → pseudo-z_0
            # z_0_pseudo = z_rand - t_large * v_teacher(z_rand, t_large, c)
            with torch.no_grad():
                v_rough = teacher(
                    hidden_states=z_rand,
                    timestep=torch.tensor([t_large], device=device, dtype=dtype),
                    guidance=torch.tensor([3.5],     device=device, dtype=dtype),
                    encoder_hidden_states=pe,
                    pooled_projections=poe,
                    txt_ids=ti,
                    img_ids=_img_ids,
                    return_dict=False,
                )[0]
            z_0_pseudo = (z_rand - t_large * v_rough).detach()

            # Step 3: FM trajectory from pseudo-z_0
            eps = torch.randn_like(z_0_pseudo)
            z_t = (1.0 - t_frac) * z_0_pseudo + t_frac * eps

            # Step 4: teacher and student velocity at z_t
            inputs = {
                "hidden_states":         z_t,
                "timestep":              torch.tensor([t_frac], device=device, dtype=dtype),
                "guidance":              torch.tensor([3.5],   device=device, dtype=dtype),
                "encoder_hidden_states": pe,
                "pooled_projections":    poe,
                "txt_ids":               ti,
                "img_ids":               _img_ids,
            }
            with torch.no_grad():
                v_teacher = teacher(**inputs, return_dict=False)[0].float().detach()
            v_student = pipe.transformer(**inputs, return_dict=False)[0].float()
            loss = F.mse_loss(v_student, v_teacher)

        elif args.loss_type == "output":
            # ---- Random-noise baseline (wrong distribution, for comparison) ----
            emb = random.choice(calib_embeds)
            raw = torch.randn(1, _num_ch, _lat_h, _lat_w, device=device, dtype=dtype)
            latents = FluxPipeline._pack_latents(raw, 1, _num_ch, _lat_h, _lat_w)
            inputs = {
                "hidden_states":         latents,
                "timestep":              torch.tensor([t_frac], device=device, dtype=dtype),
                "guidance":              torch.tensor([3.5],   device=device, dtype=dtype),
                "encoder_hidden_states": emb["prompt_embeds"].to(device=device, dtype=dtype),
                "pooled_projections":    emb["pooled_embeds"].to(device=device, dtype=dtype),
                "txt_ids":               emb["text_ids"].to(device=device, dtype=dtype),
                "img_ids":               _img_ids,
            }
            with torch.no_grad():
                v_teacher = teacher(**inputs, return_dict=False)[0].float().detach()
            v_student = pipe.transformer(**inputs, return_dict=False)[0].float()
            loss = F.mse_loss(v_student, v_teacher)

        else:  # layer
            emb = random.choice(calib_embeds)
            raw = torch.randn(1, _num_ch, _lat_h, _lat_w, device=device, dtype=dtype)
            latents = FluxPipeline._pack_latents(raw, 1, _num_ch, _lat_h, _lat_w)
            inputs = {
                "hidden_states":         latents,
                "timestep":              torch.tensor([t_frac], device=device, dtype=dtype),
                "guidance":              torch.tensor([3.5],   device=device, dtype=dtype),
                "encoder_hidden_states": emb["prompt_embeds"].to(device=device, dtype=dtype),
                "pooled_projections":    emb["pooled_embeds"].to(device=device, dtype=dtype),
                "txt_ids":               emb["text_ids"].to(device=device, dtype=dtype),
                "img_ids":               _img_ids,
            }
            teacher_cache.clear()
            with torch.no_grad():
                teacher(**inputs, return_dict=False)
            student_cache.clear()
            pipe.transformer(**inputs, return_dict=False)
            loss = layer_wise_loss(teacher_cache.cache, student_cache.cache)

        (loss / grad_accum).backward()

        if step % grad_accum == 0 or step == args.steps:
            torch.nn.utils.clip_grad_norm_(scale_params + lora_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Scheduler steps every training step (preserves cosine shape)
        scheduler.step()

        if step % 10 == 0:
            elapsed = time.time() - t0
            lr_s = optimizer.param_groups[0]["lr"]
            lr_l = optimizer.param_groups[1]["lr"]
            print(f"  step {step:4d}/{args.steps} | loss={loss.item():.5f} "
                  f"| lr={lr_l:.2e} | {elapsed:.0f}s elapsed")
            log.append({"step": step, "loss": loss.item()})
            sys.stdout.flush()

        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n  [eval] step {step}...")
            pipe.transformer.eval()
            for attr in ("text_encoder", "text_encoder_2", "vae"):
                m = getattr(pipe, attr, None)
                if m is not None: m.to(device)
            eval_images(pipe, step, eval_dir)
            for attr in ("text_encoder", "text_encoder_2", "vae"):
                m = getattr(pipe, attr, None)
                if m is not None: m.to("cpu")
            torch.cuda.empty_cache()
            pipe.transformer.train()
            print(f"  [eval] → {eval_dir}/step{step:04d}_p*.png")
            sys.stdout.flush()

    # ------------------------------------------------------------------ #
    # 6. Save checkpoint + log
    # ------------------------------------------------------------------ #
    print(f"\n[5] Saving checkpoint → {ckpt_path}")
    save_dict = {
        name: param.data
        for name, param in pipe.transformer.named_parameters()
        if name.endswith((".scale", ".lora_A", ".lora_B"))
    }
    torch.save(save_dict, ckpt_path)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"    Saved {len(save_dict)} tensors | log → {log_path}")
    print(f"\n=== Done. Total time: {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()
