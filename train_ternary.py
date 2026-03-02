"""
Self-supervised distillation: BF16 FLUX → Ternary FLUX.
Reproduces: https://chenglin-yang.github.io/1.58bit.flux.github.io/

Four loss modes:

  fm (BEST): Proper flow-matching distillation with pre-generated teacher latents.
    1. Pre-generate teacher latents: python generate_teacher_dataset.py
    2. Train: z_t = (1-t)*z_0 + t*eps, loss = MSE(student(z_t,t,c), teacher(z_t,t,c))
    Use --balanced-sampling to ensure equal gradient weight per unique prompt.
    Use --t-dist logit-normal for better intermediate timestep coverage.
    Usage: python train_ternary.py --loss-type fm --dataset output/teacher_dataset.pt \
           --balanced-sampling --t-dist logit-normal

  online: Online FM distillation. No pre-generated dataset needed.
    Pseudo-z_0 via teacher Euler denoising (--online-steps 1 = fast; 5-10 = high quality).
    Single-step (V3/V4) caused grid artifacts at high LR. Multi-step (--online-steps 5+)
    should solve this. Infinite diversity → no dataset memorization possible (V8 path).
    Usage: python train_ternary.py --loss-type online --online-steps 5

  output (baseline, wrong distribution): Teacher/student velocity MSE at random noise z_t.
    Trains on z_t=pure_noise for ALL t. Distribution shift → images stay noisy.
    Usage: python train_ternary.py --loss-type output

  layer (legacy): MSE on 29 intermediate block activations.
    Doesn't directly optimize final velocity → images stay noisy even as loss drops.
    Usage: python train_ternary.py --loss-type layer
"""
import os, sys, time, random, argparse, json, math
os.environ.setdefault("HF_HOME", "/home/jovyan/.cache/huggingface")

import torch
import torch.nn.functional as F
from pathlib import Path
from diffusers import FluxPipeline, FluxTransformer2DModel
import lpips as lpips_lib

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
    "A traditional Japanese geisha in full makeup, ornate kimono, cherry blossoms",
    "A medieval knight removing his helmet, battle-worn armor in dramatic lighting",
    "A Victorian gentleman with a top hat and monocle, foggy London street",
    "A female scientist in a futuristic laboratory, holographic interfaces",
    "A surfer riding a massive ocean wave at sunset, action photography",
    "A traditional Indian bride in red and gold attire, intricate jewelry",
    "A monk in saffron robes meditating at a mountaintop temple",
    "A pirate captain standing at the ship's helm, stormy seas",
    "A boxer mid-punch, dramatic arena lighting, sweat and intensity",
    "An elderly woman knitting by a fireplace, cozy winter scene",
    "A young girl in a sunlit flower field, reaching for butterflies",
    "A nomadic shepherd on a mountain pass, flock of sheep in mist",
    "A flamenco dancer in a red dress, mid-spin, dramatic stage light",
    "Portrait of an African tribal elder with ceremonial face paint",
    "A deep-sea diver exploring a sunken shipwreck, flashlight beams",
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
    "A misty fjord at dawn, rocky cliffs and mirror-calm water",
    "Terraced rice paddies in Bali, green layers cascading down the hillside",
    "A frozen waterfall in winter, ice crystals in morning light",
    "A bamboo forest path in Kyoto, filtered green light",
    "Thunderstorm over the Grand Canyon, multiple lightning strikes",
    "A field of lavender at sunset, purple rows to the horizon",
    "Tropical beach at night, bioluminescent waves glowing blue",
    "Autumn reflections in a still mountain lake, mirror image",
    "Red rock formations in the American Southwest, dramatic golden hour",
    "A moonlit forest clearing with fireflies, ethereal atmosphere",
    "Arctic ice floes at midnight sun, polar bear silhouette",
    "A sandstorm approaching a Middle Eastern desert city",
    "Misty Scottish highlands at dawn, ancient castle ruins",
    "A secret garden with overgrown roses and iron gates",
    "The Amazon rainforest canopy from above, morning mist",
    "A massive wave crashing on a rocky coastline, spray in air",
    # Animals & macro
    "Close-up of a red rose with water droplets, macro photography",
    "Macro photography of a butterfly on a purple flower",
    "Majestic lion on a rocky outcrop at sunrise, African savanna",
    "Hummingbird hovering beside tropical flowers, ultra-sharp detail",
    "A pod of dolphins leaping through a turquoise wave",
    "A wolf howling at the full moon on a snowy mountain ridge",
    "An eagle in flight over snow-capped mountain peaks, majestic",
    "A jaguar stalking silently through rainforest undergrowth, intense eyes",
    "Colorful macaws perched in a tropical jungle tree canopy",
    "An octopus displaying iridescent colors on a coral reef",
    "A polar bear mother and cubs on arctic sea ice",
    "A school of fish swirling in a tornado formation in clear ocean",
    "Extreme macro of a chameleon's eye, scales and iris detail",
    "A herd of elephants at a waterhole, dramatic dusty sunset",
    "A snowy owl perched on a snow-covered pine branch, winter",
    "A sea turtle gliding through crystal clear Caribbean water",
    "Two red foxes playing in autumn leaves in a forest",
    "A peacock displaying its full plumage, iridescent feathers",
    "A great white shark breaching the ocean surface, dramatic spray",
    "Macro shot of a dewdrop on a spider web at sunrise",
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
    "The interior of the Alhambra palace, intricate Moorish tile patterns",
    "A Moroccan medina at night, lanterns and ornate archways",
    "A traditional Japanese wooden pagoda in autumn foliage",
    "An ancient Roman Colosseum at golden hour, warm stone light",
    "A luxury rooftop terrace overlooking a modern city at night",
    "A Victorian greenhouse filled with exotic tropical plants",
    "Colorful favela houses cascading down a hillside in Brazil",
    "An art nouveau metro station with ornate iron and glass",
    "A shipping container port at night, cranes and colored containers",
    "A medieval cobblestone market town on a misty morning",
    "A futuristic underwater research station with glowing corridors",
    "An old wooden fishing pier at golden hour, reflections",
    "Inside a massive cave cathedral, natural light shafts",
    "A cyberpunk megacity at night, flying vehicles and holographic ads",
    "The Great Wall of China winding over mountain peaks in mist",
    "A floating village on stilts in Southeast Asia, golden light",
    # Fantasy & sci-fi
    "A dragon flying over a medieval castle at dusk",
    "A futuristic humanoid robot in a busy marketplace",
    "Abstract geometric art in vivid colors, oil on canvas",
    "Glowing bioluminescent forest at night, ethereal blue and green",
    "Steampunk airship fleet above Victorian city, dramatic storm clouds",
    "Alien planet with two moons, exotic bioluminescent flora",
    "Crystal cave with prismatic light refractions, underground lake",
    "A wizard casting a spell in a dark enchanted forest",
    "A giant ancient tree housing an entire elven city in its canopy",
    "A time traveler emerging from a glowing portal in Victorian London",
    "An underwater mermaid kingdom with coral spires and sea glass",
    "A portal to another dimension opening in a dark forest",
    "A spaceship graveyard drifting through an asteroid belt",
    "A phoenix rising from the ashes, golden fire and feathers",
    "An ice palace in an arctic wasteland, blue crystalline spires",
    "A giant mecha robot standing in a rain-soaked ruined city",
    "A fairytale cottage in an enchanted mushroom forest",
    "A cosmic deity silhouetted against a swirling galaxy",
    "An ancient sea kraken rising from a storm-tossed ocean",
    "A temple built inside an active volcano crater, lava moat",
    "A magical library with floating books and glowing orbs",
    "Alien spacecraft wreckage in a terrestrial desert, old and overgrown",
    "A necromancer summoning spirits in a graveyard at midnight",
    "A sky city of floating islands connected by vine bridges",
    # Still life & food
    "A freshly baked sourdough loaf on a rustic wooden table, warm tones",
    "Colorful Indian spices arranged in small bowls, overhead view",
    "Rainy window with a coffee cup and a book, cozy atmosphere",
    "A Japanese tea ceremony setup on a bamboo mat, zen simplicity",
    "Fresh sushi arrangement on black slate, artistic overhead composition",
    "Old leather-bound books stacked in a dusty antique library nook",
    "A collection of vintage glass bottles with colored liquids, sunlit",
    "Macro shot of a snowflake crystal on dark velvet, perfect geometry",
    "Freshly baked croissants with steam rising in a Parisian café",
    "A vibrant farmer's market stall with colorful vegetables and flowers",
    "A single white candle flame reflected in dark water, minimal",
    "A pile of autumn leaves in gold and red, texture shot from above",
    "A stack of colorful French macarons in a Parisian pastry shop",
    "Macro photography of honey dripping from a wooden spoon",
    "An elaborate charcuterie board on a marble table, overhead",
    # Vehicles & industry
    "Formula 1 racing car blurred at speed on a night circuit",
    "An old rusted ship on a beach, low tide, dramatic sky",
    "Inside a busy forge, molten metal pouring, sparks flying",
    "A vintage 1930s biplane banking in a summer sky, dramatic clouds",
    "A luxury yacht moored in a Mediterranean harbor at golden hour",
    "A classic 1960s American muscle car on a rain-slicked neon street",
    "A hot air balloon drifting over the Serengeti at sunrise",
    "A cargo ship battling massive storm waves, waves over the bow",
    "A nuclear submarine surfacing in the arctic, ice breaking",
    "An old steam locomotive in a mountain blizzard, headlight piercing",
    # Art styles
    "Impressionist painting of a Parisian boulevard in the rain",
    "Watercolor illustration of a magical treehouse village at dusk",
    "Charcoal sketch style portrait of a Renaissance nobleman",
    "Pop art comic style superhero action scene",
    "A ukiyo-e style woodblock print scene, Mount Fuji in the distance",
    "A baroque oil painting of a lavish royal feast, candlelight",
    "A cubist portrait of a woman, fragmented geometric planes",
    "A pointillist park scene, thousands of tiny colored dots",
    "A Chinese ink wash painting of misty mountain valleys",
    "An art deco travel poster for Paris, 1920s geometric style",
    "A psychedelic poster, swirling neon colors and fractals",
    "A vintage Soviet propaganda poster style illustration",
    "A detailed pen and ink scientific illustration of a nautilus",
    "A neon sign art installation in a dark gallery space",
    "A linocut print of an ocean scene, bold black and white lines",
    # Emotional & conceptual
    "A silhouette of a lone figure standing at a cliff edge at sunset",
    "Two children sharing a secret, sitting in a sunlit treehouse",
    "An elderly violinist playing in an empty concert hall, evening light",
    "A soldier returning home, tearful reunion at a train station",
    "A crowd of protesters with candles in the night, peaceful vigil",
    "A child reading under a tree, dappled light and wonder",
    "Two elderly friends playing chess in a sunny park, laughing",
    "A person standing in the rain at a crossroads, looking up",
    # Extra diversity
    "Neon-lit karaoke bar interior, Tokyo, late night festive crowd",
    "A traditional Moroccan hammam, steam and intricate tilework",
    "A carnival funhouse mirror, surreal distorted reflections",
    "A cave painting in Lascaux style, ancient ochre animals",
    "A professional esports tournament arena, screens and crowds",
    "A coral bleaching scene underwater, hauntingly beautiful",
    "A glass chess set on a transparent table, macro detail",
    "An origami swan in a field of rice paper, soft lighting",
]

EVAL_PROMPTS = [
    "Cyberpunk samurai on a neon-lit rooftop at dusk, dramatic rim lighting, 32-bit render",
    "A fantasy landscape with mountains and a river",
    "Portrait of a young woman with wild curly hair in golden light",
    "Aerial view of a coastal city at sunset",
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
                   help=("fm: proper FM distillation with pre-generated teacher latents (recommended). "
                         "online: on-the-fly pseudo-z_0 via teacher Euler denoising. Infinite diversity. "
                         "output: velocity MSE at random noise (wrong distribution, baseline). "
                         "layer: intermediate activation MSE (legacy)."))
    p.add_argument("--online-steps", type=int, default=1,
                   help="Number of Euler steps for pseudo-z_0 in online FM mode. "
                        "Default=1 (fast but noisy). 5-10 = much cleaner pseudo-z_0, "
                        "reduces grid artifacts, but increases per-step teacher cost.")
    p.add_argument("--dataset",    type=str,   default="output/teacher_dataset.pt",
                   help="Path to teacher latents dataset (required for --loss-type fm)")
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--grad-accum",         type=int,   default=1)
    p.add_argument("--no-svd",             action="store_true")
    p.add_argument("--init-ckpt",          type=str,   default=None,
                   help="Path to a prior checkpoint to warm-start from (loads scale+lora weights)")
    p.add_argument("--lpips-weight",       type=float, default=0.0,
                   help="Weight for LPIPS perceptual loss added on top of FM MSE loss. "
                        "Decodes student/teacher z_0 through frozen VAE and computes AlexNet "
                        "perceptual distance. Improves aesthetic quality / sharpness. "
                        "Recommended: 0.05-0.2. Default: 0 (disabled).")
    p.add_argument("--lpips-freq",         type=int,   default=1,
                   help="Compute LPIPS loss every N steps (default 1 = every step). "
                        "Set higher to reduce compute overhead.")
    p.add_argument("--t-dist",             type=str,   default="uniform",
                   choices=["uniform", "logit-normal"],
                   help="Timestep sampling distribution. "
                        "'uniform': t~U[0,1] (default, classic FM). "
                        "'logit-normal': t=sigmoid(N(0,0.5)), concentrates training budget at "
                        "intermediate t (0.3-0.7) where velocity field is most uncertain. "
                        "Improves sharpness / perceptual quality.")
    p.add_argument("--balanced-sampling",  action="store_true",
                   help="Sample training items by prompt then by item, rather than uniform over "
                        "all items. Ensures each unique prompt gets equal gradient weight "
                        "regardless of how many latent samples it has. "
                        "Recommended when dataset has unequal items-per-prompt.")
    p.add_argument("--calib-prompts-file", type=str, default=None,
                   help="Path to a text file with one prompt per line. When provided, these "
                        "prompts replace CALIB_PROMPTS for online-mode training embeddings. "
                        "Use prompts_1000.txt to get full 1000-prompt diversity in online mode.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    device = "cuda"
    dtype  = torch.bfloat16

    ckpt_tag  = f"r{args.rank}_res{args.res}_s{args.steps}_{args.loss_type}"
    if args.lpips_weight > 0:
        ckpt_tag += f"_lpips{args.lpips_weight:.0e}"
    ckpt_path = OUTPUT_DIR / f"ternary_distilled_{ckpt_tag}.pt"
    eval_dir  = OUTPUT_DIR / f"eval_ternary_{ckpt_tag}"
    log_path  = OUTPUT_DIR / f"training_log_{ckpt_tag}.json"

    print(f"=== Ternary FLUX Distillation ===")
    print(f"  loss={args.loss_type}, steps={args.steps}, rank={args.rank}, res={args.res}, "
          f"grad_accum={args.grad_accum}, grad_checkpointing={args.grad_checkpointing}")
    print(f"  t-dist={args.t_dist}, balanced-sampling={args.balanced_sampling}")
    if args.lpips_weight > 0:
        print(f"  LPIPS perceptual loss: weight={args.lpips_weight}, freq=every {args.lpips_freq} steps")

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
    if args.calib_prompts_file:
        with open(args.calib_prompts_file) as _f:
            _active_prompts = [line.strip() for line in _f if line.strip()]
        print(f"    Loaded {len(_active_prompts)} prompts from {args.calib_prompts_file}")
    else:
        _active_prompts = CALIB_PROMPTS
    calib_embeds = []
    with torch.no_grad():
        for prompt in _active_prompts:
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
    # Offload text encoders always; keep VAE on GPU when LPIPS is enabled
    for attr in ("text_encoder", "text_encoder_2"):
        m = getattr(pipe, attr, None)
        if m is not None: m.to("cpu")
    if args.lpips_weight == 0:
        vae_m = getattr(pipe, "vae", None)
        if vae_m is not None: vae_m.to("cpu")
    torch.cuda.empty_cache()
    print(f"    VRAM after offload: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # LPIPS perceptual loss model (AlexNet, ~50 MB)
    lpips_fn = None
    if args.lpips_weight > 0:
        print(f"\n[2c] Loading LPIPS model (AlexNet)...")
        lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
        lpips_fn.requires_grad_(False)
        lpips_fn.eval()
        print(f"    LPIPS ready. VAE kept on GPU for perceptual decode.")

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
        # Build prompt-level index for balanced sampling
        from collections import defaultdict
        _prompt_groups: dict = defaultdict(list)
        for _i, _item in enumerate(fm_dataset):
            _prompt_groups[_item["prompt"]].append(_i)
        _unique_prompts = list(_prompt_groups.keys())
        n_unique = len(_unique_prompts)
        if args.balanced_sampling:
            print(f"    Balanced sampling: {n_unique} unique prompts "
                  f"(each prompt equiprobable regardless of sample count)")
        else:
            print(f"    Uniform item sampling: {len(fm_dataset)} items, "
                  f"{n_unique} unique prompts")
        teacher_cache = student_cache = None

    elif args.loss_type == "online":
        teacher_cache = student_cache = None
        _n_step_label = f"{args.online_steps}-step" if args.online_steps > 1 else "single-step"
        print(f"\n[3] Online FM distillation: pseudo-z_0 via {_n_step_label} teacher denoising.")
        print(f"    Each step: z_rand~N(0,I) → teacher {args.online_steps}-step Euler → z_0_pseudo → FM trajectory")
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
    fm_loss    = torch.tensor(0.0)
    lpips_loss = torch.tensor(0.0)

    for step in range(1, args.steps + 1):
        # Timestep sampling
        if args.t_dist == "logit-normal":
            # Logit-normal with sigma=0.5: concentrates around t=0.5
            # sigmoid(N(0, 0.5)) has ~68% of mass in [sigmoid(-0.5), sigmoid(0.5)] = [0.38, 0.62]
            u = random.gauss(0.0, 0.5)
            t_frac = 1.0 / (1.0 + math.exp(-u))
        else:
            t_frac = random.uniform(0.0, 1.0)

        if args.loss_type == "fm":
            # ---- Correct flow-matching: z_t = (1-t)*z_0 + t*eps ----
            if args.balanced_sampling:
                # Sample prompt uniformly → then sample a random item from that prompt's items
                pkey = random.choice(_unique_prompts)
                item = fm_dataset[random.choice(_prompt_groups[pkey])]
            else:
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
            fm_loss = F.mse_loss(v_student, v_teacher)

            # Perceptual (LPIPS) loss: decode z_0_pred and z_0 through frozen VAE
            lpips_loss = torch.tensor(0.0, device=device)
            # Only apply LPIPS when t_frac < 0.8: at high noise levels z_0_pred is too
            # inaccurate to provide a stable perceptual supervision signal.
            if lpips_fn is not None and step % args.lpips_freq == 0 and t_frac < 0.8:
                # Reconstruct z_0 prediction from student velocity
                # FM: z_t = (1-t)*z_0 + t*eps  →  z_0 = z_t - t*v
                z_0_pred = z_t.float() - t_frac * v_student  # [1, seq, 64], float32
                # Unpack packed FLUX latents → spatial [1, C, H, W]
                _vae_sf = pipe.vae_scale_factor
                z_0_pred_sp = FluxPipeline._unpack_latents(
                    z_0_pred.to(dtype), args.res, args.res, _vae_sf)           # bfloat16
                z_0_pred_sp = (z_0_pred_sp / pipe.vae.config.scaling_factor
                               + pipe.vae.config.shift_factor)
                img_student = pipe.vae.decode(z_0_pred_sp).sample.float()      # [1,3,H,W]
                img_student = img_student.clamp(-1.0, 1.0)
                # Downsample to 256px for LPIPS (saves memory; perceptual features don't need full res)
                img_student_sm = F.interpolate(img_student, size=256,
                                               mode="bilinear", align_corners=False)
                with torch.no_grad():
                    z_0_ref_sp = FluxPipeline._unpack_latents(
                        z_0.to(dtype), args.res, args.res, _vae_sf)
                    z_0_ref_sp = (z_0_ref_sp / pipe.vae.config.scaling_factor
                                  + pipe.vae.config.shift_factor)
                    img_teacher_dec = pipe.vae.decode(z_0_ref_sp).sample.float()
                    img_teacher_sm  = F.interpolate(
                        img_teacher_dec.clamp(-1.0, 1.0), size=256,
                        mode="bilinear", align_corners=False)
                lpips_loss = lpips_fn(img_student_sm, img_teacher_sm).mean()

            loss = fm_loss + args.lpips_weight * lpips_loss

        elif args.loss_type == "online":
            # ---- Online FM: pseudo-z_0 via multi-step teacher Euler denoising ----
            # With --online-steps=1: fast but noisy (original V3/V4, caused artifacts at high LR)
            # With --online-steps=5-10: much cleaner pseudo-z_0 → better training signal
            emb = random.choice(calib_embeds)
            pe  = emb["prompt_embeds"].to(device=device, dtype=dtype)
            poe = emb["pooled_embeds"].to(device=device, dtype=dtype)
            ti  = emb["text_ids"].to(device=device, dtype=dtype)

            # Start from pure Gaussian noise (t=1.0) and denoise with N Euler steps to t=0
            raw_rand = torch.randn(1, _num_ch, _lat_h, _lat_w, device=device, dtype=dtype)
            z = FluxPipeline._pack_latents(raw_rand, 1, _num_ch, _lat_h, _lat_w)
            n_denoise = max(1, args.online_steps)
            with torch.no_grad():
                for k in range(n_denoise):
                    # Timestep goes from 1.0 down to dt (N steps, ending just above 0)
                    t_k  = 1.0 - k / n_denoise
                    dt   = 1.0 / n_denoise
                    v_k  = teacher(
                        hidden_states=z,
                        timestep=torch.tensor([t_k], device=device, dtype=dtype),
                        guidance=torch.tensor([3.5], device=device, dtype=dtype),
                        encoder_hidden_states=pe,
                        pooled_projections=poe,
                        txt_ids=ti,
                        img_ids=_img_ids,
                        return_dict=False,
                    )[0]
                    z = z - dt * v_k  # Euler step: z_{t-dt} = z_t - dt * v(z_t, t)
            z_0_pseudo = z.detach()

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
            fm_loss = F.mse_loss(v_student, v_teacher)

            # Perceptual (LPIPS) loss: student z_0_pred vs pseudo-z_0 reference
            lpips_loss = torch.tensor(0.0, device=device)
            if lpips_fn is not None and step % args.lpips_freq == 0 and t_frac < 0.8:
                z_0_pred = z_t.float() - t_frac * v_student
                _vae_sf = pipe.vae_scale_factor
                z_0_pred_sp = FluxPipeline._unpack_latents(
                    z_0_pred.to(dtype), args.res, args.res, _vae_sf)
                z_0_pred_sp = (z_0_pred_sp / pipe.vae.config.scaling_factor
                               + pipe.vae.config.shift_factor)
                img_student = pipe.vae.decode(z_0_pred_sp).sample.float().clamp(-1.0, 1.0)
                img_student_sm = F.interpolate(img_student, size=256,
                                               mode="bilinear", align_corners=False)
                with torch.no_grad():
                    z_0_ref_sp = FluxPipeline._unpack_latents(
                        z_0_pseudo.to(dtype), args.res, args.res, _vae_sf)
                    z_0_ref_sp = (z_0_ref_sp / pipe.vae.config.scaling_factor
                                  + pipe.vae.config.shift_factor)
                    img_ref = pipe.vae.decode(z_0_ref_sp).sample.float().clamp(-1.0, 1.0)
                    img_ref_sm = F.interpolate(img_ref, size=256,
                                               mode="bilinear", align_corners=False)
                lpips_loss = lpips_fn(img_student_sm, img_ref_sm).mean()

            loss = fm_loss + args.lpips_weight * lpips_loss

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
            lr_l = optimizer.param_groups[1]["lr"]
            lpips_str = (f" | lpips={lpips_loss.item():.4f}" if lpips_fn is not None else "")
            fm_str    = (f" | fm={fm_loss.item():.5f}" if lpips_fn is not None else "")
            print(f"  step {step:4d}/{args.steps} | loss={loss.item():.5f}"
                  f"{fm_str}{lpips_str}"
                  f" | lr={lr_l:.2e} | {elapsed:.0f}s elapsed")
            log.append({"step": step, "loss": loss.item(),
                        "fm_loss": fm_loss.item() if args.loss_type == "fm" else None,
                        "lpips_loss": lpips_loss.item() if lpips_fn is not None else None})
            sys.stdout.flush()

        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n  [eval] step {step}...")
            pipe.transformer.eval()
            # Text encoders always need to come to GPU for inference; VAE may already be there
            for attr in ("text_encoder", "text_encoder_2", "vae"):
                m = getattr(pipe, attr, None)
                if m is not None: m.to(device)
            eval_images(pipe, step, eval_dir)
            # After eval: offload text encoders. Only offload VAE if LPIPS is not using it.
            for attr in ("text_encoder", "text_encoder_2"):
                m = getattr(pipe, attr, None)
                if m is not None: m.to("cpu")
            if args.lpips_weight == 0:
                vae_m = getattr(pipe, "vae", None)
                if vae_m is not None: vae_m.to("cpu")
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
