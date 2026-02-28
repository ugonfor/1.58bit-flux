"""
Generate teacher (BF16) latents for flow-matching distillation.

The 1.58-bit FLUX paper uses self-supervised fine-tuning:
  1. Generate images from BF16 teacher using text prompts only.
  2. Encode to latent z_0.
  3. Train student with proper flow matching: z_t = (1-t)*z_0 + t*eps, velocity = eps - z_0.

This script handles step 1+2: generates z_0 latents from BF16 teacher.

Usage:
  python generate_teacher_dataset.py --n-images 50 --steps 28 --seed 42
  → saves output/teacher_dataset.pt
"""
import os, argparse, time, torch
os.environ.setdefault("HF_HOME", "/home/jovyan/.cache/huggingface")

from pathlib import Path
from diffusers import FluxPipeline
from models.ternary import memory_stats

MODEL_NAME = "black-forest-labs/FLUX.1-dev"
OUTPUT_DIR = Path("output")

PROMPTS = [
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
    # Extra diversity
    "Macro shot of a dewdrop on a spider web at sunrise",
    "Neon-lit karaoke bar interior, Tokyo, late night crowd",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-images",  type=int, default=50)
    p.add_argument("--steps",     type=int, default=28)
    p.add_argument("--res",       type=int, default=1024)
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--out",       type=str, default="output/teacher_dataset.pt")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda"
    dtype  = torch.bfloat16

    print(f"=== Teacher Dataset Generation ===")
    print(f"  n_images={args.n_images}, steps={args.steps}, res={args.res}")

    print("\nLoading BF16 pipeline...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, local_files_only=True,
    ).to(device)
    print(f"  VRAM after load: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    dataset = []
    t0 = time.time()

    for i in range(args.n_images):
        prompt = PROMPTS[i % len(PROMPTS)]
        seed   = args.seed + i
        gen = torch.Generator("cuda").manual_seed(seed)

        # Generate with output_type="latent" → packed z_0 [1, seq_len, 64]
        with torch.no_grad():
            result = pipe(
                prompt,
                generator=gen,
                num_inference_steps=args.steps,
                guidance_scale=3.5,
                height=args.res,
                width=args.res,
                output_type="latent",
            )
        latent = result.images.cpu()
        del result  # release inference activations

        # Encode prompt (text encoders on GPU)
        with torch.no_grad():
            pe, poe, ti = pipe.encode_prompt(
                prompt=prompt, prompt_2=None, device=device,
                num_images_per_prompt=1, max_sequence_length=256,
            )

        dataset.append({
            "latent_z0":     latent,
            "prompt_embeds": pe.cpu(),
            "pooled_embeds": poe.cpu(),
            "text_ids":      ti.cpu(),
            "prompt":        prompt,
            "seed":          seed,
        })
        del pe, poe, ti
        torch.cuda.empty_cache()  # release caching allocator memory after each image

        elapsed = time.time() - t0
        remaining = elapsed / (i + 1) * (args.n_images - i - 1)
        print(f"  [{i+1:3d}/{args.n_images}] '{prompt[:50]}...' | "
              f"elapsed={elapsed:.0f}s eta={remaining:.0f}s")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, out_path)
    print(f"\nSaved {len(dataset)} items → {out_path}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")

    # Verify latent shape
    sample = dataset[0]
    print(f"  Latent z_0 shape: {sample['latent_z0'].shape}")
    print(f"  Prompt embeds: {sample['prompt_embeds'].shape}")


if __name__ == "__main__":
    main()
