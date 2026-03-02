"""
Diverse prompt evaluation: BF16 vs V6 on 20 unseen prompts.

Generates images with BF16 and V6, then computes:
  - CLIP score (text-image alignment)
  - Aesthetic score (zero-shot CLIP proxy)
  - LPIPS vs BF16 (perceptual distance)

Saves:
  output/eval_diverse_bf16/   — BF16 reference images
  output/eval_diverse_v6/     — V6 images
  output/diverse_scores.txt   — metric table
  output/viz/diverse_grid.png — visual comparison grid

Usage:
  python eval_diverse.py --ckpt output/ternary_distilled_r64_res1024_s2000_fm_lpips1e-01.pt
"""
import os, sys, json, argparse, random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import lpips as lpips_lib
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from diffusers import FluxPipeline

from models.ternary import quantize_to_ternary

os.environ.setdefault("HF_HOME", "/home/jovyan/.cache/huggingface")
MODEL_NAME = "black-forest-labs/FLUX.1-dev"
CLIP_LOCAL  = ("/home/jovyan/.cache/huggingface/hub/"
               "models--openai--clip-vit-base-patch32/snapshots/"
               "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")

OUTPUT_DIR = Path("output")
VIZ_DIR    = OUTPUT_DIR / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# 20 diverse prompts — none overlap with training eval prompts
DIVERSE_PROMPTS = [
    # Animals
    "A majestic lion resting on a savanna at golden hour, dramatic side lighting",
    "A colorful parrot perched on a tropical branch, macro photography",
    "A wolf howling at the full moon in a snowy forest, moody blue lighting",
    # Architecture
    "Gothic cathedral interior with stained glass windows, rays of light",
    "A futuristic skyscraper with reflective glass facade at blue hour",
    "Traditional Japanese pagoda surrounded by cherry blossom trees",
    # Landscapes
    "Northern lights over a frozen lake with reflections, long exposure",
    "A volcanic eruption at night with lava flowing into the ocean",
    "Rolling lavender fields in Provence at sunrise, warm golden light",
    # People / Portraits
    "An elderly fisherman with weathered face and kind eyes, natural light",
    "A ballet dancer mid-leap on an outdoor stage at dusk",
    "A street musician playing saxophone in a rain-soaked alley, neon reflections",
    # Food / Still life
    "An elaborate sushi platter with fresh salmon and tuna, restaurant photography",
    "A rustic wooden table with freshly baked bread and herbs, warm kitchen light",
    # Sci-fi / Fantasy
    "A dragon flying over a medieval castle during a thunderstorm",
    "An astronaut floating in space with Earth and Moon in background",
    "A magical forest with glowing mushrooms and fireflies at night",
    # Abstract / Art styles
    "A watercolor painting of Venice canals at sunset, impressionistic style",
    "Oil painting portrait of a woman in Renaissance style, Rembrandt lighting",
    # Urban / Street
    "Rainy Tokyo street at night, reflections on wet pavement, neon signs",
]

AES_POSITIVE = [
    "a high quality, beautiful, detailed, sharp photograph",
    "award-winning photo, stunning, vibrant, well-composed, 4K HDR",
    "masterpiece, professional photography, cinematic, rich colors",
]
AES_NEGATIVE = [
    "a blurry, low quality, noisy, distorted image",
    "ugly, amateur, flat, washed out, dull colors, artifact",
]


def aesthetic_score(clip_model, clip_processor, img_pil, device):
    img_inputs  = clip_processor(images=[img_pil], return_tensors="pt").to(device)
    text_inputs = clip_processor(text=AES_POSITIVE + AES_NEGATIVE,
                                  return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        i_feat = clip_model.get_image_features(**img_inputs).float()
        t_feat = clip_model.get_text_features(**text_inputs).float()
    i_feat = i_feat / i_feat.norm(dim=-1, keepdim=True)
    t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
    sims      = (i_feat @ t_feat.T).squeeze(0)
    pos_score = sims[:len(AES_POSITIVE)].mean().item()
    neg_score = sims[len(AES_POSITIVE):].mean().item()
    raw   = pos_score - neg_score
    score = (raw + 0.15) / 0.30 * 10.0
    return float(torch.tensor(score).clamp(0, 10))


def clip_score_fn(clip_model, clip_processor, img_pil, prompt, device):
    inputs = clip_processor(text=[prompt], images=[img_pil],
                            return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = clip_model(**inputs)
    i = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    t = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
    return (i * t).sum().item()


THUMB = 384
FONT_SZ = 16


def load_font(size=FONT_SZ):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def annotate(img, lines):
    bar_h = (FONT_SZ + 5) * len(lines) + 6
    out   = Image.new("RGB", (img.width, img.height + bar_h), (20, 20, 20))
    out.paste(img, (0, 0))
    draw  = ImageDraw.Draw(out)
    font  = load_font(FONT_SZ)
    y = img.height + 3
    for line in lines:
        draw.text((4, y), line, fill=(220, 220, 220), font=font)
        y += FONT_SZ + 5
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default="output/ternary_distilled_r64_res1024_s2000_fm_lpips1e-01.pt")
    p.add_argument("--rank",   type=int, default=64)
    p.add_argument("--steps",  type=int, default=28)
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-gen", action="store_true",
                   help="Skip image generation (use existing files)")
    return p.parse_args()


def generate_images(pipe, prompts, out_dir, steps, seed, label):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Generating {len(prompts)} images [{label}] → {out_dir}")
    for i, prompt in enumerate(prompts):
        out_path = out_dir / f"p{i:02d}.png"
        if out_path.exists():
            print(f"    p{i:02d} already exists, skipping")
            continue
        gen = torch.Generator(device="cpu").manual_seed(seed + i)
        img = pipe(
            prompt=prompt,
            height=1024, width=1024,
            num_inference_steps=steps,
            guidance_scale=3.5,
            generator=gen,
        ).images[0]
        img.save(out_path)
        print(f"    p{i:02d} saved → {out_path.name}")
    sys.stdout.flush()


def main():
    args = parse_args()
    device = args.device

    bf16_dir = OUTPUT_DIR / "eval_diverse_bf16"
    v6_dir   = OUTPUT_DIR / "eval_diverse_v6"

    # ------------------------------------------------------------------ #
    # 1. Generate images
    # ------------------------------------------------------------------ #
    if not args.skip_gen:
        print("=== [1/3] BF16 generation ===")
        pipe_bf16 = FluxPipeline.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, local_files_only=True).to(device)
        generate_images(pipe_bf16, DIVERSE_PROMPTS, bf16_dir, args.steps, args.seed, "BF16")
        del pipe_bf16; torch.cuda.empty_cache()

        print("\n=== [1/3] V6 generation ===")
        pipe_v6 = FluxPipeline.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, local_files_only=True).to(device)
        quantize_to_ternary(pipe_v6.transformer, per_channel=True,
                            lora_rank=args.rank, svd_init=False)
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
        state = {k: v for k, v in pipe_v6.transformer.named_parameters()}
        for n, t in ckpt.items():
            if n in state:
                state[n].data.copy_(t.to(torch.bfloat16))
        print(f"    Loaded {len(ckpt)} tensors from {args.ckpt}")
        generate_images(pipe_v6, DIVERSE_PROMPTS, v6_dir, args.steps, args.seed, "V6")
        del pipe_v6; torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 2. Score
    # ------------------------------------------------------------------ #
    print("\n=== [2/3] Scoring ===")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_LOCAL, local_files_only=True)
    clip_model     = CLIPModel.from_pretrained(CLIP_LOCAL, local_files_only=True).to(device)
    clip_model.eval()

    lpips_fn  = lpips_lib.LPIPS(net="alex").to(device)
    lpips_fn.eval()
    to_tensor = transforms.ToTensor()

    rows = []
    for i, prompt in enumerate(DIVERSE_PROMPTS):
        bf16_path = bf16_dir / f"p{i:02d}.png"
        v6_path   = v6_dir   / f"p{i:02d}.png"
        if not bf16_path.exists() or not v6_path.exists():
            print(f"  p{i:02d}: missing image, skipping")
            continue

        img_bf16 = Image.open(bf16_path).convert("RGB")
        img_v6   = Image.open(v6_path).convert("RGB")

        aes_bf16 = aesthetic_score(clip_model, clip_processor, img_bf16, device)
        aes_v6   = aesthetic_score(clip_model, clip_processor, img_v6,   device)
        cl_bf16  = clip_score_fn(clip_model, clip_processor, img_bf16, prompt, device)
        cl_v6    = clip_score_fn(clip_model, clip_processor, img_v6,   prompt, device)

        bf16_t = to_tensor(img_bf16).unsqueeze(0).to(device) * 2 - 1
        v6_t   = to_tensor(img_v6.resize((img_bf16.width, img_bf16.height))).unsqueeze(0).to(device) * 2 - 1
        with torch.no_grad():
            lp = lpips_fn(v6_t, bf16_t).item()

        short = prompt[:45] + "..." if len(prompt) > 45 else prompt
        print(f"  p{i:02d} {short}")
        print(f"       BF16: aes={aes_bf16:.2f} clip={cl_bf16:.4f}")
        print(f"       V6  : aes={aes_v6:.2f}   clip={cl_v6:.4f}  lpips={lp:.4f}")
        rows.append({
            "prompt": prompt, "idx": i,
            "bf16_aes": aes_bf16, "bf16_clip": cl_bf16,
            "v6_aes":   aes_v6,   "v6_clip":   cl_v6, "v6_lpips": lp,
        })

    # Averages
    avg_bf16_aes  = sum(r["bf16_aes"]  for r in rows) / len(rows)
    avg_bf16_clip = sum(r["bf16_clip"] for r in rows) / len(rows)
    avg_v6_aes    = sum(r["v6_aes"]    for r in rows) / len(rows)
    avg_v6_clip   = sum(r["v6_clip"]   for r in rows) / len(rows)
    avg_v6_lpips  = sum(r["v6_lpips"]  for r in rows) / len(rows)

    print(f"\n{'='*60}")
    print(f"  AVERAGE ({len(rows)} prompts)")
    print(f"  BF16: aes={avg_bf16_aes:.3f}  clip={avg_bf16_clip:.4f}")
    print(f"  V6  : aes={avg_v6_aes:.3f}  clip={avg_v6_clip:.4f}  lpips={avg_v6_lpips:.4f}")
    print(f"  Aes gap  : {avg_v6_aes - avg_bf16_aes:+.3f}")
    print(f"  CLIP gap : {avg_v6_clip - avg_bf16_clip:+.4f} ({(avg_v6_clip/avg_bf16_clip-1)*100:+.1f}%)")
    print(f"{'='*60}")

    # Save scores
    summary = {
        "n_prompts": len(rows),
        "bf16": {"avg_aes": avg_bf16_aes, "avg_clip": avg_bf16_clip},
        "v6":   {"avg_aes": avg_v6_aes, "avg_clip": avg_v6_clip, "avg_lpips": avg_v6_lpips},
        "per_prompt": rows,
    }
    out_json = OUTPUT_DIR / "diverse_scores.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_json}")

    # Text table
    table = [f"{'#':<4} {'Prompt':<46} {'BF16_aes':>8} {'BF16_clip':>9} {'V6_aes':>6} {'V6_clip':>8} {'V6_lpips':>9}"]
    table.append("-" * 95)
    for r in rows:
        short = r["prompt"][:44]
        table.append(f"{r['idx']:<4} {short:<46} {r['bf16_aes']:>8.3f} {r['bf16_clip']:>9.4f} "
                     f"{r['v6_aes']:>6.3f} {r['v6_clip']:>8.4f} {r['v6_lpips']:>9.4f}")
    table.append("-" * 95)
    table.append(f"{'AVG':<51} {avg_bf16_aes:>8.3f} {avg_bf16_clip:>9.4f} "
                 f"{avg_v6_aes:>6.3f} {avg_v6_clip:>8.4f} {avg_v6_lpips:>9.4f}")
    table_str = "\n".join(table)
    out_txt = OUTPUT_DIR / "diverse_scores.txt"
    out_txt.write_text(table_str)
    print(f"Saved: {out_txt}")

    # ------------------------------------------------------------------ #
    # 3. Visual grid (5 representative prompts)
    # ------------------------------------------------------------------ #
    print("\n=== [3/3] Building visual grid ===")
    sample_indices = [0, 3, 6, 9, 12, 15, 17, 19]  # diverse sample
    grid_rows = []
    for i in sample_indices:
        r = next((x for x in rows if x["idx"] == i), None)
        bf16_p = bf16_dir / f"p{i:02d}.png"
        v6_p   = v6_dir   / f"p{i:02d}.png"
        if not bf16_p.exists() or not v6_p.exists():
            continue

        bf16_img = Image.open(bf16_p).convert("RGB").resize((THUMB, THUMB), Image.LANCZOS)
        v6_img   = Image.open(v6_p).convert("RGB").resize((THUMB, THUMB), Image.LANCZOS)

        bf16_ann = annotate(bf16_img, [
            f"BF16  aes={r['bf16_aes']:.2f} clip={r['bf16_clip']:.4f}",
        ])
        v6_ann = annotate(v6_img, [
            f"V6    aes={r['v6_aes']:.2f}   clip={r['v6_clip']:.4f}  lpips={r['v6_lpips']:.3f}",
        ])

        row_h = max(bf16_ann.height, v6_ann.height)
        row   = Image.new("RGB", (THUMB * 2, row_h + FONT_SZ + 8), (15, 15, 15))
        row.paste(bf16_ann, (0, FONT_SZ + 8))
        row.paste(v6_ann,   (THUMB, FONT_SZ + 8))
        draw = ImageDraw.Draw(row)
        label = DIVERSE_PROMPTS[i][:70]
        draw.text((4, 4), label, fill=(255, 220, 80), font=load_font(FONT_SZ))
        grid_rows.append(row)

    if grid_rows:
        total_h = sum(r.height for r in grid_rows)
        grid = Image.new("RGB", (THUMB * 2, total_h), (10, 10, 10))
        y = 0
        for r in grid_rows:
            grid.paste(r, (0, y)); y += r.height
        out_grid = VIZ_DIR / "diverse_grid.png"
        grid.save(out_grid)
        print(f"Saved: {out_grid}  ({grid.width}×{grid.height}px)")

    print("\nDone.")


if __name__ == "__main__":
    main()
