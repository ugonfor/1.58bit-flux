"""
Diverse prompt evaluation: BF16 vs any number of student models on 20 unseen prompts.

Generates images, then computes:
  - CLIP score (text-image alignment)
  - Aesthetic score (zero-shot CLIP proxy, 0-10)
  - LPIPS vs BF16 (perceptual distance, lower = closer to BF16)

Usage:
  # Single model (default ckpt)
  python eval_diverse.py

  # Compare V6 and V7
  python eval_diverse.py --models V6:output/ternary_distilled_..._v6.pt V7:output/ternary_distilled_..._v7.pt

  # Skip generation (re-score existing images)
  python eval_diverse.py --models V6:... V7:... --skip-gen
"""
import os, sys, json, argparse
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

# 20 diverse prompts — none overlap with the 4 training eval prompts
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

PROMPT_CATEGORIES = [
    "Animals", "Animals", "Animals",
    "Architecture", "Architecture", "Architecture",
    "Landscapes", "Landscapes", "Landscapes",
    "Portraits", "Portraits", "Portraits",
    "Food", "Food",
    "Fantasy", "Sci-fi", "Fantasy",
    "Art styles", "Art styles",
    "Urban",
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

THUMB   = 512
FONT_SZ = 18


def load_font(size=FONT_SZ):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def annotate(img, lines, bg=(20, 20, 20)):
    bar_h = (FONT_SZ + 5) * len(lines) + 6
    out   = Image.new("RGB", (img.width, img.height + bar_h), bg)
    out.paste(img, (0, 0))
    draw  = ImageDraw.Draw(out)
    font  = load_font(FONT_SZ)
    y     = img.height + 3
    for line in lines:
        draw.text((4, y), line, fill=(220, 220, 220), font=font)
        y += FONT_SZ + 5
    return out


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
    i_feat = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    t_feat = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
    return (i_feat * t_feat).sum().item()


def load_student_pipe(ckpt_path, rank, device):
    pipe = FluxPipeline.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, local_files_only=True).to(device)
    quantize_to_ternary(pipe.transformer, per_channel=True,
                        lora_rank=rank, svd_init=False)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = {k: v for k, v in pipe.transformer.named_parameters()}
    loaded = 0
    for n, t in ckpt.items():
        if n in state:
            state[n].data.copy_(t.to(torch.bfloat16))
            loaded += 1
    print(f"    Loaded {loaded} tensors from {ckpt_path}")
    return pipe


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
        print(f"    p{i:02d} saved")
    sys.stdout.flush()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["V6:output/ternary_distilled_r64_res1024_s2000_fm_lpips1e-01.pt"],
                   help="label:ckpt_path pairs for student models to evaluate")
    p.add_argument("--rank",     type=int, default=64)
    p.add_argument("--steps",    type=int, default=28)
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-gen", action="store_true",
                   help="Skip image generation (re-score existing files)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device

    student_models = []
    for item in args.models:
        parts = item.split(":")
        label, ckpt_path = parts[0], parts[1]
        rank = int(parts[2]) if len(parts) > 2 else args.rank
        student_models.append((label, ckpt_path, rank))

    bf16_dir = OUTPUT_DIR / "eval_diverse_bf16"

    def _all_images_exist(d: Path) -> bool:
        return all((d / f"p{i:02d}.png").exists() for i in range(len(DIVERSE_PROMPTS)))

    # ------------------------------------------------------------------ #
    # 1. Generate BF16 reference images
    # ------------------------------------------------------------------ #
    if not args.skip_gen:
        print("=== [1/3] BF16 reference generation ===")
        if _all_images_exist(bf16_dir):
            print(f"  All BF16 images already exist in {bf16_dir}, skipping pipe load.")
        else:
            pipe_bf16 = FluxPipeline.from_pretrained(
                MODEL_NAME, torch_dtype=torch.bfloat16, local_files_only=True).to(device)
            generate_images(pipe_bf16, DIVERSE_PROMPTS, bf16_dir,
                            args.steps, args.seed, "BF16")
            del pipe_bf16; torch.cuda.empty_cache()

        for label, ckpt_path, rank in student_models:
            img_dir = OUTPUT_DIR / f"eval_diverse_{label.lower()}"
            if _all_images_exist(img_dir):
                print(f"\n  All {label} images already exist in {img_dir}, skipping pipe load.")
                continue
            print(f"\n=== [1/3] {label} generation ===")
            pipe_s = load_student_pipe(ckpt_path, rank, device)
            generate_images(pipe_s, DIVERSE_PROMPTS, img_dir,
                            args.steps, args.seed, label)
            del pipe_s; torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 2. Score all models
    # ------------------------------------------------------------------ #
    print("\n=== [2/3] Scoring ===")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_LOCAL, local_files_only=True)
    clip_model     = CLIPModel.from_pretrained(CLIP_LOCAL, local_files_only=True).to(device)
    clip_model.eval()

    lpips_fn  = lpips_lib.LPIPS(net="alex").to(device)
    lpips_fn.eval()
    to_tensor = transforms.ToTensor()

    # Collect per-prompt scores for every model
    all_results = {"BF16": []}
    for label, *_ in student_models:
        all_results[label] = []

    for i, prompt in enumerate(DIVERSE_PROMPTS):
        bf16_path = bf16_dir / f"p{i:02d}.png"
        if not bf16_path.exists():
            print(f"  p{i:02d}: missing BF16 image, skipping"); continue

        img_bf16  = Image.open(bf16_path).convert("RGB")
        aes_bf16  = aesthetic_score(clip_model, clip_processor, img_bf16, device)
        cl_bf16   = clip_score_fn(clip_model, clip_processor, img_bf16, prompt, device)
        bf16_t    = to_tensor(img_bf16).unsqueeze(0).to(device) * 2 - 1
        all_results["BF16"].append(
            {"idx": i, "prompt": prompt, "aes": aes_bf16, "clip": cl_bf16, "lpips": None})

        for label, *_ in student_models:
            img_dir = OUTPUT_DIR / f"eval_diverse_{label.lower()}"
            s_path  = img_dir / f"p{i:02d}.png"
            if not s_path.exists():
                print(f"  p{i:02d}: missing {label} image, skipping")
                all_results[label].append(None); continue

            img_s  = Image.open(s_path).convert("RGB")
            aes_s  = aesthetic_score(clip_model, clip_processor, img_s, device)
            cl_s   = clip_score_fn(clip_model, clip_processor, img_s, prompt, device)
            s_t    = to_tensor(img_s.resize((img_bf16.width, img_bf16.height))).unsqueeze(0).to(device) * 2 - 1
            with torch.no_grad():
                lp = lpips_fn(s_t, bf16_t).item()
            all_results[label].append(
                {"idx": i, "prompt": prompt, "aes": aes_s, "clip": cl_s, "lpips": lp})

        cat = PROMPT_CATEGORIES[i]
        print(f"  p{i:02d} [{cat:12s}] {prompt[:50]}")
        print(f"    BF16: aes={aes_bf16:.2f} clip={cl_bf16:.4f}")
        for label, *_ in student_models:
            r = all_results[label][-1]
            if r:
                print(f"    {label:<6}: aes={r['aes']:.2f} clip={r['clip']:.4f} lpips={r['lpips']:.4f}")

    # Averages and per-category breakdown
    print(f"\n{'='*70}")
    print(f"  AVERAGES ({len(DIVERSE_PROMPTS)} prompts)")
    averages = {}
    for label in ["BF16"] + [m[0] for m in student_models]:
        valid = [r for r in all_results[label] if r is not None]
        if not valid: continue
        avg_aes  = sum(r["aes"]  for r in valid) / len(valid)
        avg_clip = sum(r["clip"] for r in valid) / len(valid)
        lp_vals  = [r["lpips"] for r in valid if r["lpips"] is not None]
        avg_lp   = sum(lp_vals) / len(lp_vals) if lp_vals else None
        averages[label] = {"avg_aes": avg_aes, "avg_clip": avg_clip, "avg_lpips": avg_lp}
        lp_str = f"  lpips={avg_lp:.4f}" if avg_lp is not None else ""
        print(f"  {label:<6}: aes={avg_aes:.3f}  clip={avg_clip:.4f}{lp_str}")

    bf16_avg_clip = averages["BF16"]["avg_clip"]
    for label, *_ in student_models:
        if label in averages:
            pct = averages[label]["avg_clip"] / bf16_avg_clip * 100
            print(f"  {label} / BF16 CLIP: {pct:.1f}%")
    print(f"{'='*70}")

    # ------------------------------------------------------------------ #
    # 3. Text table + JSON
    # ------------------------------------------------------------------ #
    labels_ordered = ["BF16"] + [m[0] for m in student_models]

    header = f"{'#':<3} {'Cat':<12} {'Prompt':<44}"
    for lb in labels_ordered:
        header += f"  {lb+'_aes':>8} {lb+'_clip':>9}"
        if lb != "BF16":
            header += f" {lb+'_lpips':>9}"
    table  = [header, "-" * len(header)]

    for i in range(len(DIVERSE_PROMPTS)):
        bf16_r = all_results["BF16"][i] if i < len(all_results["BF16"]) else None
        if bf16_r is None: continue
        cat   = PROMPT_CATEGORIES[i]
        short = DIVERSE_PROMPTS[i][:43]
        row   = f"{i:<3} {cat:<12} {short:<44}"
        for lb in labels_ordered:
            r = all_results[lb][i] if i < len(all_results[lb]) else None
            if r is None:
                row += f"  {'N/A':>8} {'N/A':>9}"
                if lb != "BF16": row += f" {'N/A':>9}"
            else:
                row += f"  {r['aes']:>8.3f} {r['clip']:>9.4f}"
                if lb != "BF16":
                    row += f" {r['lpips']:>9.4f}"
        table.append(row)

    table.append("-" * len(header))
    avg_row = f"{'AVG':<3} {'':<12} {'':<44}"
    for lb in labels_ordered:
        if lb in averages:
            avg_row += f"  {averages[lb]['avg_aes']:>8.3f} {averages[lb]['avg_clip']:>9.4f}"
            if lb != "BF16":
                lp = averages[lb]["avg_lpips"]
                avg_row += f" {lp:>9.4f}" if lp is not None else f" {'ref':>9}"
        else:
            avg_row += f"  {'N/A':>8} {'N/A':>9}"
    table.append(avg_row)
    table_str = "\n".join(table)

    out_txt = OUTPUT_DIR / "diverse_scores.txt"
    out_txt.write_text(table_str)
    print(f"\nSaved: {out_txt}")

    out_json = OUTPUT_DIR / "diverse_scores.json"
    out_json.write_text(json.dumps(
        {"averages": averages, "per_prompt": {lb: all_results[lb] for lb in labels_ordered}},
        indent=2))
    print(f"Saved: {out_json}")

    # ------------------------------------------------------------------ #
    # 4. Visual comparison grid (8 representative prompts)
    # ------------------------------------------------------------------ #
    print("\n=== [3/3] Building visual grid ===")
    n_cols       = 1 + len(student_models)   # BF16 + students
    sample_idx   = [0, 3, 6, 9, 12, 15, 17, 19]
    all_dirs     = [("BF16", bf16_dir)] + [
        (lb, OUTPUT_DIR / f"eval_diverse_{lb.lower()}") for lb, *_ in student_models]
    grid_rows    = []

    for i in sample_idx:
        cells = []
        for lb, d in all_dirs:
            p = d / f"p{i:02d}.png"
            img = (Image.open(p).convert("RGB").resize((THUMB, THUMB), Image.LANCZOS)
                   if p.exists() else Image.new("RGB", (THUMB, THUMB), (60, 60, 60)))
            r = all_results[lb][i] if i < len(all_results[lb]) else None
            if r:
                lp_str = f" lp={r['lpips']:.3f}" if r["lpips"] is not None else ""
                ann = annotate(img, [f"{lb} a={r['aes']:.2f} c={r['clip']:.4f}{lp_str}"])
            else:
                ann = annotate(img, [f"{lb} N/A"])
            cells.append(ann)

        row_h = max(c.height for c in cells)
        row_w = THUMB * n_cols
        row   = Image.new("RGB", (row_w, row_h + FONT_SZ + 6), (15, 15, 15))
        for ci, c in enumerate(cells):
            row.paste(c, (ci * THUMB, FONT_SZ + 6))
        draw  = ImageDraw.Draw(row)
        cat   = PROMPT_CATEGORIES[i]
        label = f"[{cat}] {DIVERSE_PROMPTS[i][:80]}"
        draw.text((4, 2), label, fill=(255, 220, 80), font=load_font(FONT_SZ - 2))
        grid_rows.append(row)

    if grid_rows:
        total_h = sum(r.height for r in grid_rows)
        grid    = Image.new("RGB", (THUMB * n_cols, total_h), (10, 10, 10))
        y = 0
        for r in grid_rows:
            grid.paste(r, (0, y)); y += r.height
        out_grid = VIZ_DIR / "diverse_grid.png"
        grid.save(out_grid)
        print(f"Saved grid: {out_grid}  ({grid.width}×{grid.height}px)")

    print("\nDone.")


if __name__ == "__main__":
    main()
