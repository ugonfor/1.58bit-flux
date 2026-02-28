"""
Evaluate CLIP score of a ternary-distilled FLUX checkpoint.

Usage:
  python eval_ternary_clip.py --ckpt output/ternary_distilled_r64_res1024_s2000.pt \
                               --rank 64 --steps 30 --seed 42
"""
import os, argparse, torch
os.environ.setdefault("HF_HOME", "/home/jovyan/.cache/huggingface")

from pathlib import Path
from diffusers import FluxPipeline
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

from models.ternary import quantize_to_ternary

MODEL_NAME = "black-forest-labs/FLUX.1-dev"
CLIP_PATH  = ("/home/jovyan/.cache/huggingface/hub/"
              "models--openai--clip-vit-base-patch32/snapshots/"
              "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")

EVAL_PROMPTS = [
    "Cyberpunk samurai on a neon-lit rooftop at dusk, dramatic rim lighting, 32-bit render",
    "A fantasy landscape with mountains and a river",
    "Portrait of a young woman with wild curly hair in golden light",
    "Aerial view of a coastal city at sunset",
]


def clip_score(model, processor, img: Image.Image, prompt: str, device: str) -> float:
    inputs = processor(text=[prompt], images=[img], return_tensors="pt",
                       padding=True).to(device)
    with torch.no_grad():
        out = model(**inputs)
    return torch.cosine_similarity(out.image_embeds, out.text_embeds).item()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument("--rank",    type=int, default=64)
    p.add_argument("--steps",   type=int, default=30,   help="Inference denoising steps")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--save-dir", type=str, default=None,
                   help="Directory to save eval images (optional)")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda"
    dtype  = torch.bfloat16

    print(f"Loading pipeline...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, local_files_only=True,
    ).to(device)

    print(f"Quantizing to ternary (rank={args.rank})...")
    quantize_to_ternary(pipe.transformer, per_channel=True, lora_rank=args.rank,
                        svd_init=False)  # weights will be overwritten by checkpoint

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    missing, unexpected = [], []
    state = {k: v for k, v in pipe.transformer.named_parameters()}
    for name, tensor in ckpt.items():
        if name in state:
            state[name].data.copy_(tensor.to(dtype))
        else:
            unexpected.append(name)
    for name in state:
        if name.endswith((".scale", ".lora_A", ".lora_B")) and name not in ckpt:
            missing.append(name)
    if missing:
        print(f"  WARNING: {len(missing)} params not in checkpoint (using init values)")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys in checkpoint")
    print(f"  Loaded {len(ckpt)} tensors OK")

    print(f"\nLoading CLIP...")
    clip_proc  = CLIPProcessor.from_pretrained(CLIP_PATH, local_files_only=True)
    clip_model = CLIPModel.from_pretrained(CLIP_PATH, local_files_only=True).to(device).eval()

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    scores = []
    print(f"\nGenerating {len(EVAL_PROMPTS)} images ({args.steps} steps, seed={args.seed}):")
    for i, prompt in enumerate(EVAL_PROMPTS):
        gen = torch.Generator("cuda").manual_seed(args.seed)
        img = pipe(prompt, generator=gen,
                   num_inference_steps=args.steps, guidance_scale=3.5).images[0]
        if args.save_dir:
            img.save(Path(args.save_dir) / f"p{i}.png")
        sc = clip_score(clip_model, clip_proc, img, prompt, device)
        scores.append(sc)
        print(f"  [{i}] CLIP={sc:.4f}  {prompt[:60]}...")

    avg = sum(scores) / len(scores)
    print(f"\n  Average CLIP: {avg:.4f}  (BF16 reference: 0.322)")
    print(f"  Gap vs BF16: {(avg - 0.322) / 0.322 * 100:+.1f}%")


if __name__ == "__main__":
    main()
