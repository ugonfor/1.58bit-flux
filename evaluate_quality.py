"""
Quality evaluation: PSNR vs BF16 reference + CLIP score (text-image alignment).
"""
import os, torch, numpy as np
os.environ.setdefault("HF_HOME", "/home/jovyan/.cache/huggingface")

from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

SAMPLES = Path("output/samples")
CLIP_PATH = "/home/jovyan/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

PROMPTS = [
    "Cyberpunk samurai on a neon-lit rooftop at dusk, dramatic rim lighting, 32-bit render",
    "A fantasy landscape with mountains and a river",
]

CONFIGS = [
    ("bf16",    SAMPLES / "bf16"),
    ("w8r0",    SAMPLES / "w8"  / "rank0"),
    ("w4r0",    SAMPLES / "w4"  / "rank0"),
    ("w4r64",   SAMPLES / "w4"  / "rank64"),
    ("w3r0",    SAMPLES / "w3"  / "rank0"),
    ("w2r0",    SAMPLES / "w2"  / "rank0"),
    ("w2r64",   SAMPLES / "w2"  / "rank64"),
    ("ternary", SAMPLES / "ternary"),
]


def load_np(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)


def psnr(ref: np.ndarray, img: np.ndarray) -> float:
    mse = np.mean((ref - img) ** 2)
    return float(20 * np.log10(255.0 / np.sqrt(mse))) if mse > 1e-10 else float("inf")


def clip_score(model, processor, img_arr: np.ndarray, prompt: str, device: str) -> float:
    img_pil = Image.fromarray(img_arr.astype(np.uint8))
    inputs = processor(text=[prompt], images=[img_pil], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(**inputs)
    return torch.cosine_similarity(out.image_embeds, out.text_embeds).item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading CLIP (ViT-B/32)...")
    processor = CLIPProcessor.from_pretrained(CLIP_PATH, local_files_only=True)
    model = CLIPModel.from_pretrained(CLIP_PATH, local_files_only=True).to(device).eval()

    bf16_imgs = [load_np(sorted((SAMPLES / "bf16").glob("*.png"))[i]) for i in range(len(PROMPTS))]
    # BF16 CLIP scores as reference
    bf16_clips = [clip_score(model, processor, bf16_imgs[i], PROMPTS[i], device) for i in range(len(PROMPTS))]

    hdr = f"{'Config':10s}  {'PSNR-0':>7s}  {'PSNR-1':>7s}  {'PSNR':>7s}  {'CLIP-0':>7s}  {'CLIP-1':>7s}  {'CLIP':>7s}"
    print("\n" + hdr)
    print("-" * len(hdr))

    for tag, dirpath in CONFIGS:
        pngs = sorted(dirpath.glob("*.png"))
        if len(pngs) < len(PROMPTS):
            print(f"{tag:10s}  (missing)")
            continue

        imgs = [load_np(pngs[i]) for i in range(len(PROMPTS))]
        psnrs = [psnr(bf16_imgs[i], imgs[i]) for i in range(len(PROMPTS))]
        clips = [clip_score(model, processor, imgs[i], PROMPTS[i], device) for i in range(len(PROMPTS))]

        def fp(v): return f"{v:7.2f}" if v != float("inf") else "    inf"
        print(f"{tag:10s}  {fp(psnrs[0])}  {fp(psnrs[1])}  {fp(sum(psnrs)/2)}  "
              f"{clips[0]:7.4f}  {clips[1]:7.4f}  {sum(clips)/2:7.4f}")

    print(f"\n(BF16 CLIP reference: {bf16_clips[0]:.4f} / {bf16_clips[1]:.4f})")


if __name__ == "__main__":
    main()
