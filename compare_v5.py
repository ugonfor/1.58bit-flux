"""
Compare BF16 vs V2 vs V5 eval images for post-005.
Run after V5 CLIP eval completes.

Usage:
  python compare_v5.py
  → saves output/viz/v5_comparison.png
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path("output")
VIZ_DIR = OUTPUT_DIR / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

THUMB = 512
FONT_SIZE = 24

PROMPT_LABELS = [
    "Cyberpunk samurai",
    "Fantasy landscape",
    "Portrait (curly hair)",
    "Aerial coastal city",
]

COLS = [
    ("BF16 (reference)",   OUTPUT_DIR / "bf16_reference"),
    ("V2 (200 imgs, 51 prompts)", OUTPUT_DIR / "eval_fm_clip_v2"),
    ("V5 (548 imgs, 174 prompts)", OUTPUT_DIR / "eval_fm_clip_v5"),
]


def try_load(path: Path) -> Image.Image | None:
    if path.exists():
        return Image.open(path).convert("RGB").resize((THUMB, THUMB), Image.LANCZOS)
    return None


def placeholder(text: str = "N/A") -> Image.Image:
    img = Image.new("RGB", (THUMB, THUMB), (60, 60, 60))
    draw = ImageDraw.Draw(img)
    draw.text((THUMB // 2 - 20, THUMB // 2 - 12), text, fill=(200, 200, 200))
    return img


def labeled(img: Image.Image, text: str) -> Image.Image:
    bar_h = FONT_SIZE + 12
    out = Image.new("RGB", (img.width, img.height + bar_h), (20, 20, 20))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    tx = (img.width - tw) // 2
    draw.text((tx, img.height + 6), text, fill=(240, 240, 240), font=font)
    return out


def header_bar(text: str, width: int) -> Image.Image:
    h = FONT_SIZE + 16
    bar = Image.new("RGB", (width, h), (40, 40, 80))
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((width - tw) // 2, 8), text, fill=(220, 220, 255), font=font)
    return bar


def main():
    # Grid: rows = prompts, cols = models
    n_rows = len(PROMPT_LABELS)
    n_cols = len(COLS)

    rows = []
    for row_i, prompt_label in enumerate(PROMPT_LABELS):
        cells = []
        for col_name, col_dir in COLS:
            img_path = col_dir / f"p{row_i}.png"
            img = try_load(img_path) or placeholder("missing")
            img = labeled(img, f"p{row_i}: {prompt_label[:20]}")
            cells.append(img)

        # Hstack cells in this row
        row_img = Image.new("RGB", (sum(c.width for c in cells), cells[0].height), (20, 20, 20))
        x = 0
        for c in cells:
            row_img.paste(c, (x, 0))
            x += c.width
        rows.append(row_img)

    # Add column headers at top
    cell_w = THUMB
    total_w = cell_w * n_cols
    headers = []
    for col_name, _ in COLS:
        h = header_bar(col_name, cell_w)
        headers.append(h)
    header_row = Image.new("RGB", (total_w, headers[0].height), (20, 20, 20))
    x = 0
    for h in headers:
        header_row.paste(h, (x, 0))
        x += h.width

    all_rows = [header_row] + rows
    total_h = sum(r.height for r in all_rows)
    grid = Image.new("RGB", (total_w, total_h), (10, 10, 10))
    y = 0
    for r in all_rows:
        grid.paste(r, (0, y))
        y += r.height

    out_path = VIZ_DIR / "v5_comparison.png"
    grid.save(out_path)
    print(f"Saved: {out_path}  ({grid.width}×{grid.height}px)")

    # Also try to print available CLIP scores
    clip_v2_file = OUTPUT_DIR / "eval_fm_clip_v2.log"
    clip_v5_file = OUTPUT_DIR / "eval_fm_clip_v5.log"
    for label, fpath in [("V2", clip_v2_file), ("V5", clip_v5_file)]:
        if fpath.exists():
            lines = fpath.read_text().splitlines()
            scores = [l for l in lines if "CLIP=" in l or "Average" in l]
            print(f"\n{label} CLIP scores:")
            for s in scores[-6:]:
                print(f"  {s.strip()}")


if __name__ == "__main__":
    main()
