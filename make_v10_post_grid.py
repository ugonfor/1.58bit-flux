"""
Generate comparison grids for post-013 (V10/V10b).
Creates:
  1. output/viz/v10_full_grid.png — all 20 prompts, BF16 | V9b | V10b
  2. output/viz/v10_highlights.png — 8 selected prompts (wins + regressions)
  3. output/viz/v10_convergence.png — V10 vs V10b convergence comparison
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("output")
VIZ_DIR = OUTPUT_DIR / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    ("Animals",      "Lion on savanna at golden hour"),
    ("Animals",      "Parrot on tropical branch"),
    ("Animals",      "Wolf howling at full moon"),
    ("Architecture", "Gothic cathedral interior"),
    ("Architecture", "Futuristic skyscraper"),
    ("Architecture", "Japanese pagoda + cherry blossoms"),
    ("Landscapes",   "Northern lights over frozen lake"),
    ("Landscapes",   "Volcanic eruption at night"),
    ("Landscapes",   "Lavender fields in Provence"),
    ("Portraits",    "Elderly fisherman portrait"),
    ("Portraits",    "Ballet dancer mid-leap"),
    ("Portraits",    "Street musician in rain"),
    ("Food",         "Sushi platter"),
    ("Food",         "Rustic bread on wooden table"),
    ("Fantasy",      "Dragon over medieval castle"),
    ("Sci-fi",       "Astronaut floating in space"),
    ("Fantasy",      "Magical forest with fireflies"),
    ("Art styles",   "Venice canals watercolor"),
    ("Art styles",   "Renaissance oil portrait"),
    ("Urban",        "Rainy Tokyo street at night"),
]

BF16_CLIPS = [0.3276,0.3220,0.3417,0.3134,0.3246,0.3510,0.3006,0.3456,0.3384,0.3845,0.3433,0.3677,0.3122,0.3433,0.3668,0.3365,0.3684,0.3179,0.2902,0.3534]
V9B_CLIPS  = [0.3128,0.3232,0.3340,0.2941,0.3000,0.3507,0.2873,0.3355,0.3262,0.3263,0.2616,0.3133,0.2181,0.2672,0.2976,0.3071,0.3404,0.2850,0.2918,0.3006]
V10_CLIPS  = [0.3159,0.3338,0.3239,0.3194,0.3054,0.3526,0.2428,0.3229,0.2951,0.3408,0.2644,0.3252,0.2643,0.2566,0.3595,0.1423,0.3626,0.2988,0.1756,0.3146]
V10B_CLIPS = [0.3255,0.3201,0.3301,0.3114,0.3049,0.3458,0.2915,0.3291,0.3068,0.3653,0.3095,0.3037,0.2343,0.2471,0.3065,0.1970,0.3822,0.2883,0.2860,0.3146]

DIRS = {
    "BF16": OUTPUT_DIR / "eval_diverse_bf16",
    "V9b":  OUTPUT_DIR / "eval_diverse_v9b",
    "V10b": OUTPUT_DIR / "eval_diverse_v10b",
}
MODELS = ["BF16", "V9b", "V10b"]
CLIPS  = {"BF16": BF16_CLIPS, "V9b": V9B_CLIPS, "V10b": V10B_CLIPS}

THUMB = 256
LABEL_H = 36
HEADER_H = 50
PAD = 4


def load_thumb(path, size=THUMB):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return img


def color_for_pct(pct):
    if pct >= 95:   return (40, 180, 40)
    elif pct >= 85: return (200, 160, 0)
    else:           return (200, 50, 50)


def make_score_bar(clip, bf16_clip, width, height):
    pct = clip / bf16_clip * 100
    color = color_for_pct(pct)
    img = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    bar_w = int(width * min(pct, 100) / 100)
    draw.rectangle([0, 0, bar_w, height], fill=color)
    text = f"{pct:.1f}%"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2), text, fill=(255, 255, 255), font=font)
    return img


# ── 1. Full grid ──────────────────────────────────────────────────────────
def make_full_grid():
    n_prompts = len(PROMPTS)
    n_models  = len(MODELS)
    col_w = THUMB + PAD
    row_h = THUMB + LABEL_H + PAD
    total_w = HEADER_H + n_models * col_w + PAD
    total_h = n_prompts * row_h + HEADER_H

    canvas = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw   = ImageDraw.Draw(canvas)
    try:
        hfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        pfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        hfont = pfont = ImageFont.load_default()

    header_colors = {"BF16": (60, 100, 180), "V9b": (80, 140, 80), "V10b": (160, 60, 160)}
    for mi, m in enumerate(MODELS):
        x = HEADER_H + mi * col_w
        draw.rectangle([x, 0, x + THUMB, HEADER_H - 2], fill=header_colors[m])
        bbox = draw.textbbox((0, 0), m, font=hfont)
        tw = bbox[2] - bbox[0]
        draw.text((x + (THUMB - tw) // 2, 12), m, fill=(255, 255, 255), font=hfont)

    for pi, (cat, label) in enumerate(PROMPTS):
        y_base = HEADER_H + pi * row_h
        draw.rectangle([0, y_base, HEADER_H - 2, y_base + row_h - PAD], fill=(35, 35, 55))
        draw.text((2, y_base + THUMB // 2 - 6), f"p{pi:02d}", fill=(180, 180, 180), font=pfont)
        for mi, m in enumerate(MODELS):
            x = HEADER_H + mi * col_w
            thumb = load_thumb(DIRS[m] / f"p{pi:02d}.png")
            canvas.paste(thumb, (x, y_base))
            bar = make_score_bar(CLIPS[m][pi], BF16_CLIPS[pi], THUMB, LABEL_H - PAD)
            canvas.paste(bar, (x, y_base + THUMB))

    out = VIZ_DIR / "v10_full_grid.png"
    canvas.save(out)
    print(f"Saved: {out}  ({canvas.width}×{canvas.height}px)")


# ── 2. Highlights grid ────────────────────────────────────────────────────
HIGHLIGHT_INDICES = [
    16,  # Magic forest — exceeds BF16 (103.7%)
    9,   # Fisherman — big win (+39.0)
    10,  # Ballet — huge improvement (+47.9)
    3,   # Cathedral — recovered vs V9b
    12,  # Sushi — still weak but improving
    15,  # Astronaut — catastrophic failure
    5,   # Pagoda — near-BF16
    14,  # Dragon — solid improvement
]

def make_highlights_grid():
    n_sel = len(HIGHLIGHT_INDICES)
    col_w = THUMB + PAD
    row_h = THUMB + LABEL_H + PAD
    LABEL_W = 180
    total_w = LABEL_W + len(MODELS) * col_w + PAD
    total_h = n_sel * row_h + HEADER_H + 10

    canvas = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw   = ImageDraw.Draw(canvas)
    try:
        hfont  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        pfont  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        pfontb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except Exception:
        hfont = pfont = pfontb = ImageFont.load_default()

    header_colors = {"BF16": (60, 100, 180), "V9b": (80, 140, 80), "V10b": (160, 60, 160)}
    for mi, m in enumerate(MODELS):
        x = LABEL_W + mi * col_w
        draw.rectangle([x, 0, x + THUMB, HEADER_H - 2], fill=header_colors[m])
        bbox = draw.textbbox((0, 0), m, font=hfont)
        tw = bbox[2] - bbox[0]
        draw.text((x + (THUMB - tw) // 2, 14), m, fill=(255, 255, 255), font=hfont)

    for si, pi in enumerate(HIGHLIGHT_INDICES):
        cat, label = PROMPTS[pi]
        y_base = HEADER_H + 10 + si * row_h
        draw.rectangle([0, y_base, LABEL_W - 2, y_base + row_h - PAD], fill=(30, 30, 48))
        draw.text((6, y_base + 6), f"p{pi:02d} [{cat}]", fill=(160, 180, 220), font=pfontb)
        words = label.split()
        line, lines = "", []
        for w in words:
            test = (line + " " + w).strip()
            if len(test) > 20:
                if line: lines.append(line)
                line = w
            else:
                line = test
        if line: lines.append(line)
        for li, l in enumerate(lines[:4]):
            draw.text((6, y_base + 22 + li * 14), l, fill=(200, 200, 200), font=pfont)

        delta = V10B_CLIPS[pi] - V9B_CLIPS[pi]
        dcol = (60, 200, 60) if delta >= 0 else (220, 60, 60)
        dsign = "+" if delta >= 0 else ""
        draw.text((6, y_base + THUMB - 20), f"V10b vs V9b: {dsign}{delta*1000:.1f}", fill=dcol, font=pfontb)

        for mi, m in enumerate(MODELS):
            x = LABEL_W + mi * col_w
            thumb = load_thumb(DIRS[m] / f"p{pi:02d}.png")
            canvas.paste(thumb, (x, y_base))
            bar = make_score_bar(CLIPS[m][pi], BF16_CLIPS[pi], THUMB, LABEL_H - PAD)
            canvas.paste(bar, (x, y_base + THUMB))

    out = VIZ_DIR / "v10_highlights.png"
    canvas.save(out)
    print(f"Saved: {out}  ({canvas.width}×{canvas.height}px)")


# ── 3. Convergence plot: V10 (6k cold) vs V10b (12k warm) per-prompt ────
def make_convergence_plot():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#12121f")

    x = np.arange(len(PROMPTS))
    bf16_pct = [100.0] * 20
    v9b_pct  = [v / b * 100 for v, b in zip(V9B_CLIPS, BF16_CLIPS)]
    v10_pct  = [v / b * 100 for v, b in zip(V10_CLIPS, BF16_CLIPS)]
    v10b_pct = [v / b * 100 for v, b in zip(V10B_CLIPS, BF16_CLIPS)]

    w = 0.22
    bars_v9b  = ax.bar(x - w, v9b_pct, w, label="V9b (r64, 6k)", color="#55aa55", alpha=0.8)
    bars_v10  = ax.bar(x,     v10_pct, w, label="V10 (r128, 6k cold)", color="#cc6666", alpha=0.8)
    bars_v10b = ax.bar(x + w, v10b_pct, w, label="V10b (r128, 12k)", color="#aa55cc", alpha=0.8)

    ax.axhline(100, color="#ff6666", linewidth=1, linestyle=":", alpha=0.6, label="BF16 = 100%")
    ax.axhline(90, color="#888888", linewidth=0.5, linestyle="--", alpha=0.4)

    labels = [f"p{i:02d}" for i in range(20)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, color="#aaaacc")
    ax.set_ylabel("OOD CLIP (% of BF16)", color="#aaaacc", fontsize=10)
    ax.set_title("Rank-128 Convergence: 6k Cold → 12k Warm-Start", color="#ddddff", fontsize=12, pad=10)
    ax.tick_params(colors="#aaaacc")
    ax.spines[:].set_color("#444466")
    ax.set_ylim(30, 115)
    ax.legend(framealpha=0.3, facecolor="#22223a", edgecolor="#666688",
              labelcolor="#ccccee", fontsize=8, loc="lower left")
    plt.tight_layout()

    out = VIZ_DIR / "v10_convergence.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Building V10 post grids...")
    make_full_grid()
    make_highlights_grid()
    make_convergence_plot()
    print("Done.")
