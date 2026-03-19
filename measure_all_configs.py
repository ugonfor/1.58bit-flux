"""
Comprehensive memory + speed + quality measurement for all configurations:
  1. BF16 (baseline)
  2. Ternary INT8 + LoRA-r64 (current implementation)
  3. Ternary Packed 2-bit + LoRA-r64 (new packed storage)
  4. Ternary INT8 + LoRA-r128
  5. Ternary Packed 2-bit + LoRA-r128

Measures: static weight memory, inference peak VRAM, generation speed, CLIP quality.
"""
import os, gc, time, json, torch
os.environ.setdefault("HF_HOME", "/home/jovyan/.cache/huggingface")

from diffusers import FluxPipeline
from models.ternary import quantize_to_ternary
from models.ternary_packed import pack_model

MODEL_NAME = "black-forest-labs/FLUX.1-dev"
PROMPT = "A majestic lion resting on a savanna at golden hour"
RES = 1024
STEPS = 30
SEED = 42

CKPTS = {
    "r64": ("output/ternary_distilled_r64_res1024_s6000_fm_lpips1e-01.pt", 64),
    "r128": ("output/ternary_distilled_r128_res1024_s12000_fm_lpips1e-01.pt", 128),
}

results = []


def mem_gb():
    return torch.cuda.memory_allocated() / 1024**3


def peak_gb():
    return torch.cuda.max_memory_allocated() / 1024**3


def reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def load_ckpt(pipe, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=True)
    state = {k: v for k, v in pipe.transformer.named_parameters()}
    for name, tensor in ckpt.items():
        if name in state:
            state[name].data.copy_(tensor.to(torch.bfloat16))
    del ckpt
    torch.cuda.empty_cache()


def run_inference(pipe):
    gen = torch.Generator("cuda").manual_seed(SEED)
    with torch.no_grad():
        img = pipe(PROMPT, height=RES, width=RES, num_inference_steps=STEPS,
                   guidance_scale=3.5, generator=gen, output_type="pil").images[0]
    return img


def time_inference(pipe, n_runs=3):
    """Time inference, return avg seconds per run."""
    # Warmup
    run_inference(pipe)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_inference(pipe)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


# ── 1. BF16 Baseline ─────────────────────────────────────────────────────
print("=" * 70)
print("CONFIG: BF16 (no quantization)")
print("=" * 70)
reset()

pipe = FluxPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                    local_files_only=True).to("cuda")
static_bf16 = mem_gb()
print(f"  Static: {static_bf16:.2f} GB")

torch.cuda.reset_peak_memory_stats()
t_bf16 = time_inference(pipe, n_runs=3)
peak_bf16 = peak_gb()
print(f"  Peak:   {peak_bf16:.2f} GB")
print(f"  Speed:  {t_bf16:.2f} s/image")

# Save BF16 image for quality comparison
img_bf16 = run_inference(pipe)
img_bf16.save("output/measure_bf16.png")

del pipe
reset()
print()

results.append({
    "config": "BF16", "rank": "-", "packing": "bf16",
    "static_gb": round(static_bf16, 2), "peak_gb": round(peak_bf16, 2),
    "speed_s": round(t_bf16, 2),
})

# ── 2-5. Ternary configs ─────────────────────────────────────────────────
for ckpt_name, (ckpt_path, rank) in CKPTS.items():
    for use_packing in [False, True]:
        pack_label = "packed" if use_packing else "int8"
        config_name = f"Ternary-{ckpt_name}-{pack_label}"

        print("=" * 70)
        print(f"CONFIG: {config_name} (LoRA-r{rank}, {pack_label})")
        print("=" * 70)
        reset()

        pipe = FluxPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                            local_files_only=True).to("cuda")
        quantize_to_ternary(pipe.transformer, lora_rank=rank, svd_init=False)
        load_ckpt(pipe, ckpt_path)

        if use_packing:
            pack_model(pipe.transformer)

        static = mem_gb()
        print(f"  Static: {static:.2f} GB")

        # Count weight memory breakdown
        transformer = pipe.transformer
        buf_bytes = sum(b.numel() * b.element_size() for b in transformer.buffers())
        param_bytes = sum(p.numel() * p.element_size() for p in transformer.parameters())
        print(f"  Transformer: buffers={buf_bytes/1024**3:.2f}GB  params={param_bytes/1024**3:.2f}GB")

        torch.cuda.reset_peak_memory_stats()
        t = time_inference(pipe, n_runs=3)
        peak = peak_gb()
        print(f"  Peak:   {peak:.2f} GB")
        print(f"  Speed:  {t:.2f} s/image")

        # Save image
        img = run_inference(pipe)
        img.save(f"output/measure_{config_name.lower().replace('-','_')}.png")

        pct_peak = peak / peak_bf16 * 100
        speedup = t_bf16 / t
        print(f"  vs BF16: peak={pct_peak:.1f}%, speed={speedup:.2f}x")

        results.append({
            "config": config_name, "rank": rank, "packing": pack_label,
            "static_gb": round(static, 2), "peak_gb": round(peak, 2),
            "speed_s": round(t, 2),
            "peak_pct_bf16": round(pct_peak, 1),
            "speedup_vs_bf16": round(speedup, 2),
        })

        del pipe
        reset()
        print()


# ── Summary ───────────────────────────────────────────────────────────────
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Config':<30s} {'Static':>8s} {'Peak':>8s} {'Speed':>8s} {'Peak%':>7s} {'Speedup':>8s}")
for r in results:
    print(f"{r['config']:<30s} {r['static_gb']:>7.2f}G {r['peak_gb']:>7.2f}G {r['speed_s']:>7.2f}s "
          f"{r.get('peak_pct_bf16', 100):>6.1f}% {r.get('speedup_vs_bf16', 1.0):>7.2f}x")

# Save results
with open("output/measure_all_configs.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: output/measure_all_configs.json")
