from models.transformer_flux import (
    FluxTransformer2DModel as FluxTransformer2DModelQuant,
)
from diffusers import FluxPipeline, DiffusionPipeline
import torch

from pathlib import Path
from tqdm import tqdm

import torch.nn as nn
import logging
import glob

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger(__name__)


def count_parameters(model):
    from models.utils_quant import QuantizeLinear

    total_params = sum(p.numel() for p in model.parameters())
    linear_params = 0
    quantize_linear_params = 0

    for name, module in model.named_modules():
        # First check for QuantizeLinear specifically
        if isinstance(module, QuantizeLinear):
            quantize_linear_params += sum(p.numel() for p in module.parameters())
        # Then check for regular nn.Linear (but not QuantizeLinear)
        elif isinstance(module, nn.Linear):
            linear_params += sum(p.numel() for p in module.parameters())

    linear_ratio = linear_params / total_params if total_params > 0 else 0
    quantize_ratio = quantize_linear_params / total_params if total_params > 0 else 0
    total_linear_params = linear_params + quantize_linear_params
    quantize_linear_ratio = (
        quantize_linear_params / total_linear_params if total_linear_params > 0 else 0
    )

    print(f"Total parameters: {total_params}")
    print(f"Regular Linear layer parameters: {linear_params}")
    print(f"QuantizeLinear parameters: {quantize_linear_params}")
    print(f"Regular Linear layer parameter ratio: {linear_ratio:.4f}")
    print(f"QuantizeLinear parameter ratio (vs total): {quantize_ratio:.4f}")
    print(
        f"QuantizeLinear parameter ratio (vs all linear): {quantize_linear_ratio:.4f}"
    )

    return {
        "total": total_params,
        "linear": linear_params,
        "quantize_linear": quantize_linear_params,
        "linear_ratio": linear_ratio,
        "quantize_ratio": quantize_ratio,
        "quantize_linear_ratio": quantize_linear_ratio,
    }


def calculate_scale_and_zero_point(args):
    """병렬 처리를 위한 헬퍼 함수"""
    name, weight_param, w_bits = args

    # Calculate scale
    xmax, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
    maxq = 2 ** (w_bits - 1) - 1
    scale = xmax / maxq

    # Calculate zero_point
    zero_point_name = name.replace("weight_scale", "weight_zero_point")
    xmin, _ = torch.min(weight_param, dim=-1, keepdim=True)
    xmax, _ = torch.max(weight_param, dim=-1, keepdim=True)
    qmin = 0
    qmax = 2**w_bits - 1

    # Calculate zero point: zero_point = qmin - round(xmin / scale)
    zero_point = qmin - torch.round(xmin / scale)
    zero_point = torch.clamp(zero_point, qmin, qmax)

    return name, scale, zero_point_name, zero_point


def initialize_low_rank_svd_task(args):
    """병렬 처리를 위한 SVD 초기화 헬퍼 함수"""
    low_rank_A_name, original_weight, quantized_weight, low_rank_dim = args

    with torch.no_grad():
        # Calculate residual
        residual = original_weight - quantized_weight

        # Convert to float32 for SVD computation if needed
        original_dtype = residual.dtype
        if residual.dtype == torch.bfloat16:
            residual = residual.float()

        # SVD decomposition of residual
        U, S, Vt = torch.linalg.svd(residual, full_matrices=False)

        # Take top-k components
        k = min(low_rank_dim, S.shape[0])
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]

        # Initialize A and B such that A @ B ≈ residual
        # A = U_k @ sqrt(S_k), B = sqrt(S_k) @ Vt_k
        sqrt_S_k = torch.sqrt(S_k).unsqueeze(0)
        A_init = U_k * sqrt_S_k
        B_init = sqrt_S_k.T * Vt_k

        # Pad with zeros if low_rank_dim > k
        if low_rank_dim > k:
            A_pad = torch.zeros(
                original_weight.shape[0],
                low_rank_dim - k,
                device=A_init.device,
                dtype=A_init.dtype,
            )
            B_pad = torch.zeros(
                low_rank_dim - k,
                original_weight.shape[1],
                device=B_init.device,
                dtype=B_init.dtype,
            )
            A_init = torch.cat([A_init, A_pad], dim=1)
            B_init = torch.cat([B_init, B_pad], dim=0)

        # Convert back to original dtype
        if original_dtype == torch.bfloat16:
            A_init = A_init.to(original_dtype)
            B_init = B_init.to(original_dtype)

        low_rank_B_name = low_rank_A_name.replace("low_rank_A", "low_rank_B")
        return low_rank_A_name, A_init, low_rank_B_name, B_init


def initialize_low_rank_with_svd_parallel(model, w_bits, low_rank_dim):
    """병렬 SVD를 사용해서 low-rank branch를 초기화"""
    named_params = dict(model.named_parameters())

    # Prepare tasks for parallel processing
    svd_tasks = []
    for name, param in named_params.items():
        if "low_rank_A" in name:
            weight_name = name.replace("low_rank_A", "weight")
            weight_scale_name = name.replace("low_rank_A", "weight_scale")
            weight_zero_point_name = name.replace("low_rank_A", "weight_zero_point")

            original_weight = named_params.get(weight_name)
            weight_scale = named_params.get(weight_scale_name)
            weight_zero_point = named_params.get(weight_zero_point_name)

            if all(
                x is not None
                for x in [original_weight, weight_scale, weight_zero_point]
            ):
                # Calculate quantized weight
                from models.utils_quant import LinearQuant

                quantized_weight = LinearQuant(
                    original_weight,
                    weight_scale,
                    weight_zero_point,
                    w_bits,
                    layerwise=False,
                ).to(original_weight.dtype)

                svd_tasks.append(
                    (name, original_weight, quantized_weight, low_rank_dim)
                )

    # Process SVD tasks in parallel
    with ThreadPoolExecutor(
        max_workers=min(len(svd_tasks), mp.cpu_count())
    ) as executor:
        svd_results = list(
            tqdm(
                executor.map(initialize_low_rank_svd_task, svd_tasks),
                total=len(svd_tasks),
                desc="Initializing low-rank with SVD",
            )
        )

    # Update model parameters
    low_rank_dict = {}
    for A_name, A_init, B_name, B_init in svd_results:
        low_rank_dict[A_name] = A_init
        low_rank_dict[B_name] = B_init

    # Load the initialized low-rank parameters
    model.load_state_dict(low_rank_dict, assign=True, strict=False)


def load_quantized_model(
    model_name, w_bits=16, use_low_rank=False, low_rank_dim=16, low_rank_alpha=1.0
):
    model = FluxTransformer2DModelQuant.from_pretrained(
        pretrained_model_name_or_path=model_name,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        device_map=None,
        w_bits=w_bits,
        use_low_rank=use_low_rank,
        low_rank_dim=low_rank_dim,
        low_rank_alpha=low_rank_alpha,
    )

    weight_scale_dict = {}

    if w_bits <= 8:
        # 병렬 처리를 위한 데이터 준비
        scale_tasks = []
        named_params = dict(model.named_parameters())

        for name, param in named_params.items():
            if "weight_scale" in name:
                weight_name = name.replace("weight_scale", "weight")
                weight_param = named_params.get(weight_name, None)
                if weight_param is not None:
                    scale_tasks.append((name, weight_param, w_bits))

        # ThreadPoolExecutor 사용 (GPU 메모리 공유를 위해)
        with ThreadPoolExecutor(
            max_workers=min(len(scale_tasks), mp.cpu_count())
        ) as executor:
            results = list(
                tqdm(
                    executor.map(calculate_scale_and_zero_point, scale_tasks),
                    total=len(scale_tasks),
                    desc="Calculating scales and zero points",
                )
            )

        # 결과를 dictionary에 저장
        for scale_name, scale, zero_point_name, zero_point in results:
            weight_scale_dict[scale_name] = scale
            weight_scale_dict[zero_point_name] = zero_point

        # Load quantization parameters
        model.load_state_dict(weight_scale_dict, assign=True, strict=False)

        # Initialize low-rank branch with parallel SVD if enabled
        if use_low_rank:
            print("Initializing low-rank branches with parallel SVD...")
            initialize_low_rank_with_svd_parallel(model, w_bits, low_rank_dim)

    else:
        raise NotImplementedError

    # 모델을 GPU로 이동
    model = model.to("cuda")

    return model


def generate_images(pipe, prompt, output_dir, seed):
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.manual_seed(seed)
    image = pipe(prompt, generator=generator).images[0]

    # count index of image
    image_index = len(list(glob.glob(str(output_dir / "*.png"))))
    image.save(output_dir / f"{image_index}.png")


def main(prompt):
    # Sanity Check Full Precision
    model_name = "black-forest-labs/FLUX.1-dev"
    pipe: FluxPipeline = DiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to("cuda")
    count_parameters(pipe.transformer)

    samples_dir = Path("output") / "samples"
    generate_images(pipe, prompt, samples_dir / "bf16", seed=42)
    print(f"Samples saved to '{samples_dir / 'bf16'}'")

    torch.cuda.empty_cache()

    for w_bits in [8, 4, 3, 2, 1]:
        for low_rank in [0, 4, 8, 16, 32, 64]:
            # Clear cache before loading new model
            torch.cuda.empty_cache()

            # Load new model
            pipe.transformer = load_quantized_model(
                model_name,
                w_bits=w_bits,
                use_low_rank=low_rank != 0,
                low_rank_dim=low_rank,
            )
            count_parameters(pipe.transformer)
            generate_images(
                pipe, prompt, samples_dir / f"w{w_bits}" / f"rank{low_rank}", seed=42
            )
            print(
                f"Samples saved to '{samples_dir / f'w{w_bits}' / f'rank{low_rank}'}'"
            )

            # Clean up after each iteration
            del pipe.transformer
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main(prompt="A fantasy landscape with mountains and a river")
    main(
        prompt="Cyberpunk samurai on a neon-lit rooftop at dusk, dramatic rim lighting, 32-bit render"
    )
