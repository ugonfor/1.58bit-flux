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
from typing import Tuple, Dict, Any

from models.utils_quant import QuantizeLinear

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


@torch.no_grad()
def _init_one_layer_on_stream(
    name: str,
    layer: "QuantizeLinear",
    do_svd: bool,
    do_quant: bool,
    stream: torch.cuda.Stream,
    eps: float = 1e-8,
) -> Tuple[str, Dict[str, Any]]:
    """
    한 레이어에 대해 기존 클래스 메서드만 사용:
      - layer.svd_init_low_rank()
      - layer.initialize_quant_params()
    를 지정된 CUDA stream에서 실행.
    """
    with torch.cuda.stream(stream):
        info = {
            "shape": tuple(layer.weight.shape),
            "use_low_rank": bool(layer.use_low_rank),
            "w_bits": int(layer.w_bits),
            "weight_layerwise": bool(getattr(layer, "weight_layerwise", False)),
        }

        # GPU에 없으면 이동
        if layer.weight.device.type != "cuda":
            layer.to("cuda")

        # 1) SVD(저랭크) 초기화
        if do_svd and layer.use_low_rank:
            # 클래스에 이미 있는 메서드를 그대로 사용
            layer.svd_init_low_rank()  # reduced_rank는 내부적으로 low_rank_dim 사용

        # 2) 양자화 파라미터 초기화
        if do_quant and layer.w_bits < 16:
            # 클래스 메서드 그대로 사용
            layer.initialize_quant_params(eps=eps)

    return name, info


@torch.no_grad()
def initialize_layers_parallel(
    model: nn.Module,
    do_svd: bool = True,
    do_quant: bool = True,
    max_streams: int = None,
    verbose: bool = True,
    eps: float = 1e-8,
) -> Dict[str, Dict[str, Any]]:
    """
    모델 내 모든 QuantizeLinear 레이어에 대해
    클래스 메서드만 사용하여(SVD+Quant) 병렬 초기화.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 환경이 필요합니다.")

    # 대상 레이어 수집
    targets: List[Tuple[str, "QuantizeLinear"]] = [
        (n, m) for n, m in model.named_modules() if isinstance(m, QuantizeLinear)
    ]
    if len(targets) == 0:
        if verbose:
            print("[init] QuantizeLinear 레이어가 없습니다.")
        return {}

    n_layers = len(targets)
    n_streams = max_streams or min(8, n_layers)
    streams = [torch.cuda.Stream() for _ in range(n_streams)]

    if verbose:
        print(f"[init] target layers: {n_layers}, cuda streams: {n_streams}")
        print(f"[init] do_svd={do_svd}, do_quant={do_quant}")

    results: Dict[str, Dict[str, Any]] = {}

    # Python 오버헤드 줄이기 위해 ThreadPool + 각자 stream 사용
    with ThreadPoolExecutor(max_workers=n_streams) as ex:
        futures = []
        for i, (name, layer) in enumerate(targets):
            s = streams[i % n_streams]
            futures.append(ex.submit(_init_one_layer_on_stream, name, layer, do_svd, do_quant, s, eps))

        for fut in futures:
            try:
                name, info = fut.result()
                results[name] = info
                if verbose:
                    print(f"[init] ok: {name} {info}")
            except Exception as e:
                # 레이어별 에러만 기록하고 계속 진행
                results[name] = {"error": str(e)}
                if verbose:
                    print(f"[init] FAIL: {name} -> {e}")

    # 모든 stream 동기화
    torch.cuda.synchronize()
    if verbose:
        print("[init] all CUDA streams synchronized.")
    return results


def load_quantized_model(
    model_name,
    w_bits=16,
    use_low_rank=False,
    low_rank_dim=16,
    low_rank_alpha=1.0,
    weight_layerwise: bool = False,
    verbose: bool = True,
):
    """
    1) 모델 로드
    2) GPU 이동
    3) (클래스 내 메서드만 사용해서) 레이어별 SVD + Quant 병렬 초기화
    """
    # 1) 모델 로드 (사용자 코드 그대로 유지)
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

    # (선택) 하위 모듈들이 QuantizeLinear일 때, 파라미터 일관성 보정
    for m in model.modules():
        if isinstance(m, QuantizeLinear):
            m.w_bits = w_bits
            m.use_low_rank = use_low_rank
            m.low_rank_dim = low_rank_dim
            m.low_rank_alpha = low_rank_alpha
            if hasattr(m, "weight_layerwise"):
                m.weight_layerwise = weight_layerwise

    # 2) GPU로 이동
    model = model.to("cuda")

    # 3) 병렬 초기화
    do_svd = bool(use_low_rank)
    do_quant = bool(w_bits < 16)

    _ = initialize_layers_parallel(
        model,
        do_svd=do_svd,
        do_quant=do_quant,
        max_streams=None,    # 기본 8 또는 레이어 수
        verbose=verbose,
        eps=1e-8,
    )

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
    generate_images(pipe, prompt, samples_dir / f"bf16", seed=42)
    print(f"Samples saved to '{samples_dir / f'bf16'}'")

    pipe.transformer.to("cpu")
    del pipe.transformer
    torch.cuda.empty_cache()

    for w_bits in [8, 4, 3, 2, 1]:
        for low_rank in [0, 4, 8, 16, 32, 64]:
            # Load new model
            pipe.transformer = load_quantized_model(
                model_name,
                w_bits=w_bits,
                use_low_rank=low_rank != 0,
                low_rank_dim=low_rank,
            )
            count_parameters(pipe.transformer)
            generate_images(
                pipe,
                prompt,
                samples_dir / f"w{w_bits}" / f"rank{low_rank}",
                seed=42,
            )
            print(
                f"Samples saved to '{samples_dir / f'w{w_bits}' / f'rank{low_rank}'}'"
            )

            # Clean up after each iteration
            pipe.transformer.to("cpu")
            del pipe.transformer
            torch.cuda.empty_cache()

def test(prompt):
    # Sanity Check Full Precision
    model_name = "black-forest-labs/FLUX.1-dev"
    pipe: FluxPipeline = DiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to("cuda")
    count_parameters(pipe.transformer)

    samples_dir = Path("output") / "samples"
    pipe.transformer.to("cpu")
    del pipe.transformer
    torch.cuda.empty_cache()

    for w_bits in [4]:
        for low_rank in [16, 64]:
            # Load new model
            pipe.transformer = load_quantized_model(
                model_name,
                w_bits=w_bits,
                use_low_rank=low_rank != 0,
                low_rank_dim=low_rank,
            )
            count_parameters(pipe.transformer)
            generate_images(
                pipe,
                prompt,
                samples_dir / f"w{w_bits}" / f"rank{low_rank}",
                seed=42,
            )
            print(
                f"Samples saved to '{samples_dir / f'w{w_bits}' / f'rank{low_rank}'}'"
            )

            # Clean up after each iteration
            pipe.transformer.to("cpu")
            del pipe.transformer
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Test both methods
    # test(
    #     prompt="Cyberpunk samurai on a neon-lit rooftop at dusk, dramatic rim lighting, 32-bit render",
    # )
    # test(
    #     prompt="A fantasy landscape with mountains and a river",
    # )

    main(
        prompt="Cyberpunk samurai on a neon-lit rooftop at dusk, dramatic rim lighting, 32-bit render",
    )
    main(
        prompt="A fantasy landscape with mountains and a river",
    )
