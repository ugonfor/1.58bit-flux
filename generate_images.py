from models.transformer_flux import (
    FluxTransformer2DModel as FluxTransformer2DModelQuant,
)
from diffusers import FluxPipeline, DiffusionPipeline
import torch

from pathlib import Path
from tqdm import tqdm

import torch.nn as nn
import logging
log = logging.getLogger(__name__)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    linear_params = 0
    quantize_linear_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if not 'transformer' in name:
                continue
            linear_params += sum(p.numel() for p in module.parameters())
            
            # Check if it's QuantizeLinear (imported from your utils_quant)
            from models.utils_quant import QuantizeLinear
            if isinstance(module, QuantizeLinear):
                quantize_linear_params += sum(p.numel() for p in module.parameters())

    linear_ratio = linear_params / total_params if total_params > 0 else 0
    quantize_ratio = quantize_linear_params / total_params if total_params > 0 else 0
    quantize_linear_ratio = quantize_linear_params / linear_params if linear_params > 0 else 0

    print(f"Total parameters: {total_params}")
    print(f"Linear layer parameters: {linear_params}")
    print(f"QuantizeLinear parameters: {quantize_linear_params}")
    print(f"Linear layer parameter ratio: {linear_ratio:.4f}")
    print(f"QuantizeLinear parameter ratio (vs total): {quantize_ratio:.4f}")
    print(f"QuantizeLinear parameter ratio (vs linear): {quantize_linear_ratio:.4f}")

    return {
        "total": total_params,
        "linear": linear_params,
        "quantize_linear": quantize_linear_params,
        "linear_ratio": linear_ratio,
        "quantize_ratio": quantize_ratio,
        "quantize_linear_ratio": quantize_linear_ratio
    }

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as torch_mp

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
    qmax = 2 ** w_bits - 1
    
    # Calculate zero point: zero_point = qmin - round(xmin / scale)
    zero_point = qmin - torch.round(xmin / scale)
    zero_point = torch.clamp(zero_point, qmin, qmax)
    
    return name, scale, zero_point_name, zero_point

def load_quantized_model(model_name, w_bits=16):
    model = FluxTransformer2DModelQuant.from_pretrained(
        pretrained_model_name_or_path=model_name,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        device_map=None,
        w_bits=w_bits
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
        with ThreadPoolExecutor(max_workers=min(len(scale_tasks), mp.cpu_count())) as executor:
            results = list(tqdm(
                executor.map(calculate_scale_and_zero_point, scale_tasks),
                total=len(scale_tasks),
                desc="Calculating scales and zero points"
            ))
        
        # 결과를 dictionary에 저장
        for scale_name, scale, zero_point_name, zero_point in results:
            weight_scale_dict[scale_name] = scale
            weight_scale_dict[zero_point_name] = zero_point
            
    else:
        raise NotImplementedError

    model.load_state_dict(weight_scale_dict, assign=True, strict=False)
    return model

def generate_images(pipe, prompt, output_dir, device, seed):
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.manual_seed(seed)
    images = pipe(prompt, generator=generator).images
    for i, img in enumerate(images):
        img.save(output_dir / f"image_{i}.png")

def main(prompt):
    # Sanity Check Full Precision
    model_name = "black-forest-labs/FLUX.1-dev"
    pipe: FluxPipeline = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda')
    count_parameters(pipe.transformer)

    samples_dir = Path("output") / "samples"
    generate_images(pipe, prompt, samples_dir / "bf16", 'cuda', seed=42)
    print(f"Samples saved to '{samples_dir / 'bf16'}'")

    torch.cuda.empty_cache()


    for w_bits in [1,2,3,4,8]:
        # Clear cache before loading new model
        torch.cuda.empty_cache()
        
        # Load new model
        pipe.transformer = load_quantized_model(model_name, w_bits=w_bits)
        count_parameters(pipe.transformer)
        generate_images(pipe, prompt, samples_dir / f"w{w_bits}", 'cuda', seed=42)
        print(f"Samples saved to '{samples_dir / f'w{w_bits}'}'")
        
        # Clean up after each iteration
        del pipe.transformer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main(prompt="A fantasy landscape with mountains and a river")