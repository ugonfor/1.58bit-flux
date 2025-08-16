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

def load_quantized_model(model_args, training_args, w_bits=16):
    dtype = torch.bfloat16 if training_args.bf16 else torch.float

    model = FluxTransformer2DModelQuant.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        subfolder="transformer",
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        device_map=None,
        w_bits=w_bits
    )

    weight_scale_dict = {}
    for name, param in tqdm(model.named_parameters(), desc="Initializing weight_scale and weight_zero_point"):
        if "weight_scale" in name:
            weight_name = name.replace("weight_scale", "weight")
            weight_param = dict(model.named_parameters()).get(weight_name, None)

            if w_bits <= 8:
                # Calculate scale
                xmax, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                maxq = 2 ** (w_bits - 1) - 1
                scale = xmax / maxq
                weight_scale_dict[name] = scale
                
                # Calculate zero_point
                zero_point_name = name.replace("weight_scale", "weight_zero_point")
                xmin, _ = torch.min(weight_param, dim=-1, keepdim=True)
                xmax, _ = torch.max(weight_param, dim=-1, keepdim=True)
                qmin = 0
                qmax = 2 ** w_bits - 1
                
                # Calculate zero point: zero_point = qmin - round(xmin / scale)
                zero_point = qmin - torch.round(xmin / scale)
                zero_point = torch.clamp(zero_point, qmin, qmax)
                weight_scale_dict[zero_point_name] = zero_point
            else:
                raise NotImplementedError

    model.load_state_dict(weight_scale_dict, assign=True, strict=False)

    return model

def generate_images(pipe, prompt, num_images, output_dir, device, seed):
    generator = torch.manual_seed(seed)
    images = pipe(prompt, num_images=num_images, generator=generator).images
    for i, img in enumerate(images):
        img.save(output_dir / f"image_{i}.png")

def main(prompt):
    # Sanity Check Full Precision
    model_name = "black-forest-labs/FLUX.1-dev"
    pipe: FluxPipeline = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda')

    count_parameters(pipe.transformer)

    samples_dir = Path("output") / "samples" / "bf16"
    generate_images(pipe, prompt, 2, samples_dir / "try1", 'cuda', seed=42)
    generate_images(pipe, prompt, 2, samples_dir / "try2", 'cuda', seed=42)
    print(f"Samples saved to '{samples_dir}'")
    
    del pipe.transformer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main(prompt="A fantasy landscape with mountains and a river")