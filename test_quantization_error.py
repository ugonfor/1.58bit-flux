import torch
from models.transformer_flux import (
    FluxTransformer2DModel as FluxTransformer2DModelQuant,
)
from models.utils_quant import QuantizeLinear, LinearQuant
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm


def calculate_layer_errors(original_weight, quantized_weight, low_rank_weight=None):
    """개별 레이어의 quantization error 계산"""
    with torch.no_grad():
        # Basic quantization error
        quant_error = original_weight - quantized_weight
        quant_mse = torch.mean(quant_error**2).item()
        quant_mae = torch.mean(torch.abs(quant_error)).item()

        # Relative error (normalized by original weight magnitude)
        weight_norm = torch.norm(original_weight).item()
        quant_relative_error = torch.norm(quant_error).item() / (weight_norm + 1e-8)

        errors = {
            "quantization_mse": quant_mse,
            "quantization_mae": quant_mae,
            "quantization_relative_error": quant_relative_error,
            "original_weight_norm": weight_norm,
        }

        # Low-rank compensation error if available
        if low_rank_weight is not None:
            compensated_weight = quantized_weight + low_rank_weight
            compensation_error = original_weight - compensated_weight
            comp_mse = torch.mean(compensation_error**2).item()
            comp_mae = torch.mean(torch.abs(compensation_error)).item()
            comp_relative_error = torch.norm(compensation_error).item() / (
                weight_norm + 1e-8
            )

            # Error reduction metrics
            mse_reduction = (quant_mse - comp_mse) / (quant_mse + 1e-8)
            mae_reduction = (quant_mae - comp_mae) / (quant_mae + 1e-8)
            relative_error_reduction = (quant_relative_error - comp_relative_error) / (
                quant_relative_error + 1e-8
            )

            errors.update(
                {
                    "compensation_mse": comp_mse,
                    "compensation_mae": comp_mae,
                    "compensation_relative_error": comp_relative_error,
                    "mse_reduction_ratio": mse_reduction,
                    "mae_reduction_ratio": mae_reduction,
                    "relative_error_reduction_ratio": relative_error_reduction,
                }
            )

        return errors


def extract_weights_from_model(model, w_bits):
    """모델에서 original, quantized, low-rank weights 추출"""
    layer_errors = {}

    for name, module in tqdm(model.named_modules(), desc="Analyzing layers"):
        if isinstance(module, QuantizeLinear):
            layer_name = name

            # Get original weight
            original_weight = module.weight.detach().clone()

            # Calculate quantized weight
            if w_bits < 16:
                quantized_weight = LinearQuant(
                    original_weight,
                    module.weight_scale,
                    module.weight_zero_point,
                    w_bits,
                    layerwise=False,
                ).to(original_weight.dtype)
            else:
                quantized_weight = original_weight

            # Get low-rank weight if available
            low_rank_weight = None
            if hasattr(module, "low_rank_A") and hasattr(module, "low_rank_B"):
                low_rank_weight = (
                    torch.matmul(module.low_rank_A, module.low_rank_B)
                    * module.low_rank_alpha
                )

            # Calculate errors
            errors = calculate_layer_errors(
                original_weight, quantized_weight, low_rank_weight
            )
            layer_errors[layer_name] = errors

    return layer_errors


def analyze_quantization_errors(
    model_name, w_bits_list=[8, 4, 2], low_rank_dims=[0, 8, 16, 32]
):
    """다양한 설정에서 quantization error 분석"""
    results = {}

    for w_bits in w_bits_list:
        print(f"\n=== Analyzing {w_bits}-bit quantization ===")
        results[f"{w_bits}bit"] = {}

        for low_rank_dim in low_rank_dims:
            print(f"  Low-rank dimension: {low_rank_dim}")

            # Load model with specific configuration
            model = FluxTransformer2DModelQuant.from_pretrained(
                pretrained_model_name_or_path=model_name,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
                device_map=None,
                w_bits=w_bits,
                use_low_rank=low_rank_dim > 0,
                low_rank_dim=low_rank_dim if low_rank_dim > 0 else None,
                low_rank_alpha=1.0,
            )

            # Initialize quantization parameters (same as in your original code)
            if w_bits <= 8:
                from generate_images import calculate_scale_and_zero_point
                import multiprocessing as mp
                from concurrent.futures import ThreadPoolExecutor

                weight_scale_dict = {}
                scale_tasks = []
                named_params = dict(model.named_parameters())

                for name, param in named_params.items():
                    if "weight_scale" in name:
                        weight_name = name.replace("weight_scale", "weight")
                        weight_param = named_params.get(weight_name, None)
                        if weight_param is not None:
                            scale_tasks.append((name, weight_param, w_bits))

                with ThreadPoolExecutor(
                    max_workers=min(len(scale_tasks), mp.cpu_count())
                ) as executor:
                    scale_results = list(
                        executor.map(calculate_scale_and_zero_point, scale_tasks)
                    )

                for scale_name, scale, zero_point_name, zero_point in scale_results:
                    weight_scale_dict[scale_name] = scale
                    weight_scale_dict[zero_point_name] = zero_point

                model.load_state_dict(weight_scale_dict, assign=True, strict=False)

                # Initialize low-rank branch if enabled
                if low_rank_dim > 0:
                    from generate_images import initialize_low_rank_with_svd_parallel

                    initialize_low_rank_with_svd_parallel(model, w_bits, low_rank_dim)

            # Extract and analyze errors
            layer_errors = extract_weights_from_model(model, w_bits)
            results[f"{w_bits}bit"][f"rank{low_rank_dim}"] = layer_errors

            # Clean up
            del model
            torch.cuda.empty_cache()

    return results


def summarize_results(results):
    """결과 요약 및 통계"""
    summary = {}

    for bit_config, rank_configs in results.items():
        summary[bit_config] = {}

        for rank_config, layer_errors in rank_configs.items():
            if not layer_errors:
                continue

            # Aggregate statistics across all layers
            all_quant_mse = [
                errors["quantization_mse"] for errors in layer_errors.values()
            ]
            all_quant_mae = [
                errors["quantization_mae"] for errors in layer_errors.values()
            ]
            all_quant_rel = [
                errors["quantization_relative_error"]
                for errors in layer_errors.values()
            ]

            stats = {
                "avg_quantization_mse": np.mean(all_quant_mse),
                "std_quantization_mse": np.std(all_quant_mse),
                "avg_quantization_mae": np.mean(all_quant_mae),
                "std_quantization_mae": np.std(all_quant_mae),
                "avg_quantization_relative_error": np.mean(all_quant_rel),
                "std_quantization_relative_error": np.std(all_quant_rel),
                "num_layers": len(layer_errors),
            }

            # Low-rank compensation statistics if available
            comp_mse_list = [
                errors.get("compensation_mse")
                for errors in layer_errors.values()
                if "compensation_mse" in errors
            ]
            if comp_mse_list and all(x is not None for x in comp_mse_list):
                mse_reductions = [
                    errors["mse_reduction_ratio"]
                    for errors in layer_errors.values()
                    if "mse_reduction_ratio" in errors
                ]
                mae_reductions = [
                    errors["mae_reduction_ratio"]
                    for errors in layer_errors.values()
                    if "mae_reduction_ratio" in errors
                ]
                rel_reductions = [
                    errors["relative_error_reduction_ratio"]
                    for errors in layer_errors.values()
                    if "relative_error_reduction_ratio" in errors
                ]

                stats.update(
                    {
                        "avg_compensation_mse": np.mean(comp_mse_list),
                        "std_compensation_mse": np.std(comp_mse_list),
                        "avg_mse_reduction_ratio": np.mean(mse_reductions),
                        "std_mse_reduction_ratio": np.std(mse_reductions),
                        "avg_mae_reduction_ratio": np.mean(mae_reductions),
                        "std_mae_reduction_ratio": np.std(mae_reductions),
                        "avg_relative_error_reduction_ratio": np.mean(rel_reductions),
                        "std_relative_error_reduction_ratio": np.std(rel_reductions),
                    }
                )

            summary[bit_config][rank_config] = stats

    return summary


def plot_error_analysis(summary, output_dir="output/error_analysis"):
    """Error 분석 결과 시각화"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. MSE comparison across different configurations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for i, (bit_config, rank_data) in enumerate(summary.items()):
        ax = axes[i // 2, i % 2]

        ranks = []
        quant_mse = []
        comp_mse = []

        for rank_config, stats in rank_data.items():
            rank_num = int(rank_config.replace("rank", ""))
            ranks.append(rank_num)
            quant_mse.append(stats["avg_quantization_mse"])

            if "avg_compensation_mse" in stats:
                comp_mse.append(stats["avg_compensation_mse"])
            else:
                comp_mse.append(stats["avg_quantization_mse"])

        ax.plot(
            ranks, quant_mse, "o-", label="Quantization Only", color="red", linewidth=2
        )
        ax.plot(
            ranks,
            comp_mse,
            "s-",
            label="With Low-rank Compensation",
            color="blue",
            linewidth=2,
        )
        ax.set_xlabel("Low-rank Dimension")
        ax.set_ylabel("Average MSE")
        ax.set_title(f"{bit_config} Quantization Error")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_dir / "mse_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Error reduction ratio plot
    fig, ax = plt.subplots(figsize=(12, 8))

    for bit_config, rank_data in summary.items():
        ranks = []
        reductions = []

        for rank_config, stats in rank_data.items():
            rank_num = int(rank_config.replace("rank", ""))
            if rank_num > 0 and "avg_mse_reduction_ratio" in stats:
                ranks.append(rank_num)
                reductions.append(
                    stats["avg_mse_reduction_ratio"] * 100
                )  # Convert to percentage

        if ranks:
            ax.plot(
                ranks,
                reductions,
                "o-",
                label=f"{bit_config}",
                linewidth=2,
                markersize=8,
            )

    ax.set_xlabel("Low-rank Dimension")
    ax.set_ylabel("MSE Reduction (%)")
    ax.set_title("Quantization Error Reduction by Low-rank Compensation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_dir / "error_reduction.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    model_name = "black-forest-labs/FLUX.1-dev"

    print("Starting quantization error analysis...")
    results = analyze_quantization_errors(
        model_name,
        w_bits_list=[8, 4, 2],  # 1-bit은 시간이 오래 걸릴 수 있어서 제외
        low_rank_dims=[0, 8, 16, 32],
    )

    print("\nSummarizing results...")
    summary = summarize_results(results)

    # Save detailed results
    output_dir = Path("output/error_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("QUANTIZATION ERROR ANALYSIS SUMMARY")
    print("=" * 80)

    for bit_config, rank_data in summary.items():
        print(f"\n{bit_config.upper()} QUANTIZATION:")
        print("-" * 50)

        for rank_config, stats in rank_data.items():
            rank_num = int(rank_config.replace("rank", ""))
            print(f"  Rank {rank_num}:")
            print(f"    Avg Quantization MSE: {stats['avg_quantization_mse']:.6e}")

            if "avg_compensation_mse" in stats:
                print(f"    Avg Compensation MSE: {stats['avg_compensation_mse']:.6e}")
                print(f"    MSE Reduction: {stats['avg_mse_reduction_ratio']*100:.2f}%")
                print(
                    f"    Relative Error Reduction: {stats['avg_relative_error_reduction_ratio']*100:.2f}%"
                )

    # Create visualizations
    print("\nGenerating plots...")
    plot_error_analysis(summary)

    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
