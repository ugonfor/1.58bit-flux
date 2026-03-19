"""
Custom Triton kernel for ternary {-1, 0, +1} matrix multiplication.

Packs ternary weights into 2-bit representation (4 weights per byte).
Avoids expanding to BF16/FP16 during matmul — unpacks on-the-fly in SRAM.

Encoding: 0b00 = -1, 0b01 = 0, 0b10 = +1
(so decoded value = packed_val - 1)
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ── Packing / Unpacking Utilities ─────────────────────────────────────────

def pack_ternary(weight_q: torch.Tensor) -> torch.Tensor:
    """Pack INT8 ternary {-1,0,+1} into 2-bit packed uint8 (4 weights per byte).

    Args:
        weight_q: (out_features, in_features) int8 tensor with values in {-1, 0, +1}

    Returns:
        packed: (out_features, ceil(in_features/4)) uint8 tensor
    """
    out_f, in_f = weight_q.shape
    pad = (4 - in_f % 4) % 4
    if pad > 0:
        weight_q = torch.nn.functional.pad(weight_q, (0, pad), value=0)

    encoded = (weight_q.to(torch.int32) + 1).to(torch.uint8)  # {-1,0,+1} -> {0,1,2}
    reshaped = encoded.reshape(out_f, -1, 4)
    packed = (reshaped[:, :, 0]
              | (reshaped[:, :, 1] << 2)
              | (reshaped[:, :, 2] << 4)
              | (reshaped[:, :, 3] << 6))
    return packed.contiguous()


def unpack_ternary(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """Unpack 2-bit packed uint8 back to INT8 ternary {-1,0,+1}."""
    out_f = packed.shape[0]
    p = packed.to(torch.int32)
    v0 = (p & 0x03)
    v1 = (p >> 2) & 0x03
    v2 = (p >> 4) & 0x03
    v3 = (p >> 6) & 0x03
    unpacked = torch.stack([v0, v1, v2, v3], dim=-1).reshape(out_f, -1)
    return (unpacked[:, :in_features] - 1).to(torch.int8).contiguous()


# ── Triton Kernel ─────────────────────────────────────────────────────────

def get_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ]


@triton.autotune(configs=get_autotune_configs(), key=['M', 'N', 'K_packed'])
@triton.jit
def ternary_matmul_kernel(
    a_ptr, b_packed_ptr, c_ptr, scale_ptr,
    M, N, K_packed,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Ternary matmul: C[m,n] = sum_k A[m,k] * decode(B_packed[n,k//4], k%4) * scale[n]

    A: (M, K) bf16, K = K_packed * 4
    B_packed: (N, K_packed) uint8
    scale: (N,) bf16
    C: (M, N) bf16
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over packed K dimension, one packed byte at a time
    for kp in range(K_packed):
        # Load packed byte for each N in block: (BLOCK_N,)
        b_ptrs = b_packed_ptr + offs_n * stride_bn + kp * stride_bk
        b_mask = offs_n < N
        b_packed = tl.load(b_ptrs, mask=b_mask, other=1).to(tl.int32)  # 1 = zero

        # Unpack 4 ternary values and accumulate
        # shift 0: bits [1:0]
        b_val = (b_packed & 0x03) - 1
        k_col = kp * 4
        a_ptrs = a_ptr + offs_m * stride_am + k_col * stride_ak
        a_col = tl.load(a_ptrs, mask=offs_m < M, other=0.0).to(tl.float32)
        acc += a_col[:, None] * b_val[None, :].to(tl.float32)

        # shift 1: bits [3:2]
        b_val = ((b_packed >> 2) & 0x03) - 1
        k_col = kp * 4 + 1
        a_ptrs = a_ptr + offs_m * stride_am + k_col * stride_ak
        a_col = tl.load(a_ptrs, mask=offs_m < M, other=0.0).to(tl.float32)
        acc += a_col[:, None] * b_val[None, :].to(tl.float32)

        # shift 2: bits [5:4]
        b_val = ((b_packed >> 4) & 0x03) - 1
        k_col = kp * 4 + 2
        a_ptrs = a_ptr + offs_m * stride_am + k_col * stride_ak
        a_col = tl.load(a_ptrs, mask=offs_m < M, other=0.0).to(tl.float32)
        acc += a_col[:, None] * b_val[None, :].to(tl.float32)

        # shift 3: bits [7:6]
        b_val = ((b_packed >> 6) & 0x03) - 1
        k_col = kp * 4 + 3
        a_ptrs = a_ptr + offs_m * stride_am + k_col * stride_ak
        a_col = tl.load(a_ptrs, mask=offs_m < M, other=0.0).to(tl.float32)
        acc += a_col[:, None] * b_val[None, :].to(tl.float32)

    # Apply per-channel scale
    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0).to(tl.float32)
    acc = acc * scale[None, :]

    # Store
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


# ── Python wrapper ────────────────────────────────────────────────────────

def ternary_matmul(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    scale: torch.Tensor,
    K_original: int,
) -> torch.Tensor:
    """Compute x @ (unpack(weight_packed) * scale).T using Triton kernel.

    Args:
        x: (*, K) input in bf16/fp16
        weight_packed: (N, K_packed) uint8 packed ternary
        scale: (N,) per-channel scale
        K_original: original K (before padding)

    Returns:
        out: (*, N)
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, K = x_2d.shape
    N, K_packed = weight_packed.shape

    # Pad K to multiple of 4
    pad = (4 - K % 4) % 4
    if pad > 0:
        x_2d = torch.nn.functional.pad(x_2d, (0, pad))
        K = x_2d.shape[1]

    assert K_packed == K // 4, f"K_packed={K_packed} != K//4={K//4}"

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    ternary_matmul_kernel[grid](
        x_2d, weight_packed, out, scale,
        M, N, K_packed,
        x_2d.stride(0), x_2d.stride(1),
        weight_packed.stride(0), weight_packed.stride(1),
        out.stride(0), out.stride(1),
    )

    return out.reshape(*orig_shape[:-1], N)


# ── Benchmark ─────────────────────────────────────────────────────────────

def benchmark_ternary_matmul():
    """Benchmark: Triton packed kernel vs PyTorch INT8→BF16 baseline."""
    import time

    shapes = [
        (4096, 3072, 3072),   # FLUX attention Q/K/V
        (4096, 12288, 3072),  # FLUX MLP fc1
        (4096, 3072, 12288),  # FLUX MLP fc2
        (1, 3072, 3072),      # Single token
    ]

    for M, N, K in shapes:
        device = "cuda"
        dtype = torch.bfloat16

        x = torch.randn(M, K, dtype=dtype, device=device)
        W_q = torch.randint(-1, 2, (N, K), dtype=torch.int8, device=device)
        scale = torch.randn(N, dtype=dtype, device=device).abs() * 0.01

        W_packed = pack_ternary(W_q)

        # Warmup
        for _ in range(5):
            _ = ternary_matmul(x, W_packed, scale, K)
            _ = F.linear(x, W_q.to(dtype)) * scale.view(1, -1)
        torch.cuda.synchronize()

        # Benchmark Triton packed
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            _ = ternary_matmul(x, W_packed, scale, K)
        torch.cuda.synchronize()
        t_triton = (time.perf_counter() - t0) / 30

        # Benchmark PyTorch INT8 → BF16
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            _ = F.linear(x, W_q.to(dtype)) * scale.view(1, -1)
        torch.cuda.synchronize()
        t_torch = (time.perf_counter() - t0) / 30

        # Correctness
        out_triton = ternary_matmul(x, W_packed, scale, K)
        out_torch = F.linear(x, W_q.to(dtype)) * scale.view(1, -1)
        max_err = (out_triton.float() - out_torch.float()).abs().max().item()
        rel_err = max_err / out_torch.float().abs().mean().item()

        print(f"M={M:5d} N={N:5d} K={K:5d} | "
              f"Triton: {t_triton*1000:7.2f}ms  PyTorch: {t_torch*1000:7.2f}ms  "
              f"Speedup: {t_torch/t_triton:5.2f}x | "
              f"MaxErr: {max_err:.4e}  RelErr: {rel_err:.4e} | "
              f"Mem: packed={W_packed.nbytes/1e6:.1f}MB int8={W_q.nbytes/1e6:.1f}MB bf16={N*K*2/1e6:.1f}MB")


if __name__ == "__main__":
    benchmark_ternary_matmul()
