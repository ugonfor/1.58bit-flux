"""
1.58-bit ternary quantization for FLUX transformers.
Reproduces: https://chenglin-yang.github.io/1.58bit.flux.github.io/
Method: absmean {-1, 0, +1} + trainable per-channel scale + optional LoRA adapter.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _svd_low_rank(E: torch.Tensor, rank: int):
    """Rank-r randomized SVD of a 2D matrix E. Returns (L, R) s.t. L @ R ≈ E.
    Uses torch.svd_lowrank for O(mnr) complexity instead of O(min(m,n)^2 * max(m,n)).
    """
    # svd_lowrank returns U (m,q), S (q,), Vh (n,q) where Vh is already transposed
    U, S, Vh = torch.svd_lowrank(E.float(), q=rank, niter=4)
    sqS = torch.sqrt(S)  # S already has exactly `rank` values
    L = (U * sqS).to(E.dtype)            # (out, rank)
    R = (sqS.unsqueeze(1) * Vh.T).to(E.dtype)  # (rank, in)
    return L, R


def _ternary_quant(W: torch.Tensor, per_channel: bool):
    """Quantize W to ternary {-1,0,+1} with absmean scale. Returns (W_q int8, scale float32)."""
    if per_channel:
        scale = W.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    else:
        scale = W.abs().mean().clamp(min=1e-8)
    W_q = (W / scale).round().clamp_(-1, 1).to(torch.int8)
    return W_q, scale


class TernaryLinear(nn.Module):
    """
    Linear layer with ternary weights {-1, 0, +1}.

    Effective weight:
        W_eff = weight_q * scale  +  lora_A @ lora_B   (if lora_rank > 0)

    Trainable params (when used for distillation fine-tuning):
        - scale:   per-channel absmean multiplier  (nn.Parameter)
        - lora_A, lora_B: low-rank residual adapter (nn.Parameter, zero-or-SVD init)

    When loftq_iters > 1, alternating quantization/SVD refinement (LoftQ algorithm)
    is used to minimize ||W - (W_q * scale + lora_A @ lora_B)||_F before training starts.
    """

    def __init__(
        self,
        linear: nn.Linear,
        per_channel: bool = True,
        lora_rank: int = 0,
        svd_init: bool = True,
        loftq_iters: int = 1,
    ):
        super().__init__()
        W = linear.weight.data.float()  # (out, in)
        dtype = linear.weight.dtype
        dev   = linear.weight.device
        out_f, in_f = W.shape

        # ---- initial ternary quantization (of raw W) ----
        W_q, scale = _ternary_quant(W, per_channel)

        self.lora_rank = lora_rank

        if lora_rank > 0 and svd_init and min(out_f, in_f) >= lora_rank:
            r = min(lora_rank, min(out_f, in_f))

            if loftq_iters <= 1:
                # Single-step SVD init: fit LoRA to ternary residual.
                E = W - W_q.float() * scale
                L, R = _svd_low_rank(E.to(dtype), r)
            else:
                # Iterative LoftQ (LoftQ paper, adapted for ternary).
                # Each iteration: re-quantize (W - LoRA), then fit LoRA to total residual.
                # NOTE: tested on synthetic data — improvement depends on weight distribution.
                L = torch.zeros(out_f, r, dtype=torch.float32, device=dev)
                R = torch.zeros(r, in_f, dtype=torch.float32, device=dev)
                for _ in range(loftq_iters):
                    # Re-quantize what the ternary must capture (W minus current LoRA)
                    W_for_ternary = W - (L @ R)
                    W_q, scale = _ternary_quant(W_for_ternary, per_channel)
                    # Fit LoRA to TOTAL residual of W (not just W_for_ternary residual)
                    E = W - W_q.float() * scale
                    L_new, R_new = _svd_low_rank(E.to(dtype), r)
                    L = L_new.float()
                    R = R_new.float()
                L = L.to(dtype)
                R = R.to(dtype)

            lora_A_init = torch.zeros(out_f, lora_rank, dtype=dtype, device=dev)
            lora_B_init = torch.zeros(lora_rank, in_f,  dtype=dtype, device=dev)
            lora_A_init[:, :r].copy_(L)
            lora_B_init[:r, :].copy_(R)
        else:
            lora_A_init = None
            lora_B_init = None

        self.register_buffer("weight_q", W_q)
        self.scale = nn.Parameter(scale.to(dtype))

        if lora_rank > 0:
            self.lora_A = nn.Parameter(lora_A_init if lora_A_init is not None
                                       else torch.zeros(out_f, lora_rank, dtype=dtype, device=dev))
            self.lora_B = nn.Parameter(lora_B_init if lora_B_init is not None
                                       else torch.zeros(lora_rank, in_f,  dtype=dtype, device=dev))

        self.bias = linear.bias
        self.in_features  = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Memory-efficient: avoid materializing full (out, in) W_eff.
        # x @ (W_q * scale).T = (x @ W_q.T) * scale.T  — mathematically identical
        W_q = self.weight_q.to(x.dtype)
        out = F.linear(x, W_q) * self.scale.view(1, -1)  # broadcast (1, out)
        if self.lora_rank > 0:
            # (x @ lora_B.T) @ lora_A.T avoids materializing (out, in) lora_A @ lora_B
            out = out + (x @ self.lora_B.T) @ self.lora_A.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bits=1.58, lora_rank={self.lora_rank}")


def quantize_to_ternary(
    model: nn.Module,
    per_channel: bool = True,
    lora_rank: int = 0,
    svd_init: bool = True,
    loftq_iters: int = 1,
    verbose: bool = False,
) -> nn.Module:
    """Replace all nn.Linear layers with TernaryLinear in-place."""
    replaced = 0

    def _replace(module: nn.Module, prefix: str = ""):
        nonlocal replaced
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, TernaryLinear):
                pass
            elif isinstance(child, nn.Linear):
                tl = TernaryLinear(child, per_channel=per_channel,
                                   lora_rank=lora_rank, svd_init=svd_init,
                                   loftq_iters=loftq_iters)
                setattr(module, name, tl)
                replaced += 1
                if verbose:
                    print(f"  [ternary] {full} ({child.in_features}→{child.out_features})")
            else:
                _replace(child, full)

    _replace(model)
    print(f"[ternary] Quantized {replaced} Linear → TernaryLinear "
          f"(lora_rank={lora_rank}, svd_init={svd_init}, loftq_iters={loftq_iters})")
    return model


def memory_stats(model: nn.Module) -> dict:
    """Estimate model memory footprint in MB."""
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return {"params_mb": params / 1024**2,
            "buffers_mb": buffers / 1024**2,
            "total_mb": (params + buffers) / 1024**2}


def trainable_params(model: nn.Module) -> list:
    """Return list of (name, param) for all trainable TernaryLinear params."""
    result = []
    for name, m in model.named_modules():
        if isinstance(m, TernaryLinear):
            result.append((f"{name}.scale", m.scale))
            if m.lora_rank > 0:
                result.append((f"{name}.lora_A", m.lora_A))
                result.append((f"{name}.lora_B", m.lora_B))
    return result
