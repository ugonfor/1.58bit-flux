"""
Packed ternary linear layer: stores weights in 2-bit packed format (4 weights/byte).
8x weight compression vs BF16, 4x vs INT8.

During forward, unpacks one layer at a time → only one layer's worth of
expanded weights in VRAM at any time. Other 503 layers stay packed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def pack_ternary_weights(weight_q: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pack INT8 ternary {-1,0,+1} to 2-bit packed uint8 (4 per byte).

    Encoding: {-1,0,+1} -> {0,1,2}, packed LSB-first.
    Returns (packed_tensor, original_in_features).
    """
    out_f, in_f = weight_q.shape
    pad = (4 - in_f % 4) % 4
    if pad > 0:
        weight_q = F.pad(weight_q, (0, pad), value=0)
    encoded = (weight_q.to(torch.int32) + 1).to(torch.uint8)
    reshaped = encoded.reshape(out_f, -1, 4)
    packed = (reshaped[:, :, 0]
              | (reshaped[:, :, 1] << 2)
              | (reshaped[:, :, 2] << 4)
              | (reshaped[:, :, 3] << 6))
    return packed.contiguous(), in_f


def unpack_ternary_weights(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """Unpack 2-bit packed uint8 → INT8 ternary {-1,0,+1}."""
    p = packed.to(torch.int32)
    v0 = p & 0x03
    v1 = (p >> 2) & 0x03
    v2 = (p >> 4) & 0x03
    v3 = (p >> 6) & 0x03
    unpacked = torch.stack([v0, v1, v2, v3], dim=-1).reshape(packed.shape[0], -1)
    return (unpacked[:, :in_features] - 1).to(torch.int8)


class PackedTernaryLinear(nn.Module):
    """TernaryLinear with 2-bit packed weight storage.

    Weight memory: out_features × ceil(in_features/4) bytes (uint8)
    vs original:  out_features × in_features bytes (int8)  → 4x savings
    vs BF16:      out_features × in_features × 2 bytes     → 8x savings

    Forward: unpack → cast → F.linear → scale → LoRA → bias
    Only one layer's expanded weights exist at a time.
    """

    def __init__(self, ternary_linear):
        """Convert a TernaryLinear to PackedTernaryLinear."""
        super().__init__()
        wq = ternary_linear.weight_q  # INT8 buffer
        packed, orig_in = pack_ternary_weights(wq)

        self.register_buffer("weight_packed", packed)
        self.in_features_orig = orig_in
        self.scale = ternary_linear.scale  # nn.Parameter
        self.lora_rank = ternary_linear.lora_rank

        if self.lora_rank > 0:
            self.lora_A = ternary_linear.lora_A
            self.lora_B = ternary_linear.lora_B

        self.bias = ternary_linear.bias
        self.in_features = ternary_linear.in_features
        self.out_features = ternary_linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unpack on-the-fly: uint8 → int8 → input dtype
        W_q = unpack_ternary_weights(self.weight_packed, self.in_features_orig)
        out = F.linear(x, W_q.to(x.dtype)) * self.scale.view(1, -1)
        if self.lora_rank > 0:
            out = out + (x @ self.lora_B.T) @ self.lora_A.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        packed_bytes = self.weight_packed.numel()
        orig_bytes = self.out_features * self.in_features
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bits=1.58(packed), lora_rank={self.lora_rank}, "
                f"compression={orig_bytes/packed_bytes:.1f}x")


def pack_model(model: nn.Module, verbose: bool = True) -> nn.Module:
    """Convert all TernaryLinear layers to PackedTernaryLinear in-place."""
    from models.ternary import TernaryLinear
    converted = 0
    saved_bytes = 0

    def _replace(module: nn.Module, prefix: str = ""):
        nonlocal converted, saved_bytes
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, TernaryLinear):
                old_bytes = child.weight_q.numel() * child.weight_q.element_size()
                packed = PackedTernaryLinear(child)
                new_bytes = packed.weight_packed.numel() * packed.weight_packed.element_size()
                saved_bytes += old_bytes - new_bytes
                setattr(module, name, packed)
                converted += 1
            else:
                _replace(child, full)

    _replace(model)
    if verbose:
        print(f"[pack] Converted {converted} TernaryLinear → PackedTernaryLinear "
              f"(saved {saved_bytes/1024**2:.0f} MB)")
    return model
