# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

class LinearQuant:
    def __init__(self, tensor, scale, zero_point, n_bits, layerwise=False):
        self.tensor = tensor
        self.scale = scale
        self.zero_point = zero_point
        self.n_bits = n_bits
        self.layerwise = layerwise
        
    def __call__(self, tensor, scale, zero_point, n_bits, layerwise=False):
        return self.quantize_dequantize(tensor, scale, zero_point, n_bits, layerwise)
    
    def quantize_dequantize(self, tensor, scale, zero_point, n_bits, layerwise=False):
        # Calculate quantization range
        qmin = 0
        qmax = 2 ** n_bits - 1
        
        if layerwise:
            # Use single scale and zero_point for entire tensor
            scale = scale.view(1, 1)
            zero_point = zero_point.view(1, 1)
        
        # Quantize: round((x / scale) + zero_point)
        quantized = torch.clamp(
            torch.round(tensor / scale + zero_point), 
            qmin, qmax
        )
        
        # Dequantize: scale * (quantized - zero_point)
        dequantized = scale * (quantized - zero_point)
        
        return dequantized
    
    def to(self, dtype):
        return self.quantize_dequantize(
            self.tensor, 
            self.scale, 
            self.zero_point, 
            self.n_bits, 
            self.layerwise
        ).to(dtype)

class QuantizeLinear(nn.Linear):
    def __init__(
        self,
        *kargs,
        bias=False,
        w_bits=16,
        weight_layerwise=False,
        use_low_rank=False,
        low_rank_dim=None,
        low_rank_alpha=1.0,
    ):
        super(QuantizeLinear, self).__init__(*kargs, bias=bias)
        self.w_bits = w_bits
        self.weight_layerwise = weight_layerwise
        self.use_low_rank = use_low_rank
        self.low_rank_alpha = low_rank_alpha
        
        # Low rank decomposition parameters (not quantized)
        if self.use_low_rank:
            assert low_rank_dim is not None, "low_rank_dim must be specified when use_low_rank=True"
            self.low_rank_dim = low_rank_dim
            in_features = self.weight.shape[1]
            out_features = self.weight.shape[0]
            
            # Initialize low rank matrices A and B (full precision)
            self.low_rank_A = nn.Parameter(torch.randn(out_features, low_rank_dim) * 0.01)
            self.low_rank_B = nn.Parameter(torch.randn(low_rank_dim, in_features) * 0.01)
        
        # params for weight quant
        if self.w_bits < 16:
            if self.weight_layerwise:
                # Single scale and zero_point for entire weight tensor
                self.weight_scale = nn.Parameter(torch.Tensor(1, 1))
                self.weight_zero_point = nn.Parameter(torch.Tensor(1, 1))
            else:
                # Per-channel (output channel) scale and zero_point
                self.weight_scale = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))
                self.weight_zero_point = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))

    def forward(self, input_):
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight

        if self.w_bits >= 16:
            base_weight = self.weight
        elif self.w_bits <= 8:
            base_weight = LinearQuant(
                real_weights,
                self.weight_scale,
                self.weight_zero_point,
                self.w_bits,
                self.weight_layerwise,
            ).to(input_.dtype)
        else:
            raise NotImplementedError

        # Add low rank branch if enabled (full precision)
        if self.use_low_rank:
            low_rank_weight = torch.matmul(self.low_rank_A, self.low_rank_B)
            final_weight = base_weight + self.low_rank_alpha * low_rank_weight
        else:
            final_weight = base_weight

        out = nn.functional.linear(input_, final_weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out

    def extra_repr(self) -> str:
        return f"w_bits={self.w_bits}, weight_layerwise={self.weight_layerwise}, use_low_rank={self.use_low_rank}, low_rank_dim={self.low_rank_dim}, low_rank_alpha={self.low_rank_alpha}"
