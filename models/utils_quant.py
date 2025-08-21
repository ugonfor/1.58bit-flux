# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

def low_rank_decomposition(weight, reduced_rank=32):
    """
    :param          weight: The matrix to decompose, of shape (H, W)
    :param    reduced_rank: rank_of_decomposed_matrix
    :return: L, R
    """

    matrix_dimension = len(weight.size())
    assert matrix_dimension == 2, "Only Support 2D matrix"

    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    reduced_rank = int(reduced_rank)

    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh

    return L, R



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
        qmax = 2**n_bits - 1

        if layerwise:
            # Use single scale and zero_point for entire tensor
            scale = scale.view(1, 1)
            zero_point = zero_point.view(1, 1)

        # Quantize: round((x / scale) + zero_point)
        quantized = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax)

        # Dequantize: scale * (quantized - zero_point)
        dequantized = scale * (quantized - zero_point)

        return dequantized

    def to(self, dtype):
        return self.quantize_dequantize(
            self.tensor, self.scale, self.zero_point, self.n_bits, self.layerwise
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
        self.low_rank_dim = low_rank_dim

        # Low rank branch
        if self.use_low_rank:
            assert low_rank_dim is not None, "low_rank_dim must be specified when use_low_rank=True"
            in_features = self.weight.shape[1]
            out_features = self.weight.shape[0]
            self.low_rank_A = nn.Parameter(torch.randn(out_features, low_rank_dim) * 0.01)
            self.low_rank_B = nn.Parameter(torch.randn(low_rank_dim, in_features) * 0.01)

        # Params for weight quant
        if self.w_bits < 16:
            if self.weight_layerwise:
                self.weight_scale = nn.Parameter(torch.empty(1, 1))
                self.weight_zero_point = nn.Parameter(torch.empty(1, 1))
            else:
                self.weight_scale = nn.Parameter(torch.empty(self.weight.shape[0], 1))
                self.weight_zero_point = nn.Parameter(torch.empty(self.weight.shape[0], 1))

    def forward(self, input_):
        final_weight = self.effective_weight(dtype=input_.dtype)
        out = nn.functional.linear(input_, final_weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)
        return out

    def extra_repr(self) -> str:
        return (
            f"w_bits={self.w_bits}, weight_layerwise={self.weight_layerwise}, "
            f"use_low_rank={self.use_low_rank}, low_rank_dim={self.low_rank_dim}, "
            f"low_rank_alpha={self.low_rank_alpha}"
        )

    # -----------------------------
    # Quant init / helpers
    # -----------------------------
    @torch.no_grad()
    def initialize_quant_params(self, eps: float = 1e-8):
        """
        Min-Max affine quant 초기화.
        - use_low_rank=True, w_bits<=8 => (W - α·A@B) 기준
        - 그 외 => W 기준
        """
        if self.w_bits >= 16:
            return  # 양자화 안 함

        if self.use_low_rank and self.w_bits <= 8:
            LR = torch.matmul(self.low_rank_A, self.low_rank_B)
            calib_W = self.weight - self.low_rank_alpha * LR
        else:
            calib_W = self.weight

        qmin, qmax = 0.0, float(2 ** self.w_bits - 1)
        device, dtype = self.weight.device, self.weight.dtype

        if self.weight_layerwise:
            w_min = torch.min(calib_W)
            w_max = torch.max(calib_W)
            w_range = torch.clamp(w_max - w_min, min=eps)
            scale = (w_range / (qmax - qmin)).to(dtype).to(device)
            zero_point = torch.clamp(torch.round(qmin - w_min / scale), qmin, qmax).to(dtype).to(device)
            self.weight_scale.data = scale.view(1, 1)
            self.weight_zero_point.data = zero_point.view(1, 1)
        else:
            w_min = torch.min(calib_W, dim=1, keepdim=True).values
            w_max = torch.max(calib_W, dim=1, keepdim=True).values
            w_range = torch.clamp(w_max - w_min, min=1e-8)
            scale = (w_range / (qmax - qmin)).to(dtype).to(device)
            zero_point = torch.clamp(torch.round(qmin - w_min / scale), qmin, qmax).to(dtype).to(device)
            self.weight_scale.data = scale
            self.weight_zero_point.data = zero_point

    @torch.no_grad()
    def svd_init_low_rank(self, reduced_rank: int = None):
        if not self.use_low_rank:
            raise RuntimeError("use_low_rank=False; low-rank branch not enabled.")
        rank = reduced_rank if reduced_rank is not None else self.low_rank_dim
        L, R = low_rank_decomposition(self.weight, reduced_rank=rank)
        if L.shape != self.low_rank_A.shape or R.shape != self.low_rank_B.shape:
            raise RuntimeError(
                f"SVD shapes {L.shape},{R.shape} do not match A,B shapes {self.low_rank_A.shape},{self.low_rank_B.shape}"
            )
        self.low_rank_A.copy_(L)
        self.low_rank_B.copy_(R)

    # -----------------------------
    # NEW: effective weight + error measurement
    # -----------------------------
    @torch.no_grad()
    def effective_weight(self, dtype=None):
        """
        현재 설정으로 추론에 쓰일 최종 가중치 W_eff를 복원합니다.
        - w_bits>=16: (use_low_rank ? W + αAB : W)
        - w_bits<=8:  W_eff = Q(W - αAB) + αAB  (use_low_rank=False면 Q(W))
        """
        W = self.weight
        dtype = W.dtype if dtype is None else dtype

        if self.w_bits >= 16:
            if self.use_low_rank:
                LR = torch.matmul(self.low_rank_A, self.low_rank_B)
                return (W + self.low_rank_alpha * LR).to(dtype)
            return W.to(dtype)

        # <=8bit
        if self.use_low_rank:
            LR = torch.matmul(self.low_rank_A, self.low_rank_B)
            W_for_quant = W - self.low_rank_alpha * LR
        else:
            LR = None
            W_for_quant = W

        base_weight = LinearQuant(
            W_for_quant,
            self.weight_scale,
            self.weight_zero_point,
            self.w_bits,
            self.weight_layerwise,
        ).to(dtype)

        return base_weight if LR is None else (base_weight + self.low_rank_alpha * LR).to(dtype)

    @torch.no_grad()
    def quantization_error(
        self,
        input_: torch.Tensor = None,
        per_channel: bool = False,
        calibrate: bool = False,
        dtype=None,
        eps: float = 1e-12,
    ):
        """
        양자화 오차를 측정합니다.
        - 가중치 오차: E_w = W_eff - W  (subtract-before-quant에서는 순수 양자화 잔차)
        - (옵션) 출력 오차: E_y = Y_q - Y_fp, Y_fp = input @ W^T (+bias)
        Args:
            input_: (N, in_features) 제공 시 출력 오차도 계산
            per_channel: True면 채널별(행별) MSE 포함
            calibrate: True면 측정 전에 min-max로 스케일 재산정(initialize_quant_params)
            dtype: 효과 가중치/출력 계산 dtype (기본은 weight.dtype)
        Returns:
            dict:
              {
                "weight": {
                    "mse","mae","max_abs","rel_fro","snr_db",
                    "per_channel_mse": (optional, [out_features])
                  },
                "output": { ... }  # input_ 제공 시
              }
        """
        if self.w_bits < 16:
            # 스케일이 비어있거나 요청 시 재보정
            need_init = (not hasattr(self, "weight_scale")) or (self.weight_scale.numel() == 0)
            if calibrate or need_init:
                self.initialize_quant_params()

        W = self.weight
        W_eff = self.effective_weight(dtype=dtype)

        # --- weight-domain error
        Ew = (W_eff - W).to(W.dtype)
        mse_w = torch.mean(Ew.pow(2)).item()
        mae_w = torch.mean(Ew.abs()).item()
        max_w = torch.max(Ew.abs()).item()

        fro_W = torch.linalg.norm(W, ord="fro").item()
        fro_E = torch.linalg.norm(Ew, ord="fro").item()
        rel_fro = float(fro_E / (fro_W + eps))
        snr_db = float(20.0 * torch.log10(torch.tensor((fro_W + eps) / (fro_E + eps))).item())

        result = {
            "weight": {
                "mse": mse_w,
                "mae": mae_w,
                "max_abs": max_w,
                "rel_fro": rel_fro,
                "snr_db": snr_db,
            }
        }

        if per_channel:
            # 행별 평균 제곱 오차
            # Ew: (out_features, in_features)
            pcmse = torch.mean(Ew.pow(2), dim=1)  # [out_features]
            result["weight"]["per_channel_mse"] = pcmse.detach().cpu().tolist()

        # --- output-domain error (optional)
        if input_ is not None:
            inp = input_.to(W_eff.dtype)
            y_q = nn.functional.linear(inp, W_eff, self.bias)
            y_fp = nn.functional.linear(inp, W, self.bias)
            Ey = (y_q - y_fp).to(y_fp.dtype)
            mse_y = torch.mean(Ey.pow(2)).item()
            mae_y = torch.mean(Ey.abs()).item()
            max_y = torch.max(Ey.abs()).item()

            # 상대 Frobenius (출력 배치 기준)
            fro_Y = torch.linalg.norm(y_fp, ord="fro").item()
            fro_Ey = torch.linalg.norm(Ey, ord="fro").item()
            rel_fro_y = float(fro_Ey / (fro_Y + eps))
            snr_db_y = float(20.0 * torch.log10(torch.tensor((fro_Y + eps) / (fro_Ey + eps))).item())

            result["output"] = {
                "mse": mse_y,
                "mae": mae_y,
                "max_abs": max_y,
                "rel_fro": rel_fro_y,
                "snr_db": snr_db_y,
            }

        return result


if __name__ == "__main__":
    layer = QuantizeLinear(500, 1000,
                       w_bits=4, weight_layerwise=False,
                       use_low_rank=True, low_rank_dim=32, low_rank_alpha=1.0)
    layer.svd_init_low_rank()          # 선택
    layer.initialize_quant_params()    # 권장

    # 1) 가중치 기준 오차만
    err_w = layer.quantization_error()
    print(err_w)

    # 2) 입력 배치로 출력 오차까지
    x = torch.randn(8, 500)
    err = layer.quantization_error(input_=x, per_channel=True)
    print(err["weight"]["snr_db"], err["weight"]["mse"],
        err["output"]["snr_db"], err["output"]["mse"])
