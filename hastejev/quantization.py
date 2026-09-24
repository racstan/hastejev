import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union

class QuantizedLinear8bit(nn.Module):
    """
    Symmetric 8-bit Weight-Only Quantized Linear Layer.
    Stores weights as int8 tensors with per-channel FP32/FP16 scale factors.
    Reduces memory footprint by 4x while maintaining high numerical fidelity.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Buffer for quantized int8 weights
        self.register_buffer("weight_q", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scales", torch.ones((out_features, 1), dtype=torch.float32))

        if bias:
            self.register_buffer("bias", torch.zeros((out_features,), dtype=torch.float32))
        else:
            self.bias = None

    @property
    def weight(self) -> torch.Tensor:
        """Dequantized weight view for APIs that read `.weight` (e.g. MultiheadAttention)."""
        return self.weight_q.to(torch.float32) * self.scales

    @classmethod
    def from_float(cls, linear_module: nn.Linear) -> "QuantizedLinear8bit":
        q_linear = cls(linear_module.in_features, linear_module.out_features, bias=linear_module.bias is not None)
        # Keep replacement modules on the same device as the source weights.
        q_linear = q_linear.to(device=linear_module.weight.device)
        with torch.no_grad():
            w = linear_module.weight.data.float()
            # Per-channel symmetric scaling: scale = max(|w|, dim=1) / 127.0
            max_val = torch.amax(torch.abs(w), dim=1, keepdim=True).clamp(min=1e-8)
            scales = max_val / 127.0
            q_w = torch.clamp(torch.round(w / scales), -128, 127).to(torch.int8)

            q_linear.weight_q.copy_(q_w)
            q_linear.scales.copy_(scales)
            if linear_module.bias is not None:
                q_linear.bias.copy_(linear_module.bias.data.float().to(q_linear.bias.device))
        return q_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # On-the-fly dequantization: W = weight_q.float() * scales
        w_dequant = self.weight_q.to(device=x.device, dtype=x.dtype) * self.scales.to(device=x.device, dtype=x.dtype)
        b = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w_dequant, b)


class QuantizedLinear4bit(nn.Module):
    """
    Symmetric 4-bit Weight-Only Nibble-Packed Linear Layer.
    Packs two 4-bit signed integers (-8 to +7) per byte (uint8).
    Reduces memory footprint by 8x for micro-edge, WASM, and IoT inference.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        packed_in_features = (in_features + 1) // 2
        self.register_buffer("weight_packed", torch.zeros((out_features, packed_in_features), dtype=torch.uint8))
        self.register_buffer("scales", torch.ones((out_features, 1), dtype=torch.float32))

        if bias:
            self.register_buffer("bias", torch.zeros((out_features,), dtype=torch.float32))
        else:
            self.bias = None

    def _dequant_weight(self, dtype: torch.dtype) -> torch.Tensor:
        low_nibbles = (self.weight_packed & 0x0F).to(torch.int32) - 8
        high_nibbles = ((self.weight_packed >> 4) & 0x0F).to(torch.int32) - 8
        out_f, packed_f = self.weight_packed.shape
        unpacked = torch.zeros((out_f, packed_f * 2), dtype=dtype, device=self.weight_packed.device)
        unpacked[:, 0::2] = low_nibbles.to(dtype=dtype)
        unpacked[:, 1::2] = high_nibbles.to(dtype=dtype)
        return unpacked[:, :self.in_features] * self.scales.to(dtype=dtype)

    @property
    def weight(self) -> torch.Tensor:
        """Dequantized weight view for APIs that read `.weight` (e.g. MultiheadAttention)."""
        return self._dequant_weight(torch.float32)

    @classmethod
    def from_float(cls, linear_module: nn.Linear) -> "QuantizedLinear4bit":
        q_linear = cls(linear_module.in_features, linear_module.out_features, bias=linear_module.bias is not None)
        # Keep replacement modules on the same device as the source weights.
        q_linear = q_linear.to(device=linear_module.weight.device)
        with torch.no_grad():
            w = linear_module.weight.data.float()
            # Per-channel symmetric scaling: scale = max(|w|, dim=1) / 7.0
            max_val = torch.amax(torch.abs(w), dim=1, keepdim=True).clamp(min=1e-8)
            scales = max_val / 7.0
            q_w = torch.clamp(torch.round(w / scales), -8, 7).to(torch.int32)

            # Map signed int4 [-8, 7] to unsigned 4-bit [0, 15] for packing: (q_w + 8) & 0x0F
            q_w_u = (q_w + 8).to(torch.uint8)

            # Pack two nibbles per byte: low 4 bits = even col, high 4 bits = odd col
            in_f = linear_module.in_features
            packed_cols = (in_f + 1) // 2
            packed = torch.zeros((linear_module.out_features, packed_cols), dtype=torch.uint8, device=w.device)

            even_cols = q_w_u[:, 0::2]
            packed[:, :even_cols.shape[1]] = even_cols & 0x0F

            if in_f > 1:
                odd_cols = q_w_u[:, 1::2]
                packed[:, :odd_cols.shape[1]] |= (odd_cols & 0x0F) << 4

            q_linear.weight_packed.copy_(packed.to(q_linear.weight_packed.device))
            q_linear.scales.copy_(scales.to(q_linear.scales.device))
            if linear_module.bias is not None:
                q_linear.bias.copy_(linear_module.bias.data.float().to(q_linear.bias.device))
        return q_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize onto the activation device so CUDA activations match weights.
        w_dequant = self._dequant_weight(x.dtype)
        if w_dequant.device != x.device:
            w_dequant = w_dequant.to(device=x.device)
        b = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w_dequant, b)


def _replace_linear_with_quantized(module: nn.Module, target_class):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, (QuantizedLinear8bit, QuantizedLinear4bit)):
            setattr(module, name, target_class.from_float(child))
        elif isinstance(child, (QuantizedLinear8bit, QuantizedLinear4bit)):
            if isinstance(child, target_class):
                continue
            float_w = child.weight
            src_device = float_w.device
            lin = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None, device=src_device)
            with torch.no_grad():
                lin.weight.copy_(float_w.reshape(child.out_features, child.in_features).to(src_device))
                if child.bias is not None:
                    lin.bias.copy_(child.bias.reshape(-1).to(device=src_device, dtype=lin.bias.dtype))
            setattr(module, name, target_class.from_float(lin))
        else:
            _replace_linear_with_quantized(child, target_class)


def _has_float_linears(module: nn.Module) -> bool:
    return any(isinstance(m, nn.Linear) for m in module.modules())


def quantize_model(model: nn.Module, mode: str) -> nn.Module:
    """
    Applies quantization to a Haste Jev model **in place** and returns the same object.
    Supported modes:
    - 'fp32': full-precision float32
    - 'fp16': float16
    - 'bf16': bfloat16
    - 'int8' / 'int8_dynamic': PyTorch dynamic quantization (falls back to int8_weight)
    - 'int8_weight': symmetric per-channel 8-bit weight-only quantization
    - 'int4': symmetric per-channel 4-bit nibble-packed weight quantization
    """
    mode = mode.lower().strip()

    if mode == "fp32":
        model.float()
        return model

    elif mode == "fp16":
        model.half()
        return model

    elif mode == "bf16":
        model.bfloat16()
        return model

    elif mode in ("int8", "int8_dynamic"):
        # torch.ao dynamic quant is deprecated and breaks nn.TransformerEncoder
        # (out_proj.weight access). Use weight-only int8 instead.
        _replace_linear_with_quantized(model, QuantizedLinear8bit)
        return model

    elif mode == "int8_weight":
        _replace_linear_with_quantized(model, QuantizedLinear8bit)
        return model

    elif mode in ("int4", "int4_weight", "q4"):
        _replace_linear_with_quantized(model, QuantizedLinear4bit)
        return model

    else:
        raise ValueError(
            f"Unsupported quantization mode '{mode}'. "
            "Choose from: 'fp32', 'fp16', 'bf16', 'int8', 'int8_weight', 'int4'"
        )
