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

    @classmethod
    def from_float(cls, linear_module: nn.Linear) -> "QuantizedLinear8bit":
        q_linear = cls(linear_module.in_features, linear_module.out_features, bias=linear_module.bias is not None)
        with torch.no_grad():
            w = linear_module.weight.data.float()
            # Per-channel symmetric scaling: scale = max(|w|, dim=1) / 127.0
            max_val = torch.amax(torch.abs(w), dim=1, keepdim=True).clamp(min=1e-8)
            scales = max_val / 127.0
            q_w = torch.clamp(torch.round(w / scales), -128, 127).to(torch.int8)
            
            q_linear.weight_q.copy_(q_w)
            q_linear.scales.copy_(scales)
            if linear_module.bias is not None:
                q_linear.bias.copy_(linear_module.bias.data.float())
        return q_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # On-the-fly dequantization: W = weight_q.float() * scales
        w_dequant = self.weight_q.to(dtype=x.dtype) * self.scales.to(dtype=x.dtype)
        b = self.bias.to(dtype=x.dtype) if self.bias is not None else None
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

    @classmethod
    def from_float(cls, linear_module: nn.Linear) -> "QuantizedLinear4bit":
        q_linear = cls(linear_module.in_features, linear_module.out_features, bias=linear_module.bias is not None)
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
                
            q_linear.weight_packed.copy_(packed)
            q_linear.scales.copy_(scales)
            if linear_module.bias is not None:
                q_linear.bias.copy_(linear_module.bias.data.float())
        return q_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unpack uint8 into two 4-bit values and restore signed range [-8, 7]
        low_nibbles = (self.weight_packed & 0x0F).to(torch.int32) - 8
        high_nibbles = ((self.weight_packed >> 4) & 0x0F).to(torch.int32) - 8
        
        out_f, packed_f = self.weight_packed.shape
        unpacked = torch.zeros((out_f, packed_f * 2), dtype=x.dtype, device=x.device)
        unpacked[:, 0::2] = low_nibbles.to(dtype=x.dtype)
        unpacked[:, 1::2] = high_nibbles.to(dtype=x.dtype)
        
        # Truncate to exact in_features if odd
        w_dequant = unpacked[:, :self.in_features] * self.scales.to(dtype=x.dtype)
        b = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w_dequant, b)


def _replace_linear_with_quantized(module: nn.Module, target_class):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, target_class.from_float(child))
        else:
            _replace_linear_with_quantized(child, target_class)


def quantize_model(model: nn.Module, mode: str) -> nn.Module:
    """
    Applies quantization to a Haste Jev model.
    Supported modes:
    - 'fp16': Converts floating point weights to half-precision float16
    - 'bf16': Converts floating point weights to bfloat16
    - 'int8' / 'int8_dynamic': Uses PyTorch dynamic quantization for linear layers
    - 'int8_weight': Symmetric per-channel 8-bit weight-only quantization
    - 'int4': Symmetric per-channel 4-bit nibble-packed weight quantization
    """
    mode = mode.lower().strip()
    
    if mode == "fp32":
        return model.float()
        
    elif mode == "fp16":
        return model.half()
        
    elif mode == "bf16":
        return model.bfloat16()
        
    elif mode in ("int8", "int8_dynamic"):
        try:
            quantized = torch.ao.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            return quantized
        except Exception:
            # Fallback to weight-only 8-bit if dynamic quantization is unsupported on platform
            return quantize_model(model, "int8_weight")
            
    elif mode == "int8_weight":
        quantized = copy.deepcopy(model)
        _replace_linear_with_quantized(quantized, QuantizedLinear8bit)
        return quantized
        
    elif mode in ("int4", "int4_weight", "q4"):
        quantized = copy.deepcopy(model)
        _replace_linear_with_quantized(quantized, QuantizedLinear4bit)
        return quantized
        
    else:
        raise ValueError(f"Unsupported quantization mode '{mode}'. Choose from: 'fp32', 'fp16', 'bf16', 'int8', 'int8_weight', 'int4'")
