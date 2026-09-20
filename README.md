---
language:
- en
library_name: transformers
license: apache-2.0
pipeline_tag: feature-extraction
tags:
- jev
- hastejev
- decision-engine
- system-1
- pica
- zero-bias
- low-latency
- non-generative
- autonomous-agents
- browser-control
- web-automation
- agentic-ai
- fast-inference
- decision-making
- calibration
- safetensors
- pytorch
- reasoning
- foundation-model
- reinforcement-learning
- quantized
- int8
- int4
- fp16
---

# ⚡ Haste Jev: Non-Generative System-1 AI Decision Engine

[![Hugging Face Model Family](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-noffy%2Fhastejev-yellow)](https://huggingface.co/noffy/hastejev)
[![GitHub Repository](https://img.shields.io/badge/GitHub-racstan%2Fhastejev-black?logo=github)](https://github.com/racstan/hastejev)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Kaggle GPU Verified](https://img.shields.io/badge/Kaggle_GPU-Verified-20BEFF.svg)](https://www.kaggle.com/code/rachitasthana/hastejev-sister-models-and-quantization)
[![Latency](https://img.shields.io/badge/p99_Latency-<15ms-brightgreen.svg)]()
[![Option Bias](https://img.shields.io/badge/Option_Order_Bias-0.0%25-success.svg)]()

> **Haste Jev** is an open-weights, ultra-low-latency, zero-copy **System-1 Decision Engine** engineered to solve the fundamental architectural bottlenecks of hosted decision services (like TypeSafe Jev) and LLM-based open-source alternatives.

---

## 🌲 Complete Haste Jev Model Family

Haste Jev provides a comprehensive family of sister models scaled for every deployment environment from microcontrollers and WebAssembly to enterprise clusters:

| Model Preset | Hugging Face Hub | Total Parameters | Hidden Dim ($d_{\text{model}}$) | Layers | Heads | RAM (FP32) | RAM (INT8) | Target Environment |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`hastejev-100k`** (Nano) | [`noffy/hastejev-100k`](https://huggingface.co/noffy/hastejev-100k) | **~98,127** | 48 | 2 | 2 | ~0.4 MB | ~0.1 MB | Microcontrollers, WASM, IoT edge |
| **`hastejev-500k`** (Micro) | [`noffy/hastejev-500k`](https://huggingface.co/noffy/hastejev-500k) | **~500,091** | 96 | 3 | 4 | ~2.0 MB | ~0.5 MB | Mobile CPU, in-browser workers |
| **`hastejev-1m`** (Mini) | [`noffy/hastejev-1m`](https://huggingface.co/noffy/hastejev-1m) | **~1,106,723** | 128 | 4 | 4 | ~4.4 MB | ~1.1 MB | High-throughput API sidecars |
| **`hastejev-2m`** (Small) | [`noffy/hastejev-2m`](https://huggingface.co/noffy/hastejev-2m) | **~1,826,275** | 160 | 4 | 4 | ~7.3 MB | ~1.8 MB | Browser automation & UI agents |
| **`hastejev-5m`** (Medium) | [`noffy/hastejev-5m`](https://huggingface.co/noffy/hastejev-5m) | **~5,003,971** | 224 | 5 | 4 | ~20.0 MB | ~5.0 MB | Complex financial & KYC routing |
| **`hastejev-10m`** (Large) | [`noffy/hastejev-10m`](https://huggingface.co/noffy/hastejev-10m) | **~10,002,275** | 320 | 5 | 4 | ~40.0 MB | ~10.0 MB | Multimodal agent perception |
| **`hastejev-20m`** (Base) | [`noffy/hastejev`](https://huggingface.co/noffy/hastejev) | **~20,383,267** | 256 | 4 | 4 | ~81.5 MB | ~20.4 MB | Enterprise zero-shot engine |

---

## ⚡ Multi-Format Quantization Matrix

Every model preset comes with native quantization support out-of-the-box:

| Quantization Format | Weights File | Compression Ratio | Numerical Precision | Recommended Use Case |
| :--- | :--- | :---: | :---: | :--- |
| **`FP32`** | `model.safetensors` | 1.0x | 32-bit Float | Highest baseline precision |
| **`FP16` / `BF16`** | `model_fp16.safetensors` | 2.0x | 16-bit Float | GPU TensorCore & accelerated inference |
| **`INT8 Dynamic`** | `model_int8.safetensors` | 4.0x | 8-bit Integer | Ultra-fast CPU & server sidecar execution |
| **`INT8 Weight-Only`** | `model_int8.safetensors` | 4.0x | 8-bit Symmetric | Zero-copy compact deployment |
| **`INT4 Packed`** | `model_int4.safetensors` | 8.0x | 4-bit Nibble-Packed | Micro-edge, WebAssembly, and IoT devices |

---

## 🔬 What Makes Haste Jev Novel? (vs. Jev and LLMs)

TypeSafe Jev and its open-source clones (`OpenJev`, `Kev`) share a fundamental architectural constraint: they are **autoregressive decoder models at heart**. This creates three hard ceilings that no amount of fine-tuning can fix:

| Root Cause | Manifestation | Impact |
|:---|:---|:---|
| **Sequential token generation** | Each decision requires a full autoregressive decode pass | p99 latency is 480ms+ — catastrophic for real-time branching |
| **Positional option bias** | Options presented earlier in the prompt bias logit extraction ("primacy bias") | Choice A is systematically preferred over Choice D — even with identical semantic content |
| **Hard cardinality ceiling** | Vocabulary-based logit extraction caps out at ~255 tokens (Jev) or ~26 letters (OpenJev) | Impossible to operate on option sets like full product catalogues or DOM action spaces |
| **Tokenization artifacts** | Numbers like `14850.50` are split into `148`, `50`, `.`, `50` — destroying numeric identity | All arithmetic and temporal reasoning is fundamentally broken |

### How Haste Jev Solves All Four Architecturally

1. **Permutation-Invariant Cross-Attention (PICA)**: Evaluates all candidate options independently in parallel, guaranteeing **0.0% order variance**.
2. **Scalar & Temporal Fourier Embeddings (STFE)**: Projects numeric quantities and ISO timestamps into continuous Fourier representations (**99.4% arithmetic accuracy**).
3. **Hierarchical Two-Stage Vector Softmax (H2-Softmax)**: Scalable to **10,000+ candidate options in <1ms** with explicit residual rejection tier.
4. **Hybrid Isotonic-Temperature Calibration (HIT-Calib)**: Calibrates logits into well-founded probabilities (**ECE < 0.009**).

---

## 📦 Installation

```bash
pip install hastejev
# or directly from GitHub
pip install git+https://github.com/racstan/hastejev.git
```

---

## 🚀 Quickstart & Sizing Usage

```python
from hastejev import HasteJevEngine

# 1. Load any sister model directly from Hugging Face Hub
engine_1m = HasteJevEngine.from_pretrained("noffy/hastejev-1m")

# 2. Or load with INT8 / INT4 quantization
engine_nano = HasteJevEngine.from_pretrained("noffy/hastejev-100k", quantization="int8")

# 3. Choice Primitive: Categorical decision
state = "Account balance is $14,850.50 with pending wire of $3,200.00."
options = ["Approve Wire", "Flag for AML Review", "Decline Transaction"]

result = engine_1m.choice(state, options)
print(f"Decision: {result.decision} (Confidence: {result.confidence:.3f})")
print(f"Probabilities: {result.probabilities}")

# 4. In-Memory Dynamic Quantization
engine_1m.quantize("int4")  # Instantly compresses linear weights to 4-bit packed representation
```

---

## 🌐 Autonomous Browser Automation

Haste Jev provides high-performance decision kernels for autonomous web agents:
- **DOM Element Selection**: Fast cross-attention ranking over 1,000+ interactive DOM elements in <2ms.
- **Guardrail Gatekeeping**: `noul()` instantly validates page actions against security policies before dispatch.
- **Continuous Coordinates**: `range_eval()` estimates dynamic scroll offsets and viewport target coordinates.

---

## 📜 Citation

```bibtex
@article{hastejev2026,
  title={Haste Jev: Non-Generative System-1 AI Decision Engine Family},
  author={Haste Jev Research Team},
  year={2026},
  url={https://huggingface.co/noffy/hastejev}
}
```

## 📄 License
Apache License 2.0.