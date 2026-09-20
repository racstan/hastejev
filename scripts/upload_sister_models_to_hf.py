import os
import json
import torch
from huggingface_hub import HfApi, ModelCard, ModelCardData
from hastejev import HasteJevEngine, HasteJevConfig

token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
api = HfApi(token=token)

presets_info = {
    "100k": {
        "repo_id": "noffy/hastejev-100k",
        "title": "Haste Jev 100k (Nano)",
        "params": "~98k",
        "desc": "Ultra-compact System-1 decision model engineered for WebAssembly (WASM), embedded devices, micro-controllers, and edge IoT devices.",
        "target": "Microcontrollers, WASM, IoT edge",
        "ram": "< 1 MB"
    },
    "500k": {
        "repo_id": "noffy/hastejev-500k",
        "title": "Haste Jev 500k (Micro)",
        "params": "~500k",
        "desc": "Micro System-1 decision model optimized for mobile CPUs, in-browser workers, and client-side web extensions.",
        "target": "Mobile CPU, In-browser workers",
        "ram": "~2 MB"
    },
    "1m": {
        "repo_id": "noffy/hastejev-1m",
        "title": "Haste Jev 1M (Mini)",
        "params": "~1.1M",
        "desc": "High-throughput System-1 decision model designed for web API sidecars, real-time microservices, and high-frequency dispatch.",
        "target": "High-throughput API sidecars",
        "ram": "~4 MB"
    },
    "2m": {
        "repo_id": "noffy/hastejev-2m",
        "title": "Haste Jev 2M (Small)",
        "params": "~1.8M",
        "desc": "Real-time decision kernel for autonomous browser control, robotic automation, and interactive UI agents.",
        "target": "Browser automation & bots",
        "ram": "~7 MB"
    },
    "5m": {
        "repo_id": "noffy/hastejev-5m",
        "title": "Haste Jev 5M (Medium)",
        "params": "~5.0M",
        "desc": "Medium System-1 decision engine tailored for complex financial routing, AML compliance, and multi-criteria risk scoring.",
        "target": "Financial & KYC routing",
        "ram": "~19 MB"
    },
    "10m": {
        "repo_id": "noffy/hastejev-10m",
        "title": "Haste Jev 10M (Large)",
        "params": "~10.0M",
        "desc": "Large-capacity System-1 perception kernel for multimodal agents, autonomous navigation, and high-cardinality action spaces.",
        "target": "Multimodal agent kernels",
        "ram": "~38 MB"
    }
}

print("Starting sister models packaging and Hugging Face Hub release...")

for preset_name, info in presets_info.items():
    repo_id = info["repo_id"]
    print(f"\nProcessing {info['title']} -> {repo_id}...")
    
    # Create repo if not exists
    try:
        api.create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
        print(f"Repository {repo_id} is ready.")
    except Exception as e:
        print(f"Repo notice for {repo_id}: {e}")
        
    # Instantiate preset engine
    engine = HasteJevEngine(preset=preset_name)
    param_counts = engine.parameter_count
    
    tmp_dir = f"/tmp/hastejev_{preset_name}"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Save standard weights
    engine.save_pretrained(tmp_dir)
    
    # Save FP16 weights
    engine_fp16 = HasteJevEngine(preset=preset_name)
    engine_fp16.quantize("fp16")
    engine_fp16.save_pretrained(tmp_dir, quantization="fp16")
    
    # Save INT8 weights
    engine_int8 = HasteJevEngine(preset=preset_name)
    engine_int8.quantize("int8_weight")
    engine_int8.save_pretrained(tmp_dir, quantization="int8")
    
    # Save INT4 weights
    engine_int4 = HasteJevEngine(preset=preset_name)
    engine_int4.quantize("int4")
    engine_int4.save_pretrained(tmp_dir, quantization="int4")
    
    # Build Model Card
    card_data = ModelCardData(
        language=["en"],
        license="apache-2.0",
        library_name="transformers",
        tags=[
            "jev",
            "hastejev",
            f"hastejev-{preset_name}",
            "decision-engine",
            "system-1",
            "pica",
            "zero-bias",
            "low-latency",
            "non-generative",
            "autonomous-agents",
            "browser-control",
            "web-automation",
            "agentic-ai",
            "fast-inference",
            "decision-making",
            "calibration",
            "safetensors",
            "pytorch",
            "quantized",
            "int8",
            "int4",
            "fp16"
        ],
        pipeline_tag="feature-extraction"
    )
    
    body = f"""# ⚡ {info['title']} ({info['params']} Parameters)

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-{repo_id.replace('/', '%2F')}-yellow)](https://huggingface.co/{repo_id})
[![GitHub Repository](https://img.shields.io/badge/GitHub-racstan%2Fhastejev-black?logo=github)](https://github.com/racstan/hastejev)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Parameters](https://img.shields.io/badge/Parameters-{info['params']}-brightgreen.svg)]()
[![Memory](https://img.shields.io/badge/RAM_Footprint-{info['ram'].replace(' ', '_')}-purple.svg)]()

> **{info['title']}** is part of the **Haste Jev** family of open-weights, zero-bias **System-1 Decision Engines**. {info['desc']}

---

## 🔬 Model Specifications

- **Total Parameters**: {param_counts['total']:,} ({info['params']})
- **Trainable Parameters**: {param_counts['trainable']:,}
- **Buffer / Projection Table**: {param_counts['buffers']:,}
- **Hidden Dimension ($d_{{\\text{{model}}}}$)**: {engine.config.d_model}
- **Transformer Layers**: {engine.config.n_layers}
- **Attention Heads**: {engine.config.n_heads}
- **Target Deployment**: {info['target']}
- **Quantization Formats Available**: `FP32`, `FP16` (`model_fp16.safetensors`), `INT8` (`model_int8.safetensors`), `INT4` (`model_int4.safetensors`)

---

## ⚡ Quickstart

```python
from hastejev import HasteJevEngine

# 1. Load standard weights directly from Hugging Face Hub
engine = HasteJevEngine.from_pretrained("{repo_id}")

# 2. Or load with INT8 / INT4 quantization
engine_int8 = HasteJevEngine.from_pretrained("{repo_id}", quantization="int8")

# 3. Execute Decision Primitives
state = "Account balance is $14,850.50 with pending transaction of $3,200.00."
options = ["Approve Transaction", "Flag for Review", "Decline"]

res = engine.choice(state, options)
print(f"Decision: {{res.decision}} (Confidence: {{res.confidence:.3f}})")
```

---

## 📊 Complete Haste Jev Model Family

| Model | Parameters | Hidden Dim | Layers | Heads | RAM (FP32) | RAM (INT8) | Target Use Case |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| [`hastejev-100k`](https://huggingface.co/noffy/hastejev-100k) | **~98k** | 48 | 2 | 2 | ~0.4 MB | ~0.1 MB | Microcontrollers, WASM, IoT |
| [`hastejev-500k`](https://huggingface.co/noffy/hastejev-500k) | **~500k** | 96 | 3 | 4 | ~2.0 MB | ~0.5 MB | Mobile CPU, in-browser workers |
| [`hastejev-1m`](https://huggingface.co/noffy/hastejev-1m) | **~1.1M** | 128 | 4 | 4 | ~4.4 MB | ~1.1 MB | High-throughput API sidecars |
| [`hastejev-2m`](https://huggingface.co/noffy/hastejev-2m) | **~1.8M** | 160 | 4 | 4 | ~7.3 MB | ~1.8 MB | Browser automation & bots |
| [`hastejev-5m`](https://huggingface.co/noffy/hastejev-5m) | **~5.0M** | 224 | 5 | 4 | ~20.0 MB | ~5.0 MB | Financial & KYC routing |
| [`hastejev-10m`](https://huggingface.co/noffy/hastejev-10m) | **~10.0M** | 320 | 5 | 4 | ~40.0 MB | ~10.0 MB | Multimodal agent kernels |
| [`hastejev-20m`](https://huggingface.co/noffy/hastejev) | **~20.4M** | 256 | 4 | 4 | ~81.5 MB | ~20.4 MB | Enterprise decision engine |

---

## 📄 License
Apache License 2.0.
"""
    
    full_card_text = f"---\n{card_data.to_yaml()}\n---\n\n{body}"
    card = ModelCard(full_card_text)
    card.validate()
    
    with open(os.path.join(tmp_dir, "README.md"), "w") as f:
        f.write(full_card_text)
        
    # Upload entire directory to Hugging Face
    api.upload_folder(
        folder_path=tmp_dir,
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message=f"feat: release {info['title']} weights and quantized formats"
    )
    print(f"Successfully uploaded {info['title']} to {repo_id}!")

print("\n🎉 All sister models and quantized formats successfully uploaded to Hugging Face Hub!")
