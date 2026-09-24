#!/usr/bin/env python3
"""
Re-export sister-model weights to Hugging Face with *honest* model cards.

Requires HF_TOKEN (write access to noffy/*).
Quantizes in place before save_pretrained so int8/int4 files are never
mislabeled FP32 copies.

Usage:
  HF_TOKEN=hf_... python scripts/export_hf_models.py [--presets 100k,500k] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import tempfile

import torch
from huggingface_hub import HfApi, ModelCard, ModelCardData

from hastejev import HasteJevEngine

PRESETS = {
    "100k": {
        "repo_id": "noffy/hastejev-100k",
        "title": "Haste Jev 100k (Nano)",
        "role": "Edge / WASM / IoT gate",
    },
    "500k": {
        "repo_id": "noffy/hastejev-500k",
        "title": "Haste Jev 500k (Micro)",
        "role": "Mobile CPU / in-browser worker",
    },
    "1m": {
        "repo_id": "noffy/hastejev-1m",
        "title": "Haste Jev 1m (Mini)",
        "role": "API sidecar / agent router",
    },
    "2m": {
        "repo_id": "noffy/hastejev-2m",
        "title": "Haste Jev 2m (Small)",
        "role": "Browser / UI agent kernel",
    },
    "5m": {
        "repo_id": "noffy/hastejev-5m",
        "title": "Haste Jev 5m (Medium)",
        "role": "Finance / KYC workflow branch",
    },
    "10m": {
        "repo_id": "noffy/hastejev-10m",
        "title": "Haste Jev 10m (Large)",
        "role": "Richer multimodal state",
    },
    "20m": {
        "repo_id": "noffy/hastejev",
        "title": "Haste Jev 20m (Base)",
        "role": "Largest local prototype",
    },
}


def build_card(repo_id: str, title: str, role: str, engine: HasteJevEngine) -> str:
    counts = engine.parameter_count
    cfg = engine.config
    card_data = ModelCardData(
        language=["en"],
        license="apache-2.0",
        library_name="transformers",
        pipeline_tag="feature-extraction",
        tags=[
            "hastejev",
            "jev",
            "decision-engine",
            "system-1",
            "agent-routing",
            "tool-routing",
            "non-generative",
            "pica",
            "safetensors",
            "pytorch",
            "quantized",
        ],
    )
    body = f"""# {title}

**Role:** {role}

Open-weights **System-1 decision engine** for software paths that need typed
decisions under a time budget (agent routing, tool routing, intent classification,
pre-flight guardrails, browser action selection). Not a chat model.

## Measured specification

| Field | Value |
|---|---|
| Total parameters | {counts['total']:,} |
| Trainable parameters | {counts['trainable']:,} |
| Hash-table buffers | {counts['buffers']:,} |
| d_model | {cfg.d_model} |
| Layers | {cfg.n_layers} |
| Heads | {cfg.n_heads} |

Parameter counts match the GitHub README table (verified with `verify_claims.py`).

## Honest claims

| Claim | Status |
|---|---|
| PICA option-order bias = 0.0% | **Verified** architecturally |
| Exact parameter table | **Verified** |
| FP32 ~0.4 MB for 100k weights | **Verified** (weight storage only) |
| p99 &lt; 15ms / ECE &lt; 0.009 / 99.4% arithmetic | **Not verified** — do not cite from this card |
| Published latency / accuracy on your workload | **Measure yourself** |

Weights may be lightly or untrained prototypes depending on export; treat behavioral
accuracy as unknown until you evaluate on labeled data.

## Quickstart

```python
from hastejev import HasteJevEngine

eng = HasteJevEngine.from_pretrained("{repo_id}")
# eng = HasteJevEngine.from_pretrained("{repo_id}", quantization="int4")

r = eng.choice(
    "Request: reset password for user@corp.example",
    ["auth_self_service", "billing", "security_review"],
)
print(r.decision, r.confidence)
```

Install: `pip install git+https://github.com/racstan/hastejev.git`

## Files

| File | Contents |
|---|---|
| `model.safetensors` / `pytorch_model.bin` | FP32 state dict |
| `model_fp16.safetensors` | FP16 |
| `model_int8.safetensors` | True weight-only int8 (`weight_q`) when re-exported with ≥1.1.0 |
| `model_int4.safetensors` | True packed int4 (`weight_packed`) when re-exported with ≥1.1.0 |

Older revisions of `model_int8`/`model_int4` may be mislabeled FP32; re-export or
re-download after this commit.

## License
Apache-2.0
"""
    full = f"---\n{card_data.to_yaml()}\n---\n\n{body}"
    ModelCard(full).validate()
    return full


def export_preset(api: HfApi, preset: str, info: dict, dry_run: bool) -> None:
    repo_id = info["repo_id"]
    print(f"\n== {preset} -> {repo_id}")
    engine = HasteJevEngine(preset=preset)
    card = build_card(repo_id, info["title"], info["role"], engine)

    if dry_run:
        # Verify quantized save produces real packed/int keys without uploading
        with tempfile.TemporaryDirectory() as d:
            eng4 = HasteJevEngine(preset=preset)
            eng4.save_pretrained(d, quantization="int4")
            from safetensors.torch import load_file

            st = load_file(os.path.join(d, "model_int4.safetensors"))
            has_packed = any(k.endswith("weight_packed") for k in st)
            eng8 = HasteJevEngine(preset=preset)
            eng8.save_pretrained(d, quantization="int8_weight")
            st8 = load_file(os.path.join(d, "model_int8_weight.safetensors")) if os.path.exists(
                os.path.join(d, "model_int8_weight.safetensors")
            ) else load_file(os.path.join(d, "model.safetensors"))
            has_q8 = any(k.endswith("weight_q") for k in st8)
            print(f"  dry-run int4 packed={has_packed} int8 weight_q={has_q8}")
            print(f"  card bytes={len(card)}")
        return

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    with tempfile.TemporaryDirectory() as d:
        eng = HasteJevEngine(preset=preset)
        eng.save_pretrained(d)  # fp32

        eng16 = HasteJevEngine(preset=preset)
        eng16.save_pretrained(d, quantization="fp16")

        # Name matches historical Hub layout: model_int8 / model_int4
        eng8 = HasteJevEngine(preset=preset)
        eng8.quantize("int8_weight")
        eng8.save_pretrained(d, quantization="int8_weight")
        # also drop a model_int8.safetensors alias with true int8 keys
        from safetensors.torch import save_file

        state8 = {k: v.cpu() for k, v in eng8.state_dict().items()}
        save_file(state8, os.path.join(d, "model_int8.safetensors"))
        eng8.config.quantization = "int8"
        with open(os.path.join(d, "config.json"), "w") as f:
            import json

            json.dump(eng8.config.to_dict(), f, indent=2)

        eng4 = HasteJevEngine(preset=preset)
        eng4.save_pretrained(d, quantization="int4")

        with open(os.path.join(d, "README.md"), "w") as f:
            f.write(card)

        # sanity
        from safetensors.torch import load_file

        st_i8 = load_file(os.path.join(d, "model_int8.safetensors"))
        st_i4 = load_file(os.path.join(d, "model_int4.safetensors"))
        assert any(k.endswith("weight_q") for k in st_i8), "int8 export missing weight_q"
        assert any(k.endswith("weight_packed") for k in st_i4), "int4 export missing weight_packed"

        api.upload_folder(
            folder_path=d,
            repo_id=repo_id,
            repo_type="model",
            commit_message=(
                f"fix: re-export {preset} weights; true int8/int4 keys; honest model card"
            ),
        )
    print(f"  uploaded {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--presets", default="100k,500k,1m,2m,5m,10m,20m")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not args.dry_run and not token:
        raise SystemExit("Set HF_TOKEN to upload (or pass --dry-run).")

    api = HfApi(token=token)
    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    for p in presets:
        if p not in PRESETS:
            raise SystemExit(f"Unknown preset {p!r}")
        export_preset(api, p, PRESETS[p], args.dry_run)
    print("\nDone.")


if __name__ == "__main__":
    main()
