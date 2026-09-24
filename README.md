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
- agent-routing
- tool-routing
- intent-classification
- guardrails
- safetensors
- pytorch
- quantized
- int8
- int4
- fp16
---

# Haste Jev: Non-Generative System-1 AI Decision Engine

[![PyPI](https://img.shields.io/pypi/v/hastejev.svg)](https://pypi.org/project/hastejev/)
[![Hugging Face Model Family](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-noffy%2Fhastejev-yellow)](https://huggingface.co/noffy/hastejev)
[![GitHub Repository](https://img.shields.io/badge/GitHub-racstan%2Fhastejev-black?logo=github)](https://github.com/racstan/hastejev)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Option Bias](https://img.shields.io/badge/Option_Order_Bias-0.0%25-success.svg)]()

**Haste Jev** is an open-weights **System-1 decision engine** designed for software that must make **typed, high-stakes decisions under hard time budgets** — not generate prose.

It is built for paths where **accuracy of the decision** and **bounded latency** matter more than fluent language:

- **AI agent routing** — which specialist agent / workflow handles this request
- **Tool routing** — which tool or MCP endpoint to call next
- **Model cascade routing** — cheap model vs expensive model vs human escalation
- **Intent classification** — support, voice, and multimodal agents
- **Browser / UI agent action selection** — DOM control with large action spaces
- **Pre-flight guardrails** — block unsafe prompts/actions before they hit a costly LLM
- **Real-time risk & workflow branching** — approve / flag / decline style decisions

> **Honest status:** research prototype. PICA option-order invariance (0%) and the exact sister-model parameter table are **verified**. Historical badge claims (p99 &lt;15ms, ECE &lt;0.009, 99.4% arithmetic, published INT8 files) were **not reproducible** and have been removed or qualified — see `verify_claims.py` and the local audit under `docs/`. **Do not treat marketing numbers from older cards as fact.**

---

## Why System-1 for accuracy + time constraints

| Constraint | What goes wrong with a generative LLM | What Haste Jev is for |
|---|---|---|
| **Time budget** | Autoregressive decode = hundreds of ms to seconds; multi-hop agent loops stack latency | Single parallel forward pass over a typed state → typed options |
| **Accuracy of dispatch** | Free-form JSON/labels fail to parse; order of options changes logits; humans catch mistakes late | PICA scores options independently → **0% option-order bias** (verified) |
| **Bounded output space** | Open vocabulary invites format drift | `choice` / `score` / `noul` / `range_eval` / `set_choice` only |
| **Cost at volume** | Every trivial “which agent?” call bills a frontier model | Tiny local models (98k–20m) run on CPU / edge |

Industry routing stacks (LiteLLM, OpenRouter auto, vLLM semantic router, LatentGate, NeMo Switchyard, etc.) already treat **routing as a separate fast layer**. Haste Jev targets that layer — and similar “smart if-statements” where a wrong or late branch is expensive.

**When not to use it:** freeform chat, code writing, open-ended analysis, multi-step theorem proving, anything where the *words themselves* are the product. Use a generative model there; keep Haste Jev (or rules) for the **decision**.

---

## Intended use cases (important ones)

### 1. AI agent & multi-agent routing
Route a user request (or a mid-session turn) to the correct specialist agent: billing vs technical vs security vs research. Sub-100ms class routing keeps the loop responsive; confidence can gate escalation to a stronger agent or human.

```python
from hastejev import HasteJevEngine
eng = HasteJevEngine.from_pretrained("noffy/hastejev-1m")
opts = ["billing_agent", "tech_support", "security_review", "research_agent"]
r = eng.choice("Can't push — protected branch error on PR #412", opts)
print(r.decision, r.confidence)
```

### 2. Tool / MCP tool routing
With 50–100+ tools, stuffing every schema into the prompt hurts accuracy and latency. Pre-select a short tool menu with `choice`, then let the LLM only reason over the relevant subset.

### 3. Model cascade & confidence-gated routing
- High confidence → cheap model (or skip the LLM entirely)
- Mid confidence → mid-tier model  
- Low confidence / policy hit → expensive model or human review  

`noul()` returns a boolean + probability for guardrail-style gates; combine with `choice` confidence for two-axis policy.

### 4. Intent classification (voice, support, form automation)
Typed intents (`book`, `cancel`, `escalate`, …) map directly to queues / NATS subjects / workflow engines. Prefer fixed enums over free-text LLM intent tags when taxonomy is stable.

### 5. Browser automation & UI agents
- Rank large DOM action spaces (`choice` over 1k+ candidates — H2-Softmax path)
- `noul` policy checks before click/type
- `range_eval` continuous estimates (scroll offset, progress) with a predicted interval

### 6. Pre-flight guardrails
Scan prompts/actions for injection, policy violations, or sensitive-data risk **before** a costly LLM call. Not a replacement for a real security stack — a cheap first gate.

### 7. Real-time workflow & risk branching
Fraud triage, KYC escalate/hold, ad-bid accept/reject, IoT anomaly acknowledge/ignore — any fixed action set with a latency SLO.

### 8. Speculative typed fan-out
Ask many independent yes/no/score questions about one state in one parallel pass; discard the ones the code path does not need.

---

## Model family (sister models)

Parameter counts are **measured** (match README table to 0.0%). RAM columns are weight-storage estimates only.

| Preset | Hugging Face | Total params | d_model | Layers | Heads | ~RAM FP32 | Typical role |
|---|---|---:|---:|---:|---:|---:|---|
| `100k` | [`noffy/hastejev-100k`](https://huggingface.co/noffy/hastejev-100k) | 98,127 | 48 | 2 | 2 | ~0.4 MB | Edge / WASM / IoT gate |
| `500k` | [`noffy/hastejev-500k`](https://huggingface.co/noffy/hastejev-500k) | 500,091 | 96 | 3 | 4 | ~2.0 MB | Mobile / in-browser |
| `1m` | [`noffy/hastejev-1m`](https://huggingface.co/noffy/hastejev-1m) | 1,106,723 | 128 | 4 | 4 | ~4.4 MB | API sidecar router |
| `2m` | [`noffy/hastejev-2m`](https://huggingface.co/noffy/hastejev-2m) | 1,826,275 | 160 | 4 | 4 | ~7.3 MB | Browser / UI agent |
| `5m` | [`noffy/hastejev-5m`](https://huggingface.co/noffy/hastejev-5m) | 5,003,971 | 224 | 5 | 4 | ~20.0 MB | Finance / KYC branch |
| `10m` | [`noffy/hastejev-10m`](https://huggingface.co/noffy/hastejev-10m) | 10,002,275 | 320 | 5 | 4 | ~40.0 MB | Richer multimodal state |
| `20m` | [`noffy/hastejev`](https://huggingface.co/noffy/hastejev) | 20,383,267 | 256 | 4 | 4 | ~81.5 MB | Largest local prototype |

---

## Architecture (what actually ships)

| Piece | Role | Honest note |
|---|---|---|
| **PICA** | Permutation-invariant cross-attention over options | **Verified:** order bias = 0.0% |
| **STFE** | Scalar / temporal Fourier features + regex numeric parse | Plumbing present; **accuracy claims unproven** without training/eval |
| **H2-Softmax** | Top-M shortlist for large option sets (K ≫ 64) | Encoding dominates end-to-end latency at large K — measure yourself |
| **HIT-Calib** | Temperature + per-class isotonic when `fit_calibration` is called | Not auto-fit on load; published ECE not replicated |
| **Noul** | Keyword polarity gates + neural cosine fallback | Partial; not a full calibrated neural primitive alone |
| **Range** | Neural mean + log-var heads → 95% interval | Not a Gaussian process |

Text embedding is **CRC32 n-gram feature hashing** into a fixed random table — not a pretrained LM. Backbone is stock `nn.TransformerEncoder`.

---

## Installation

```bash
pip install hastejev
```

Or from source:

```bash
pip install git+https://github.com/racstan/hastejev.git
```

---

## Quickstart

```python
from hastejev import HasteJevEngine

# Hub load
eng = HasteJevEngine.from_pretrained("noffy/hastejev-1m")

# Categorical decision
state = "Incoming chat: user asks about duplicate charge on invoice 8841."
options = ["billing_refund", "tech_docs", "security_lock", "sales_followup"]
res = eng.choice(state, options)
print(res.decision, res.confidence, res.probabilities)

# Boolean gate
gate = eng.noul(state, "Is this safe to auto-route without human review?")
print(gate.is_true, gate.probability)

# In-place quantization (fp16 / bf16 / int8 / int4)
eng.quantize("int4")

# Fit calibration on held-out logits (optional)
# eng.fit_calibration(logits, labels)
```

**Quantized Hub files:** re-export with current code via `save_pretrained(..., quantization=...)`. Older `model_int8.safetensors` artifacts may still be FP32 from a pre-fix export bug — the loader now rebuilds quantized structure when it sees real packed/int keys.

---

## Primitives

| API | Output | Typical use |
|---|---|---|
| `choice(state, options)` | Decision + probabilities + confidence | Agent/tool/model routing |
| `score(state, rubric_levels)` | Expectation over ordered tiers | Priority / severity |
| `noul(state, assertion)` | Boolean + probability | Guardrails, policy |
| `range_eval(state, property)` | Value + 95% interval | Scroll, progress, thresholds |
| `set_choice(state, options)` | Multi-label subset | Feature flags, batch tags |

---

## Measured honestly

| Check | Result | How |
|---|---|---|
| Unit tests | **29/29 pass** (CPU, Python 3.14) | `CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -v` |
| Option-order bias | **0.0%** on every preset × quant mode | `benchmarks/run_benchmark.py` |
| Sister-model param table | **Exact match** to HF repositories | `verify_claims.py` |
| int8/int4 export keys | **True** `weight_q` / `weight_packed` after re-export | `scripts/export_hf_models.py --dry-run` |

Sample local micro-benchmark (shared Linux CPU, 30 iters — **your machine will differ**; full matrix in `benchmarks/results.md`):

| Preset | Quant | p50 (ms) | p99 (ms) |
|---|---|---:|---:|
| 100k | fp32 | ~10–16 | ~18–69 |
| 1m | fp32 | ~12–16 | ~20–30 |
| 20m | fp16 | ~17–18 | ~28–50 |

```bash
# Unit tests (CPU)
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -v

# Claim checks (writes claim_verification_results.json)
CUDA_VISIBLE_DEVICES="" python verify_claims.py

# Micro-benchmark (writes benchmarks/results.json + .md on your machine)
CUDA_VISIBLE_DEVICES="" python benchmarks/run_benchmark.py
```

Latency and accuracy **must** be re-measured on your hardware and with **your labeled data**. This repo does not ship a large supervised eval set; arithmetic/temporal badges from earlier marketing are not supported by code in-tree.

---

## Design principles

1. **Typed decisions only** — no free text on the hot path.
2. **Order independence** — option lists are sets, not sequences (PICA).
3. **Small enough to colocate** — run next to the orchestrator, not only in a GPU cluster.
4. **Confidence is a knob, not a promise** — use thresholds + fallbacks; measure calibration yourself.
5. **Fail closed on high risk** — when in doubt, escalate up the cascade.

---

## Citation

```bibtex
@article{hastejev2026,
  title={Haste Jev: Non-Generative System-1 AI Decision Engine Family},
  author={Haste Jev Research Team},
  year={2026},
  url={https://github.com/racstan/hastejev}
}
```

## License
Apache License 2.0.
