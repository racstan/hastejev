#!/usr/bin/env python3
"""CPU micro-benchmark for all sister presets × quant modes.

Writes results.json + results.md next to this script.
Latency is machine-specific — treat as a floor/ceiling sample, not a product claim.
"""
import json
import os
import platform
import time

import numpy as np

from hastejev import HasteJevEngine


def _env_meta() -> dict:
    try:
        import torch

        torch_ver = torch.__version__
        cuda = torch.cuda.is_available()
    except Exception:
        torch_ver, cuda = "n/a", False
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch_ver,
        "cuda_available": bool(cuda),
        "note": "CPU-oriented micro-benchmark. Latency varies by machine; re-run locally.",
    }


def benchmark_all_models(write_report: bool = True):
    presets = ["100k", "500k", "1m", "2m", "5m", "10m", "20m"]
    quant_modes = ["fp32", "fp16", "int8_weight", "int4"]
    iters = int(os.environ.get("HASTEJEV_BENCH_ITERS", "50"))

    test_cases = [
        ("Account balance is $14,850.50 with pending transaction of $3,200.00.", ["Approve", "Review", "Decline"]),
        ("User navigated to cart. Click Proceed to Checkout.", [f"DOM Button {i}" for i in range(128)]),
        ("Security audit: IP 10.0.0.1 attempted root login.", ["Block IP", "Alert Admin", "Allow Session"]),
    ]

    records = []
    print("=" * 85)
    print(
        f"{'Preset':<10} | {'Quant':<12} | {'Params':<12} | "
        f"{'p50 (ms)':<10} | {'p99 (ms)':<10} | {'Throughput (qps)':<16} | {'Order Bias':<10}"
    )
    print("=" * 85)

    for preset in presets:
        for quant in quant_modes:
            engine = HasteJevEngine(preset=preset)
            if quant != "fp32":
                try:
                    engine.quantize(quant)
                except Exception as e:
                    print(f"{preset:<10} | {quant:<12} | quantize failed: {e}")
                    continue

            params = engine.parameter_count["total"]

            for state, opts in test_cases:
                _ = engine.choice(state, opts)

            latencies = []
            for _ in range(iters):
                state, opts = test_cases[0]
                t0 = time.perf_counter()
                _ = engine.choice(state, opts)
                latencies.append((time.perf_counter() - t0) * 1000.0)

            res1 = engine.choice(test_cases[0][0], ["Approve", "Review", "Decline"])
            res2 = engine.choice(test_cases[0][0], ["Decline", "Review", "Approve"])
            diff = max(abs(res1.probabilities[k] - res2.probabilities[k]) for k in res1.probabilities)

            p50 = float(np.percentile(latencies, 50))
            p99 = float(np.percentile(latencies, 99))
            qps = float(1000.0 / p50) if p50 > 0 else float("inf")
            rec = {
                "Preset": preset,
                "Quantization": quant,
                "Total Params": params,
                "p50_ms": round(p50, 3),
                "p99_ms": round(p99, 3),
                "qps": round(qps, 1),
                "order_bias_pct": round(diff * 100, 4),
                "iters": iters,
            }
            records.append(rec)
            print(
                f"{preset:<10} | {quant:<12} | {params:<12,} | {p50:<10.3f} | "
                f"{p99:<10.3f} | {qps:<16.1f} | {diff * 100:.4f}%"
            )

    if write_report:
        out_dir = os.path.dirname(os.path.abspath(__file__))
        payload = {"meta": _env_meta(), "results": records}
        path = os.path.join(out_dir, "results.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        md = [
            "# Haste Jev micro-benchmark (this machine only)",
            "",
            "```",
            json.dumps(payload["meta"], indent=2),
            "```",
            "",
            "| Preset | Quant | p50 ms | p99 ms | Order bias |",
            "|---|---|---:|---:|---:|",
        ]
        for r in records:
            md.append(
                f"| {r['Preset']} | {r['Quantization']} | {r['p50_ms']} | "
                f"{r['p99_ms']} | {r['order_bias_pct']}% |"
            )
        with open(os.path.join(out_dir, "results.md"), "w") as f:
            f.write("\n".join(md) + "\n")
        print(f"\nWrote {path} and results.md")
    return records


if __name__ == "__main__":
    benchmark_all_models()
