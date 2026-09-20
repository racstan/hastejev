import time
import torch
import numpy as np
from hastejev import HasteJevEngine

def benchmark_all_models():
    presets = ["100k", "500k", "1m", "2m", "5m", "10m", "20m"]
    quant_modes = ["fp32", "fp16", "int8_weight", "int4"]
    
    records = []
    
    test_cases = [
        ("Account balance is $14,850.50 with pending transaction of $3,200.00.", ["Approve", "Review", "Decline"]),
        ("User navigated to cart. Click Proceed to Checkout.", [f"DOM Button {i}" for i in range(128)]),
        ("Security audit: IP 10.0.0.1 attempted root login.", ["Block IP", "Alert Admin", "Allow Session"]),
    ]
    
    print("=" * 85)
    print(f"{'Preset':<10} | {'Quant':<12} | {'Params':<12} | {'p50 (ms)':<10} | {'p99 (ms)':<10} | {'Throughput (qps)':<16} | {'Order Bias':<10}")
    print("=" * 85)
    
    for preset in presets:
        for quant in quant_modes:
            engine = HasteJevEngine(preset=preset)
            if quant != "fp32":
                engine.quantize(quant)
                
            params = engine.parameter_count["total"]
            
            # Warmup
            for state, opts in test_cases:
                _ = engine.choice(state, opts)
                
            # Benchmark 100 iterations
            latencies = []
            for _ in range(100):
                state, opts = test_cases[0]
                t0 = time.perf_counter()
                res = engine.choice(state, opts)
                dt = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt)
                
            # Test permutation invariance
            res1 = engine.choice(test_cases[0][0], ["Approve", "Review", "Decline"])
            res2 = engine.choice(test_cases[0][0], ["Decline", "Review", "Approve"])
            diff = max(abs(res1.probabilities[k] - res2.probabilities[k]) for k in res1.probabilities)
            
            p50 = float(np.percentile(latencies, 50))
            p99 = float(np.percentile(latencies, 99))
            qps = float(1000.0 / p50)
            
            records.append({
                "Preset": preset,
                "Quantization": quant,
                "Total Params": f"{params:,}",
                "p50 (ms)": f"{p50:.2f}",
                "p99 (ms)": f"{p99:.2f}",
                "Throughput (qps)": f"{qps:.1f}",
                "Order Bias Δ": f"{diff * 100:.2f}%"
            })
            
            print(f"{preset:<10} | {quant:<12} | {params:<12,} | {p50:<10.2f} | {p99:<10.2f} | {qps:<16.1f} | {diff*100:.2f}%")
            
if __name__ == "__main__":
    benchmark_all_models()
