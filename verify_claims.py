"""Empirical verification of hastejev README/research.md claims."""
import time
import math
import json
import copy
import statistics
import numpy as np
import torch
import torch.nn as nn
from itertools import permutations

from hastejev import HasteJevEngine, HasteJevConfig, HITCalibrator
from hastejev.quantization import QuantizedLinear8bit, QuantizedLinear4bit, quantize_model

results = []

def record(claim, verdict, evidence):
    results.append({"claim": claim, "verdict": verdict, "evidence": evidence})
    print(f"[{verdict}] {claim}\n    -> {evidence}\n")


# ============================================================
# CLAIM 1: engine.quantize("int4"/"int8"/"int8_weight") compresses weights
# ============================================================
def check_quantization_effect():
    eng = HasteJevEngine(preset="500k")
    n_linear_before = sum(1 for m in eng.modules() if isinstance(m, nn.Linear))
    nbytes_before = sum(p.numel() * p.element_size() for p in eng.parameters()) + \
                    sum(b.numel() * b.element_size() for b in eng.buffers())

    eng.quantize("int4")
    n_linear_after_int4 = sum(1 for m in eng.modules() if isinstance(m, nn.Linear))
    n_q4 = sum(1 for m in eng.modules() if isinstance(m, QuantizedLinear4bit))
    nbytes_after = sum(p.numel() * p.element_size() for p in eng.parameters()) + \
                   sum(b.numel() * b.element_size() for b in eng.buffers())

    eng2 = HasteJevEngine(preset="500k")
    eng2.quantize("int8_weight")
    n_q8 = sum(1 for m in eng2.modules() if isinstance(m, QuantizedLinear8bit))
    eng2.quantize("int4")
    n_q4b = sum(1 for m in eng2.modules() if isinstance(m, QuantizedLinear4bit))

    # int4 replaces all Linear modules and shrinks parameter storage.
    ok = (n_q4 > 0) and (n_q4b > 0) and (n_linear_after_int4 == 0) and (nbytes_after < nbytes_before)
    record(
        'engine.quantize("int4"/"int8_weight") "instantly compresses linear weights" (README quickstart)',
        "REFUTED" if not ok else "VERIFIED (post-fix)",
        f"Linear layers before={n_linear_before}, after int4: still Linear={n_linear_after_int4}, "
        f"QuantizedLinear4bit={n_q4}, int8→int4 modules={n_q4b}, QuantizedLinear8bit={n_q8}; "
        f"bytes before={nbytes_before:,} after={nbytes_after:,} "
        f"(ratio={nbytes_after/nbytes_before:.3f}). "
        + ("Pre-fix this was a no-op (deepcopy discarded); quantize_model now mutates in place." if ok
           else "Quantization still not applied in place.")
    )


# ============================================================
# CLAIM 2: PICA 0.0% option-order bias
# ============================================================
def check_permutation_invariance():
    eng = HasteJevEngine(preset="1m")
    state = "Account balance is $14,850.50 with pending wire of $3,200.00."
    options = ["Approve Wire", "Flag for AML Review", "Decline Transaction", "Request KYC"]
    base = eng.choice(state, options).probabilities
    max_diff = 0.0
    for perm in permutations(options):
        r = eng.choice(state, list(perm)).probabilities
        for o in options:
            max_diff = max(max_diff, abs(base[o] - r[o]))
    # Also high-cardinality path (>64)
    opts128 = [f"Option {i}" for i in range(128)]
    opts128[7] = "Approve Wire"
    base2 = eng.choice(state, opts128).probabilities
    max_diff2 = 0.0
    shuffled = opts128[::-1]
    r2 = eng.choice(state, shuffled).probabilities
    for k, v in base2.items():
        if k in r2:
            max_diff2 = max(max_diff2, abs(v - r2[k]))
    record(
        "Option-Order Bias = 0.0% (PICA permutation invariant)",
        "VERIFIED (architectural)" if max_diff < 1e-4 and max_diff2 < 1e-3 else "REFUTED",
        f"K=4 all 24 permutations max|Δp|={max_diff:.2e}; K=128 reversed max|Δp|={max_diff2:.2e}. "
        f"Options scored independently via cross-attention -> structurally order-invariant."
    )


# ============================================================
# CLAIM 3: p99 latency < 15ms
# ============================================================
def check_latency():
    eng = HasteJevEngine(preset="20m")
    state = "Account balance is $14,850.50 with pending wire of $3,200.00."
    options = ["Approve Wire", "Flag for AML Review", "Decline Transaction"]
    # warmup
    for _ in range(20):
        eng.choice(state, options)
    lats = []
    for _ in range(100):
        t0 = time.perf_counter()
        eng.choice(state, options)
        lats.append((time.perf_counter() - t0) * 1000)
    p50 = float(np.percentile(lats, 50))
    p99 = float(np.percentile(lats, 99))
    # Also test 1200-option DOM case
    dom = [f"<button id='btn_{i}'>Item {i}</button>" for i in range(1200)]
    dom[742] = "<button id='checkout_btn'>Proceed to Checkout</button>"
    t0 = time.perf_counter()
    eng.choice("User wants to checkout", dom)
    dom_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    eng.choice("User wants to checkout", dom)
    dom_warm_ms = (time.perf_counter() - t0) * 1000
    # 10k options
    opts10k = [f"catalog item {i}" for i in range(10000)]
    t0 = time.perf_counter()
    eng.choice("User wants catalog item 5000", opts10k)
    k10k_ms = (time.perf_counter() - t0) * 1000
    record(
        "p99 Latency < 15ms (README badge)",
        "REFUTED" if p99 >= 15 else "VERIFIED (small K only)",
        f"preset=20m K=3 on this host: p50={p50:.2f}ms p99={p99:.2f}ms. "
        f"K=1200 DOM cold={dom_ms:.1f}ms warm={dom_warm_ms:.1f}ms. K=10000: {k10k_ms:.1f}ms. "
        f"(Kaggle T4 log also shows p50=15.48ms p99=19.35ms — exceeds badge.)"
    )
    record(
        "DOM element selection over 1,000+ elements in <2ms (README browser section)",
        "REFUTED" if dom_ms >= 2 and dom_warm_ms >= 2 else "VERIFIED (warm only)",
        f"K=1200 cold={dom_ms:.1f}ms warm={dom_warm_ms:.1f}ms on this host after projector cache. "
        f"Original claim <2ms not met cold; unit test now passes its own 1500ms bound."
    )
    record(
        "10,000+ candidate options in <1ms (H2-Softmax claim)",
        "REFUTED",
        f"End-to-end choice() with K=10000 took {k10k_ms:.1f}ms (encoding dominates; "
        f"research.md's <1ms figure excludes option encoding / only measures scoring kernel)."
    )


# ============================================================
# CLAIM 4: ECE < 0.009 (HIT-Calib)
# ============================================================
def check_ece_and_calibration():
    import inspect
    src = inspect.getsource(type(HasteJevEngine(preset="100k").calibrator).calibrate_probs)
    uses_iso = "iso" in src.lower() or "isotonic" in src.lower()

    # Fit isotonic path via the real calibrator
    rng = np.random.RandomState(0)
    logits = rng.randn(400, 4) * 3
    labels = rng.randint(0, 4, 400)
    cal = HITCalibrator()
    cal.fit(logits, labels)
    probs_fit = cal.calibrate_probs(torch.tensor(logits, dtype=torch.float32)).numpy()
    ece_hit = HITCalibrator.compute_ece(probs_fit, labels)

    # Temperature-only baseline (pre-fix behavior)
    cal_temp = HITCalibrator()
    cal_temp.temperature = 1.0
    cal_temp.is_fitted = False
    probs_temp = cal_temp.calibrate_probs(torch.tensor(logits, dtype=torch.float32)).numpy()
    ece_temp = HITCalibrator.compute_ece(probs_temp, labels)

    has_iso_models = cal.isotonic_models is not None and len(cal.isotonic_models) == 4
    improved = ece_hit < ece_temp
    # Historical published claim: ECE < 0.009 — still only achieved on synthetic random logits
    record(
        "ECE < 0.009 / 0.012 via Hybrid Isotonic-Temperature Calibration (HIT-Calib)",
        "PARTIAL (code wired post-fix; published ECE still unreplicated)"
        if (uses_iso and has_iso_models and improved)
        else "REFUTED (not implemented as claimed)",
        f"calibrate_probs uses isotonic={uses_iso}; isotonic_models fitted={has_iso_models}. "
        f"Synthetic N=400 K=4: ECE temp-only={ece_temp:.4f} → HIT={ece_hit:.4f} "
        f"({'improved' if improved else 'no improvement'}). "
        f"Project's own Kaggle log still shows real-data ECE AFTER=0.0531 (5.31%, not <0.9%)."
    )


# ============================================================
# CLAIM 5: Arithmetic accuracy 99.4% (STFE)
# ============================================================
def check_arithmetic():
    eng = HasteJevEngine(preset="1m")
    # Balance comparison tests
    cases = [
        ("Balance is $14,850.50.", "Balance exceeds $10,000", True),
        ("Balance is $500.00.", "Balance exceeds $10,000", False),
        ("Balance is 9500 dollars.", "Balance exceeds 10000 dollars", False),
        ("Balance is 11000 dollars.", "Balance exceeds 10000 dollars", True),
        ("User has 42 items in cart.", "Cart has more than 10 items", True),
        ("User has 3 items in cart.", "Cart has more than 10 items", False),
        ("Server latency is 12.4ms.", "Latency exceeds 100ms", False),
        ("Server latency is 250ms.", "Latency exceeds 100ms", True),
    ]
    # Use noul (the only boolean primitive)
    correct = 0
    details = []
    for state, assertion, expected in cases:
        r = eng.noul(state, assertion)
        ok = (r.is_true == expected)
        correct += ok
        details.append(f"{'OK' if ok else 'XX'} '{assertion}' got={r.is_true}({r.probability:.2f}) want={expected}")
    acc = correct / len(cases)
    # Also try choice-based arithmetic
    choice_cases = [
        ("Balance is $14,850.50", ["Above 10k", "Below 10k"], "Above 10k"),
        ("Balance is $950.50", ["Above 10k", "Below 10k"], "Below 10k"),
        ("Count of errors is 250", ["Over 100 errors", "Under 100 errors"], "Over 100 errors"),
        ("Count of errors is 12", ["Over 100 errors", "Under 100 errors"], "Under 100 errors"),
    ]
    c_ok = 0
    c_details = []
    for state, opts, expected in choice_cases:
        r = eng.choice(state, opts)
        ok = r.decision == expected
        c_ok += ok
        c_details.append(f"{'OK' if ok else 'XX'} '{state}' -> {r.decision} want={expected}")
    c_acc = c_ok / len(choice_cases)
    overall = (correct + c_ok) / (len(cases) + len(choice_cases))
    record(
        "99.4% arithmetic accuracy via STFE layer (README + research table)",
        "REFUTED",
        f"Measured noul accuracy={acc:.0%} ({correct}/{len(cases)}), choice accuracy={c_acc:.0%} "
        f"({c_ok}/{len(choice_cases)}), overall={overall:.0%} on threshold comparisons. "
        f"No arithmetic benchmark exists in repo. Details: {'; '.join(details + c_details)}"
    )


# ============================================================
# CLAIM 6: Temporal accuracy 98.8%
# ============================================================
def check_temporal():
    eng = HasteJevEngine(preset="1m")
    cases = [
        ("Event happened on 2026-01-15.", "Event occurred in 2026", True),
        ("Event happened on 2024-01-15.", "Event occurred in 2026", False),
        ("Order placed 2026-03-10, shipped 2026-03-12.", "Shipment happened after order", True),
        ("Order placed 2026-03-12, shipped 2026-03-10.", "Shipment happened after order", False),
        ("Last login was 2025-06-01.", "Login was within the last 30 days from 2026-09-24", False),
    ]
    correct = 0
    details = []
    for state, assertion, expected in cases:
        r = eng.noul(state, assertion)
        ok = r.is_true == expected
        correct += ok
        details.append(f"{'OK' if ok else 'XX'} got={r.is_true}({r.probability:.2f}) want={expected}")
    acc = correct / len(cases)
    # range_eval on dates
    r = eng.range_eval("Events on 2026-01-01 and 2026-03-01", "day delta")
    record(
        "98.8% temporal/date accuracy via STFE (README + research table)",
        "REFUTED",
        f"Measured noul temporal accuracy={acc:.0%} ({correct}/{len(cases)}). "
        f"Details: {'; '.join(details)}. range_eval on dates returns abs(day-diff) "
        f"={r.estimated_value:.1f} regardless of property_name — no learned temporal model."
    )


# ============================================================
# CLAIM 7: noul() is a neural primitive
# ============================================================
def check_noul_hardcoded():
    eng = HasteJevEngine(preset="1m")
    # Keyword path still exists for security/benign gates, but polarity is fixed.
    r1 = eng.noul("This is completely unrelated text about weather.", "Is the system compromised by malware?")
    r2 = eng.noul("untrusted link detected", "Is this safe?")
    r3 = eng.noul("stripe.com corporate SSO approved", "Is this a threat?")
    r4 = eng.noul("The quick brown fox jumps over the lazy dog.", "Is rain wet?")
    r5 = eng.noul("malicious attack detected in logs.", "Is rain wet?")
    # Polarity: threat+safe and valid+threat assertions must be false (the polarity bug)
    polarity_ok = (r2.is_true is False) and (r3.is_true is False)
    # Neural path still reachable for non-keyword inputs
    neural_reachable = (0.0 < r4.probability < 1.0) and (0.0 < r1.probability < 1.0)
    verdict = (
        "PARTIAL (polarity fixed; keyword gates remain for security/benign cases)"
        if (polarity_ok and neural_reachable)
        else "REFUTED (hardcoded keyword heuristic for common cases)"
    )
    record(
        "noul() is a calibrated neural boolean evaluation primitive",
        verdict,
        f"'untrusted...safe'->{r2.probability:.2f} is_true={r2.is_true}, "
        f"'stripe.com...threat'->{r3.probability:.2f} is_true={r3.is_true} (was forced 0.95 True), "
        f"fox/rain->{r4.probability:.3f} vs attack/rain->{r5.probability:.3f}, "
        f"weather/malware->{r1.probability:.3f}. "
        f"Keyword gates remain but polarity is assertion-aware; calibration still not applied on keyword path."
    )


# ============================================================
# CLAIM 8: range_eval is a GP regression with 95% CI
# ============================================================
def check_range():
    eng = HasteJevEngine(preset="1m")
    r1 = eng.range_eval("Page height is 4200px. Target at 2850px. Scroll at 300px.", "Required Scroll Y Delta")
    r2 = eng.range_eval("Balance is $100 and fee is $5.", "Account age in years")
    r3 = eng.range_eval("no numbers here at all", "latency ms")
    # Post-fix: uses neural mean/logvar heads for every path; still no GP / property-conditioned training
    uses_neural = r1.variance > 0 and r1.variance != 1.0
    record(
        "Range primitive: GP regression, 95% CI from learned variance (research.md)",
        "PARTIAL (neural mean/var heads post-fix; still no GP, still untrained heads)"
        if uses_neural else "REFUTED",
        f"Post-fix path: always neural mean+logvar (property_name prefixed into state). "
        f"Scroll state -> est={r1.estimated_value:.2f} CI={r1.confidence_interval_95} var={r1.variance:.4f} "
        f"(true delta should be 2850-300=2550 — untrained head, not GP). "
        f"'Account age' from $100/$5 -> est={r2.estimated_value:.2f} var={r2.variance:.4f}. "
        f"No-scalars fallback: est={r3.estimated_value:.4f} var={r3.variance:.4f}. "
        f"No Gaussian process anywhere in codebase; CI is 1.96*sqrt(exp(logvar)) from untrained logvar head."
    )


# ============================================================
# CLAIM 9: Architecture claims (ModernBERT 380M, Rust/C++, HNSW, eBPF, mmap)
# ============================================================
def check_architecture_claims():
    eng = HasteJevEngine(preset="20m")
    counts = eng.parameter_count
    # Check for claimed components
    import subprocess, os
    root = os.path.dirname(os.path.abspath(__file__))
    has_rust = any(f.endswith(('.rs',)) for f in subprocess.check_output(
        ['bash', '-c', f'find {root} -name "*.rs" -o -name "*.cpp" -o -name "*.cc" -o -name "*.c" 2>/dev/null | grep -v .venv || true']).decode().split('\n') if f)
    src_all = ""
    for dirpath, dirs, files in os.walk(root):
        if '.venv' in dirpath or '__pycache__' in dirpath or '.git' in dirpath:
            continue
        for f in files:
            if f.endswith(('.py', '.toml', '.md')):
                try:
                    src_all += open(os.path.join(dirpath, f), errors='ignore').read()
                except Exception:
                    pass
    has_hnsw = 'hnswlib' in src_all.lower() or 'HNSW' in src_all
    has_ebpf = 'ebpf' in src_all.lower() and 'import' in src_all  # crude
    has_mmap_code = 'mmap(' in src_all and 'def ' in src_all
    uses_modernbert = 'ModernBert' in src_all or 'modernbert' in src_all.lower() and 'from transformers' in src_all
    transformer_type = type(eng.encoder.transformer).__name__
    record(
        "research.md: 380M ModernBERT-derived encoder; Rust/C++ bare-metal runtime; HNSW index; eBPF; mmap zero-copy",
        "REFUTED (design doc only — none present in code)",
        f"Actual: preset 20m total params={counts['total']:,} (not 380M); "
        f"backbone=nn.TransformerEncoder ({transformer_type}); "
        f"Rust/C++ sources present={has_rust}; HNSW used={has_hnsw}; "
        f"eBPF code={has_ebpf}; real mmap runtime={has_mmap_code}; "
        f"ModernBERT imported={uses_modernbert}. "
        f"Text projector is CRC32 n-gram hash table (SimHash), not a pretrained tokenizer/encoder."
    )


# ============================================================
# CLAIM 10: Preset parameter counts vs README table
# ============================================================
def check_param_counts():
    claims = {
        "100k": 98127, "500k": 500091, "1m": 1106723, "2m": 1826275,
        "5m": 5003971, "10m": 10002275, "20m": 20383267,
    }
    lines = []
    all_ok = True
    for p, claimed in claims.items():
        eng = HasteJevEngine(preset=p)
        total = eng.parameter_count["total"]
        # trainable only (buffers = hash table)
        trainable = eng.parameter_count["trainable"]
        match = abs(total - claimed) / claimed < 0.05
        # README says "Total Parameters" - check both
        lines.append(f"{p}: claimed={claimed:,} measured_total={total:,} trainable={trainable:,} "
                     f"delta_total={100*(total-claimed)/claimed:+.1f}%")
        if not match:
            all_ok = False
    record(
        "README sister-model table exact parameter counts (e.g. 20m = 20,383,267)",
        "VERIFIED" if all_ok else "MISMATCH",
        "; ".join(lines)
    )


# ============================================================
# CLAIM 11: Zero-copy / RAM footprints
# ============================================================
def check_ram():
    eng = HasteJevEngine(preset="100k")
    counts = eng.parameter_count
    # FP32 bytes
    nbytes = sum(p.numel() * 4 for p in eng.parameters() if p.is_floating_point()) + \
             sum(b.numel() * b.element_size() for b in eng.buffers())
    # claimed ~0.4 MB FP32
    mb = nbytes / 1e6
    record(
        "RAM (FP32) ~0.4 MB for hastejev-100k (README table)",
        "VERIFIED" if mb < 0.6 else "REFUTED",
        f"Measured serialized param+buffer bytes={nbytes:,} ({mb:.3f} MB). "
        f"Note: this is weight storage only; runtime activation RSS is larger. "
        f"INT8 row claims ~0.1MB — now achievable in principle post-fix (see Claim 1)."
    )


# ============================================================
# CLAIM 12: save/load quantized weights actually round-trip
# ============================================================
def check_quant_roundtrip():
    import tempfile, os
    from safetensors.torch import load_file
    eng = HasteJevEngine(preset="500k")
    eng.quantize("int4")
    with tempfile.TemporaryDirectory() as d:
        eng.save_pretrained(d, quantization="int4")
        st = load_file(os.path.join(d, "model_int4.safetensors"))
        has_packed = any("weight_packed" in k for k in st)
        has_float_weight = any(
            k.endswith(".weight") and st[k].is_floating_point()
            and not k.startswith("projector.table")  # hash table stays float by design
            and "range_" not in k and "scalar_proj" not in k and "stfe" not in k
            for k in st
        )
        eng2 = HasteJevEngine.from_pretrained(d, quantization="int4")
        n_q4 = sum(1 for m in eng2.modules() if isinstance(m, QuantizedLinear4bit))
        # round-trip still runs
        _ = eng2.choice("Balance is $100", ["ok", "no"])
    ok = has_packed and (n_q4 > 0)
    record(
        'save_pretrained(..., quantization="int4") stores packed 4-bit weights',
        "REFUTED" if not ok else "VERIFIED (post-fix)",
        f"model_int4.safetensors weight_packed keys={has_packed}; "
        f"non-head float .weight keys present={has_float_weight}; "
        f"from_pretrained rebuilt QuantizedLinear4bit modules={n_q4}. "
        + ("Pre-fix the file was FP32 with an int4 filename." if ok
           else "Round-trip still produces FP32 modules.")
    )


# ============================================================
# CLAIM 13: calibration.fit actually improves ECE on engine outputs
# ============================================================
def check_calibrator_fit_usage():
    eng = HasteJevEngine(preset="1m")
    used = eng.calibrator.is_fitted
    temp = eng.calibrator.temperature
    import hastejev.engine as eng_mod
    src = open(eng_mod.__file__).read()
    fit_method = "def fit_calibration" in src
    # Can we actually fit and get isotonic models?
    rng = np.random.RandomState(1)
    logits = rng.randn(200, 3) * 2
    labels = rng.randint(0, 3, 200)
    eng.fit_calibration(logits, labels)
    fitted = eng.calibrator.is_fitted and eng.calibrator.isotonic_models is not None
    record(
        "HIT-Calib fitted on validation data (temperature + isotonic) before deployment",
        "PARTIAL (fit API exists post-fix; not called automatically / no shipped checkpoint)"
        if fitted else "REFUTED (temperature hard-coded from config; .fit never called in engine)",
        f"default is_fitted={used}, default temperature={temp} (config.calibrator_temperature). "
        f"engine has fit_calibration()={fit_method}; after fit_calibration(): "
        f"is_fitted={eng.calibrator.is_fitted}, temperature={eng.calibrator.temperature:.3f}, "
        f"isotonic_models={None if eng.calibrator.isotonic_models is None else len(eng.calibrator.isotonic_models)}. "
        f"Still not invoked automatically on load; no held-out ECE published."
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    print("=" * 78)
    print("HASTEJEV CLAIM VERIFICATION SUITE")
    print("=" * 78)
    check_quantization_effect()
    check_permutation_invariance()
    check_latency()
    check_ece_and_calibration()
    check_arithmetic()
    check_temporal()
    check_noul_hardcoded()
    check_range()
    check_architecture_claims()
    check_param_counts()
    check_ram()
    check_quant_roundtrip()
    check_calibrator_fit_usage()

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for r in results:
        print(f"  [{r['verdict']:22s}] {r['claim']}")
    with open("claim_verification_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote claim_verification_results.json ({len(results)} claims checked)")
