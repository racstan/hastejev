import pytest
import os
import shutil
import tempfile
import torch
import numpy as np

from hastejev import HasteJevEngine, HasteJevConfig, quantize_model

class TestPresetsAndQuantization:
    
    @pytest.mark.parametrize("preset,expected_min_params,expected_max_params", [
        ("100k", 70_000, 150_000),
        ("500k", 400_000, 600_000),
        ("1m",   900_000, 1_300_000),
        ("2m",   1_700_000, 2_500_000),
        ("5m",   4_500_000, 6_000_000),
        ("10m",  9_000_000, 13_500_000),
        ("20m",  18_000_000, 22_000_000),
    ])
    def test_preset_instantiation_and_parameter_counts(self, preset, expected_min_params, expected_max_params):
        engine = HasteJevEngine(preset=preset)
        counts = engine.parameter_count
        total = counts["total"]
        
        assert expected_min_params <= total <= expected_max_params, (
            f"Preset {preset} total params {total:,} outside expected range [{expected_min_params:,}, {expected_max_params:,}]"
        )
        
        # Test basic execution on each preset
        state = "User clicked on the settings button at 2026-03-15."
        options = ["Open Settings", "Close Modal", "Navigate Home"]
        res = engine.choice(state, options)
        assert res.decision in options
        assert 0.0 <= res.confidence <= 1.0

    def test_pica_order_invariance_across_presets(self):
        presets = ["100k", "500k", "1m", "5m"]
        state = "Transaction value is $4,500.00 submitted from IP 192.168.1.1."
        options = ["Approve", "Review", "Decline"]
        options_perm = ["Decline", "Approve", "Review"]
        
        for p in presets:
            engine = HasteJevEngine(preset=p)
            res1 = engine.choice(state, options)
            res2 = engine.choice(state, options_perm)
            
            for opt in options:
                p1 = res1.probabilities[opt]
                p2 = res2.probabilities[opt]
                assert abs(p1 - p2) < 1e-4, f"Permutation variance detected in preset {p} for option {opt}: {p1} vs {p2}"

    @pytest.mark.parametrize("quant_mode", ["fp16", "bf16", "int8", "int8_weight", "int4"])
    def test_quantization_modes(self, quant_mode):
        engine = HasteJevEngine(preset="500k")
        engine.quantize(quant_mode)
        
        state = "The server CPU usage is 94.2% on host prod-east-1."
        options = ["Trigger Alert", "Ignore Alert", "Restart Service"]
        
        # Choice primitive
        res_choice = engine.choice(state, options)
        assert res_choice.decision in options
        
        # Noul primitive
        res_noul = engine.noul(state, "CPU usage exceeds 90%")
        assert isinstance(res_noul.is_true, (bool, np.bool_))
        
        # Range primitive
        res_range = engine.range_eval(state, "CPU Metric")
        assert res_range.estimated_value is not None

    def test_save_and_load_quantized_model(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            # Create a 1M model and quantize with int8_weight
            engine = HasteJevEngine(preset="1m")
            engine.quantize("int8_weight")

            state = "Security audit log: unauthorized SSH attempt from 10.0.0.42"
            options = ["Block IP", "Allow Session", "Log Warning"]
            res_before = engine.choice(state, options)

            # Save pretrained with quantization
            engine.save_pretrained(tmp_dir, quantization="int8_weight")

            # Load back
            loaded = HasteJevEngine.from_pretrained(tmp_dir, quantization="int8_weight")
            res_after = loaded.choice(state, options)

            assert loaded.config.preset_name == "1m"
            assert res_before.decision == res_after.decision
            assert abs(res_before.probabilities[res_before.decision] - res_after.probabilities[res_after.decision]) < 1e-3
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @pytest.mark.parametrize("quant_mode", ["int8_weight", "int4"])
    def test_quantized_roundtrip_device_consistency(self, quant_mode):
        """Quantized buffers and activations must share one device (regression: CUDA load)."""
        tmp_dir = tempfile.mkdtemp()
        try:
            engine = HasteJevEngine(preset="100k")
            engine.save_pretrained(tmp_dir, quantization=quant_mode)

            for device in [torch.device("cpu")] + (
                [torch.device("cuda")] if torch.cuda.is_available() else []
            ):
                loaded = HasteJevEngine.from_pretrained(tmp_dir, quantization=quant_mode, device=device)
                param_devices = {p.device.type for p in loaded.parameters()}
                buffer_devices = {b.device.type for b in loaded.buffers()}
                assert param_devices == {device.type}, f"params on {param_devices} expected {device.type}"
                assert buffer_devices == {device.type}, f"buffers on {buffer_devices} expected {device.type}"

                res = loaded.choice(
                    "Route chat: cannot push to protected branch",
                    ["billing_agent", "tech_support", "security_review"],
                )
                assert res.decision in (
                    "billing_agent",
                    "tech_support",
                    "security_review",
                )
                assert res.probabilities
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
