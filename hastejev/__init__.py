"""
Haste Jev: Non-Generative System-1 AI Decision Engine.
Zero-latency, zero-bias, open-weights decision primitive framework.
"""

from hastejev.engine import HasteJevEngine
from hastejev.config import HasteJevConfig
from hastejev.quantization import quantize_model, QuantizedLinear8bit, QuantizedLinear4bit
from hastejev.layers import (
    STFELayer,
    PICAHead,
    H2SoftmaxEngine,
    ScalarTemporalParser,
    FastSubwordProjector,
)
from hastejev.primitives import (
    ChoiceResult,
    ScoreResult,
    NoulResult,
    RangeResult,
    SetChoiceResult,
)
from hastejev.calibration import HITCalibrator

__version__ = "1.1.0"

__all__ = [
    "HasteJevEngine",
    "HasteJevConfig",
    "quantize_model",
    "QuantizedLinear8bit",
    "QuantizedLinear4bit",
    "STFELayer",
    "PICAHead",
    "H2SoftmaxEngine",
    "ScalarTemporalParser",
    "FastSubwordProjector",
    "ChoiceResult",
    "ScoreResult",
    "NoulResult",
    "RangeResult",
    "SetChoiceResult",
    "HITCalibrator",
]
