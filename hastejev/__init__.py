"""
hastejev: Ultra-Low-Latency, Zero-Copy System-1 AI Decision Engine
"""

from hastejev.engine import HasteJevEngine
from hastejev.layers import STFELayer, PICAHead, H2SoftmaxEngine, ScalarTemporalParser
from hastejev.calibration import HITCalibrator
from hastejev.primitives import ChoiceResult, ScoreResult, NoulResult, RangeResult, SetChoiceResult

__version__ = "0.1.0"
__all__ = [
    "HasteJevEngine",
    "STFELayer",
    "PICAHead",
    "H2SoftmaxEngine",
    "ScalarTemporalParser",
    "HITCalibrator",
    "ChoiceResult",
    "ScoreResult",
    "NoulResult",
    "RangeResult",
    "SetChoiceResult",
]
