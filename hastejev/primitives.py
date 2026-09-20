from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class ChoiceResult:
    primitive: str = "Choice"
    decision: str = ""
    index: int = 0
    probabilities: Dict[str, float] = None
    confidence: float = 0.0
    mode: str = "PICA"
    residual_mass: Optional[float] = None

@dataclass
class ScoreResult:
    primitive: str = "Score"
    expectation_score: float = 0.0
    min_tier: int = 1
    max_tier: int = 5
    distribution: Dict[str, float] = None

@dataclass
class NoulResult:
    primitive: str = "Noul"
    assertion: str = ""
    is_true: bool = False
    probability: float = 0.0
    confidence: float = 0.0

@dataclass
class RangeResult:
    primitive: str = "Range"
    property: str = ""
    estimated_value: float = 0.0
    confidence_interval_95: List[float] = None
    variance: float = 0.0

@dataclass
class SetChoiceResult:
    primitive: str = "SetChoice"
    selected_subset: List[str] = None
    marginal_probabilities: Dict[str, float] = None
