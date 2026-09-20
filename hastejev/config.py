from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

@dataclass
class HasteJevConfig:
    """
    Configuration specification for Haste Jev System-1 Decision Engine models.
    Supports predefined presets from 100k (nano) to 20M (base) parameters.
    """
    preset_name: str = "20m"
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: Optional[int] = None
    table_size: int = 65536
    num_frequencies: int = 32
    vocab_size: int = 30522
    calibrator_temperature: float = 1.0
    torch_dtype: str = "float32"
    quantization: Optional[str] = None  # None, 'fp16', 'bf16', 'int8', 'int8_weight', 'int4'
    hastejev_version: str = "1.1.0"

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = self.d_model * 4

    @classmethod
    def from_preset(cls, preset: str, **kwargs) -> "HasteJevConfig":
        """
        Instantiates a HasteJevConfig matching predefined parameter targets.
        Presets: '100k', '500k', '1m', '2m', '5m', '10m', '20m'
        """
        preset_clean = preset.lower().replace("hastejev-", "").replace("hastejev_", "")
        
        PRESETS: Dict[str, Dict[str, Any]] = {
            "100k": {
                "preset_name": "100k",
                "d_model": 48,
                "n_layers": 2,
                "n_heads": 2,
                "table_size": 512,
                "num_frequencies": 12,
            },
            "500k": {
                "preset_name": "500k",
                "d_model": 96,
                "n_layers": 3,
                "n_heads": 4,
                "table_size": 1024,
                "num_frequencies": 24,
            },
            "1m": {
                "preset_name": "1m",
                "d_model": 128,
                "n_layers": 4,
                "n_heads": 4,
                "table_size": 1536,
                "num_frequencies": 32,
            },
            "2m": {
                "preset_name": "2m",
                "d_model": 160,
                "n_layers": 4,
                "n_heads": 4,
                "table_size": 2560,
                "num_frequencies": 32,
            },
            "5m": {
                "preset_name": "5m",
                "d_model": 224,
                "n_layers": 5,
                "n_heads": 4,
                "table_size": 7296,
                "num_frequencies": 32,
            },
            "10m": {
                "preset_name": "10m",
                "d_model": 320,
                "n_layers": 5,
                "n_heads": 4,
                "table_size": 9830,
                "num_frequencies": 32,
            },
            "20m": {
                "preset_name": "20m",
                "d_model": 256,
                "n_layers": 4,
                "n_heads": 4,
                "table_size": 65536,
                "num_frequencies": 32,
            }
        }
        
        if preset_clean not in PRESETS:
            valid_keys = ", ".join(PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset}'. Available presets: {valid_keys}")
            
        cfg_dict = PRESETS[preset_clean].copy()
        cfg_dict.update(kwargs)
        return cls(**cfg_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architectures": ["HasteJevEngine"],
            "model_type": "hastejev",
            "preset_name": self.preset_name,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "table_size": self.table_size,
            "num_frequencies": self.num_frequencies,
            "vocab_size": self.vocab_size,
            "calibrator_temperature": self.calibrator_temperature,
            "torch_dtype": self.torch_dtype,
            "quantization": self.quantization,
            "hastejev_version": self.hastejev_version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HasteJevConfig":
        return cls(
            preset_name=data.get("preset_name", "custom"),
            d_model=data.get("d_model", 256),
            n_layers=data.get("n_layers", 4),
            n_heads=data.get("n_heads", 4),
            d_ff=data.get("d_ff", None),
            table_size=data.get("table_size", 65536),
            num_frequencies=data.get("num_frequencies", 32),
            vocab_size=data.get("vocab_size", 30522),
            calibrator_temperature=data.get("calibrator_temperature", 1.0),
            torch_dtype=data.get("torch_dtype", "float32"),
            quantization=data.get("quantization", None),
            hastejev_version=data.get("hastejev_version", "1.1.0")
        )
