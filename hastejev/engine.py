import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from hastejev.layers import STFELayer, ScalarTemporalParser, PICAHead, H2SoftmaxEngine
from hastejev.calibration import HITCalibrator
from hastejev.primitives import ChoiceResult, ScoreResult, NoulResult, RangeResult, SetChoiceResult

class BidirectionalEncoderBackbone(nn.Module):
    """
    ModernBERT-inspired Bidirectional Transformer Encoder backbone.
    Processes textual tokens and fuses STFE continuous embeddings.
    """
    def __init__(self, vocab_size: int = 30522, d_model: int = 256, n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.stfe = STFELayer(d_model=d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, scalars: Optional[torch.Tensor] = None) -> torch.Tensor:
        tok_emb = self.token_embeddings(input_ids)
        if scalars is not None and scalars.size(1) > 0:
            stfe_emb = self.stfe(scalars)
            tok_emb = torch.cat([tok_emb, stfe_emb], dim=1)
            
        hidden = self.transformer(tok_emb)
        return self.norm(hidden)

class HasteJevEngine(nn.Module):
    """
    hastejev: Ultra-Low-Latency, Zero-Copy System-1 AI Decision Engine.
    
    Provides 5 native decision primitives:
    - choice(state, options): Categorical classification with entropy confidence
    - score(state, rubric_levels): Ordinal expectation scoring across rubric tiers
    - noul(state, assertion): Calibrated boolean truth evaluation
    - range_eval(state, property_name): Continuous scalar regression with 95% confidence interval
    - set_choice(state, options, threshold): Multi-label combinatorial subset selection
    """
    def __init__(self, d_model: int = 256, device: Optional[torch.device] = None):
        super().__init__()
        self.d_model = d_model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder = BidirectionalEncoderBackbone(d_model=d_model).to(self.device)
        self.pica = PICAHead(d_model=d_model).to(self.device)
        self.h2_softmax = H2SoftmaxEngine(self.pica, d_model=d_model, device=self.device)
        self.calibrator = HITCalibrator()
        
        self.range_mean = nn.Linear(d_model, 1).to(self.device)
        self.range_logvar = nn.Linear(d_model, 1).to(self.device)
        self.vocab_size = 30522
        
        # Inference-only engine: disable dropout for deterministic, permutation-invariant outputs
        self.eval()

    def _tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        clean_text, scalars = ScalarTemporalParser.parse_text_entities(text)
        tokens = [abs(hash(w)) % self.vocab_size for w in clean_text.split()][:128]
        if not tokens:
            tokens = [0]
        input_ids = torch.tensor([tokens], device=self.device, dtype=torch.long)
        
        if scalars:
            scalar_tensor = torch.tensor([scalars], device=self.device, dtype=torch.float32)
        else:
            scalar_tensor = torch.zeros((1, 0), device=self.device, dtype=torch.float32)
            
        return input_ids, scalar_tensor

    def encode_text(self, text: str) -> torch.Tensor:
        input_ids, scalars = self._tokenize(text)
        with torch.no_grad():
            return self.encoder(input_ids, scalars)

    def choice(self, state: str, options: List[str]) -> ChoiceResult:
        with torch.no_grad():
            state_rep = self.encode_text(state)
            opt_reps = torch.cat([self.encode_text(opt).mean(dim=1, keepdim=True) for opt in options], dim=1)
            
            if len(options) > 64:
                indices, probs, res_prob = self.h2_softmax.evaluate_high_cardinality(state_rep, opt_reps)
                probs = self.calibrator.calibrate_probs(probs.unsqueeze(0)).squeeze(0)
                sel_cands = [options[i] for i in indices.cpu().numpy()]
                p_list = probs.cpu().numpy().tolist()
                best_idx = int(np.argmax(p_list))
                return ChoiceResult(
                    primitive="Choice",
                    decision=sel_cands[best_idx],
                    index=best_idx,
                    probabilities={sel_cands[i]: float(p_list[i]) for i in range(len(sel_cands))},
                    confidence=float(1.0 - (res_prob.item() if res_prob is not None else 0.0)),
                    mode="H2-Softmax",
                    residual_mass=float(res_prob.cpu().numpy())
                )
            
            logits = self.pica(state_rep, opt_reps)
            probs = self.calibrator.calibrate_probs(logits).squeeze(0)
            
            K = len(options)
            p_np = probs.detach().cpu().numpy()
            entropy = -np.sum(p_np * np.log(np.clip(p_np, 1e-12, 1.0)))
            max_entropy = np.log(K) if K > 1 else 1.0
            confidence = float(1.0 - (entropy / max_entropy))
            
            best_idx = int(torch.argmax(probs).item())
            return ChoiceResult(
                primitive="Choice",
                decision=options[best_idx],
                index=best_idx,
                probabilities={options[i]: float(p_np[i]) for i in range(K)},
                confidence=confidence,
                mode="PICA"
            )

    def score(self, state: str, rubric_levels: List[str]) -> ScoreResult:
        with torch.no_grad():
            res = self.choice(state, rubric_levels)
            probs = np.array(list(res.probabilities.values()))
            N = len(rubric_levels)
            tier_values = np.arange(1, N + 1)
            expectation = float(np.sum(tier_values * probs))
            return ScoreResult(
                primitive="Score",
                expectation_score=expectation,
                min_tier=1,
                max_tier=N,
                distribution=res.probabilities
            )

    def noul(self, state: str, assertion: str) -> NoulResult:
        with torch.no_grad():
            res = self.choice(f"Context: {state} | Claim: {assertion}", ["Refute / False", "Verify / True"])
            true_prob = res.probabilities["Verify / True"]
            return NoulResult(
                primitive="Noul",
                assertion=assertion,
                is_true=true_prob >= 0.5,
                probability=true_prob,
                confidence=res.confidence
            )

    def range_eval(self, state: str, property_name: str) -> RangeResult:
        with torch.no_grad():
            state_rep = self.encode_text(f"{property_name}: {state}").mean(dim=1)
            mean = self.range_mean(state_rep).item()
            var = torch.exp(self.range_logvar(state_rep)).item()
            std = math.sqrt(var)
            return RangeResult(
                primitive="Range",
                property=property_name,
                estimated_value=mean,
                confidence_interval_95=[mean - 1.96 * std, mean + 1.96 * std],
                variance=var
            )

    def set_choice(self, state: str, options: List[str], threshold: float = 0.5) -> SetChoiceResult:
        with torch.no_grad():
            state_rep = self.encode_text(state)
            opt_reps = torch.cat([self.encode_text(opt).mean(dim=1, keepdim=True) for opt in options], dim=1)
            logits = self.pica(state_rep, opt_reps).squeeze(0)
            marginal_probs = torch.sigmoid(logits).detach().cpu().numpy()
            
            selected = [options[i] for i in range(len(options)) if marginal_probs[i] >= threshold]
            return SetChoiceResult(
                primitive="SetChoice",
                selected_subset=selected,
                marginal_probabilities={options[i]: float(marginal_probs[i]) for i in range(len(options))}
            )

    def save_pretrained(self, save_directory: str):
        """
        Exports the Haste Jev model in industry-standard format (config.json + model.safetensors + pytorch_model.bin).
        """
        import os
        import json
        from safetensors.torch import save_file

        os.makedirs(save_directory, exist_ok=True)
        config = {
            "architectures": ["HasteJevEngine"],
            "model_type": "hastejev",
            "d_model": self.d_model,
            "vocab_size": self.vocab_size,
            "calibrator_temperature": self.calibrator.temperature,
            "torch_dtype": "float32",
            "hastejev_version": "1.0.0"
        }
        
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            
        weights_path_safetensors = os.path.join(save_directory, "model.safetensors")
        save_file(self.state_dict(), weights_path_safetensors)
        
        weights_path_bin = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), weights_path_bin)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, device: Optional[torch.device] = None) -> "HasteJevEngine":
        """
        Loads a Haste Jev model from a local directory or the Hugging Face Hub (e.g. 'noffy/hastejev').
        """
        import os
        import json
        
        model_dir = pretrained_model_name_or_path
        if not os.path.isdir(pretrained_model_name_or_path):
            try:
                from huggingface_hub import snapshot_download
                model_dir = snapshot_download(repo_id=pretrained_model_name_or_path)
            except Exception as e:
                raise ValueError(f"Could not find local directory or download from Hugging Face Hub '{pretrained_model_name_or_path}': {e}")
                
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {}
            
        d_model = config.get("d_model", 256)
        instance = cls(d_model=d_model, device=device)
        
        safetensors_path = os.path.join(model_dir, "model.safetensors")
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path, device=str(instance.device))
            instance.load_state_dict(state_dict)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location=instance.device)
            instance.load_state_dict(state_dict)
            
        if "calibrator_temperature" in config:
            instance.calibrator.temperature = config["calibrator_temperature"]
            instance.calibrator.is_fitted = True
            
        instance.eval()
        return instance

