import math
import re
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class STFELayer(nn.Module):
    """
    Scalar and Temporal Fourier Embedding (STFE) Layer.
    Maps continuous scalars and temporal offsets into latent d_model representations.
    """
    def __init__(self, d_model: int = 256, num_frequencies: int = 32):
        super().__init__()
        self.d_model = d_model
        self.num_frequencies = num_frequencies
        self.B = nn.Parameter(torch.randn(num_frequencies, 1) * 0.5)
        fourier_dim = num_frequencies * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        batch_size, seq_len = x.shape
        x_flat = x.view(-1, 1)
        projected = torch.matmul(x_flat, self.B.T) * math.pi
        
        sin_feats = torch.sin(projected)
        cos_feats = torch.cos(projected)
        fourier_feats = torch.cat([sin_feats, cos_feats], dim=-1)
        
        embeddings = self.mlp(fourier_feats)
        return embeddings.view(batch_size, seq_len, self.d_model)

class ScalarTemporalParser:
    """
    Extracts continuous scalar numbers and ISO-8601 timestamps from state payloads.
    """
    EPOCH_BASE = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
    
    @classmethod
    def parse_text_entities(cls, text: str) -> Tuple[str, List[float]]:
        scalars = []
        iso_pattern = r'\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?\b'
        
        def replace_iso(match):
            date_str = match.group(0)
            try:
                if 'T' in date_str:
                    dt = datetime.datetime.fromisoformat(date_str)
                else:
                    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
                offset = (dt.timestamp() - cls.EPOCH_BASE) / 86400.0
                scalars.append(float(offset))
                return f"<STFE_TEMP_{len(scalars)-1}>"
            except Exception:
                return date_str
        
        text = re.sub(iso_pattern, replace_iso, text)
        num_pattern = r'(?<![\w<])[-+]?\d+(?:\.\d+)?(?![\w>])'
        
        def replace_num(match):
            val = float(match.group(0))
            scalars.append(val)
            return f"<STFE_NUM_{len(scalars)-1}>"
            
        text = re.sub(num_pattern, replace_num, text)
        return text, scalars

class PICAHead(nn.Module):
    """
    Permutation-Invariant Cross-Attention (PICA) Head.
    Scores each candidate option independently against state context to eliminate option-order bias.
    """
    def __init__(self, d_model: int = 256, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.score_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, state_rep: torch.Tensor, option_reps: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mha(query=option_reps, key=state_rep, value=state_rep)
        fused = self.norm(option_reps + attn_out)
        scores = self.score_proj(fused).squeeze(-1)
        return scores

class H2SoftmaxEngine:
    """
    Hierarchical Two-Stage Vector Softmax for scaling candidate sets past 10,000+ options.
    """
    def __init__(self, pica_head: PICAHead, d_model: int = 256, default_top_m: int = 32, device: Optional[torch.device] = None):
        self.pica = pica_head
        self.d_model = d_model
        self.default_top_m = default_top_m
        dev = device or torch.device('cpu')
        self.residual_vector = nn.Parameter(torch.randn(1, 1, d_model, device=dev) * 0.1)

    def evaluate_high_cardinality(
        self, 
        state_rep: torch.Tensor, 
        all_option_reps: torch.Tensor,
        top_m: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        M = top_m or self.default_top_m
        K = all_option_reps.size(1)
        
        if K <= M:
            logits = self.pica(state_rep, all_option_reps)
            probs = F.softmax(logits, dim=-1)
            indices = torch.arange(K, device=state_rep.device)
            return indices, probs, torch.tensor(0.0, device=state_rep.device)

        # Stage 1: Dense Latent Filtering
        state_pool = state_rep.mean(dim=1, keepdim=True)
        coarse_scores = torch.bmm(all_option_reps, state_pool.transpose(1, 2)).squeeze(-1)
        _, top_indices = torch.topk(coarse_scores, k=M, dim=-1)
        
        gathered_opts = torch.gather(
            all_option_reps, 1, top_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        
        # Stage 2: Exact PICA Scoring with dynamic residual tier
        opts_with_residual = torch.cat([gathered_opts, self.residual_vector.to(state_rep.device)], dim=1)
        exact_logits = self.pica(state_rep, opts_with_residual)
        exact_probs = F.softmax(exact_logits, dim=-1)
        
        candidate_probs = exact_probs[:, :M]
        residual_prob = exact_probs[:, M:]
        
        return top_indices.squeeze(0), candidate_probs.squeeze(0), residual_prob.squeeze()
