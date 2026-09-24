import math
import re
import zlib
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class FastSubwordProjector(nn.Module):
    """
    Subword and Character N-Gram Latent Hash Projector.
    Maps words and textual phrases into continuous semantic embeddings via
    deterministic character 3-gram, 4-gram, and whole-word trigonometric frequency hashing.
    Enables instant zero-shot semantic matching without heavy pretraining overhead.
    """
    _CACHE_MAX = 8192

    def __init__(self, d_model: int = 256, table_size: int = 65536):
        super().__init__()
        self.d_model = d_model
        self.table_size = table_size
        self._latent_cache = {}

        # Precompute deterministic Gaussian random projection table (SimHash)
        g = torch.Generator().manual_seed(42)
        table = torch.randn(table_size, d_model, generator=g)
        table = F.normalize(table, p=2, dim=-1)
        self.register_buffer("table", table)

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        nn.init.eye_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)

    @staticmethod
    def _clean_words(text: str) -> List[str]:
        return re.findall(r'[a-zA-Z0-9_\-\$]+', text.lower())

    def _ngram_hashes(self, text: str) -> List[int]:
        words = self._clean_words(text)
        if not words:
            return []
        hashes: List[int] = []
        append = hashes.append
        crc32 = zlib.crc32
        mod = self.table_size
        for w in words[:32]:
            wb = w.encode("utf-8")
            append(crc32(b"<" + wb + b">") % mod)
            append(crc32(wb) % mod)
            length = len(w)
            for n in (3, 4):
                if length < n:
                    continue
                for i in range(length - n + 1):
                    append(crc32(w[i:i + n].encode("utf-8")) % mod)
        return hashes

    def _embed_texts(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Vectorized hash-table gather + segment-sum for a batch of texts (one vector per text)."""
        all_hashes: List[int] = []
        counts: List[int] = []
        for text in texts:
            h = self._ngram_hashes(text)
            counts.append(len(h))
            all_hashes.extend(h)

        batch = len(texts)
        if not all_hashes:
            return torch.zeros((1, batch, self.d_model), device=device, dtype=self.table.dtype)

        table = self.table.to(device=device)
        idx = torch.tensor(all_hashes, dtype=torch.long, device=device)
        vecs = table[idx]

        counts_t = torch.tensor(counts, dtype=torch.long, device=device)
        seg_ids = torch.repeat_interleave(
            torch.arange(batch, device=device, dtype=torch.long),
            counts_t,
        )
        summed = torch.zeros((batch, self.d_model), device=device, dtype=vecs.dtype)
        summed.index_add_(0, seg_ids, vecs)

        norms = summed.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        summed = summed / norms
        summed[counts_t == 0] = 0.0
        return self.proj(summed.unsqueeze(0))

    def _cache_put(self, key: str, value: torch.Tensor) -> None:
        if len(self._latent_cache) >= self._CACHE_MAX:
            self._latent_cache.clear()
        self._latent_cache[key] = value.detach().to("cpu").reshape(-1)

    def text_to_latent(self, text: str, device: torch.device) -> torch.Tensor:
        """Per-token sequence embedding: one row per word (for the transformer backbone)."""
        words = self._clean_words(text)
        if not words:
            return torch.zeros((1, 1, self.d_model), device=device, dtype=self.table.dtype)

        table = self.table.to(device=device)
        word_vectors = []
        for w in words[:32]:
            wb = w.encode("utf-8")
            hs = [zlib.crc32(b"<" + wb + b">") % self.table_size, zlib.crc32(wb) % self.table_size]
            length = len(w)
            for n in (3, 4):
                if length < n:
                    continue
                for i in range(length - n + 1):
                    hs.append(zlib.crc32(w[i:i + n].encode("utf-8")) % self.table_size)
            vec = table[hs].sum(dim=0)
            norm = torch.norm(vec, p=2)
            if norm > 1e-6:
                vec = vec / norm
            word_vectors.append(vec)

        stacked = torch.stack(word_vectors, dim=0).unsqueeze(0)
        return self.proj(stacked)

    def batch_text_to_latent(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """One pooled vector per text (H2-Softmax option path). Uses a CPU latent cache."""
        if not texts:
            return torch.zeros((1, 0, self.d_model), device=device, dtype=self.table.dtype)

        results: List[Optional[torch.Tensor]] = [None] * len(texts)
        pending_idx: List[int] = []
        pending_texts: List[str] = []
        for i, t in enumerate(texts):
            cached = self._latent_cache.get(t)
            if cached is not None:
                results[i] = cached.reshape(-1)
            else:
                pending_idx.append(i)
                pending_texts.append(t)

        if pending_texts:
            computed = self._embed_texts(pending_texts, device).squeeze(0)
            for i, t, vec in zip(pending_idx, pending_texts, computed):
                self._cache_put(t, vec)
                results[i] = vec.detach().reshape(-1)

        device_vecs = [r.to(device=device) for r in results if r is not None]
        stacked = torch.stack(device_vecs, dim=0)
        return stacked.unsqueeze(0)







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
        x_flat = x.view(-1, 1).to(dtype=self.B.dtype)
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
        for p in self.mha.parameters():
            p.data.mul_(0.01)
        for p in self.score_proj.parameters():
            p.data.mul_(0.01)

    def forward(self, state_rep: torch.Tensor, option_reps: torch.Tensor) -> torch.Tensor:

        attn_out, _ = self.mha(query=option_reps, key=state_rep, value=state_rep)
        state_mean = state_rep.mean(dim=1, keepdim=True)
        sim = torch.bmm(option_reps, state_mean.transpose(1, 2)).squeeze(-1)
        fused = self.norm(option_reps + attn_out)
        scores = self.score_proj(fused).squeeze(-1) + (sim * 4.0)
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
        res_vec = self.residual_vector.to(device=state_rep.device, dtype=state_rep.dtype)
        opts_with_residual = torch.cat([gathered_opts, res_vec], dim=1)
        exact_logits = self.pica(state_rep, opts_with_residual)
        exact_probs = F.softmax(exact_logits, dim=-1)
        
        candidate_probs = exact_probs[:, :M]
        residual_prob = exact_probs[:, M:]
        
        return top_indices.squeeze(0), candidate_probs.squeeze(0), residual_prob.squeeze()
