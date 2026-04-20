"""FourierAttention — Fourier-generated Q/K/V projection weights.

Key insight: In multi-head attention, the Q/K/V projection matrices
(W_q, W_k, W_v) are (d_model, d_model) each — the bulk of transformer params.
If we can generate these from Fourier coefficients, we get massive compression.

Two modes:
1. FourierInitAttention — init Q/K/V from Fourier, then train normally
2. FourierWeightAttention — generate Q/K/V on-the-fly (extreme compression)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierInitAttention(nn.MultiheadAttention):
    """Multi-head attention with Fourier-initialized Q/K/V projections.
    
    Same API as nn.MultiheadAttention but with structured Fourier initialization
    for the in_proj_weight (Q, K, V concatenated).
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        k: int = 64,
        n_bands: int = 3,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__(embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=True)
        self._fourier_init_proj(k, n_bands)
    
    def _fourier_init_proj(self, k: int, n_bands: int):
        """Initialize the in_proj_weight (3*embed_dim, embed_dim) from Fourier."""
        with torch.no_grad():
            total_k = k * n_bands
            d = self.embed_dim
            # in_proj_weight shape: (3*d, d) for Q, K, V concatenated
            rows, cols = 3 * d, d
            n_weights = rows * cols
            
            # Multi-scale Fourier frequencies
            omega = torch.zeros(total_k, 3)
            phi = torch.zeros(total_k)
            for band in range(n_bands):
                s, e = band * k, (band + 1) * k
                omega[s:e] = torch.randn(k, 3) * (1 + band * 3)
                phi[s:e] = torch.randn(k) * math.pi
            
            # Position grid
            i_idx = (torch.arange(rows, dtype=torch.float32) + 1) / (rows + 1)
            j_idx = (torch.arange(cols, dtype=torch.float32) + 1) / (cols + 1)
            ii, jj = torch.meshgrid(i_idx, j_idx, indexing='ij')
            # l_val encodes "attention layer" — use 0.5 as default
            l_val = 0.5
            pos = torch.stack([ii, jj, torch.full_like(ii, l_val)], dim=-1).reshape(-1, 3)
            
            # Fourier features
            args = pos @ omega.T + phi
            features = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
            alpha = torch.randn(2 * total_k) * 0.01 / (1 + torch.arange(2 * total_k).float())
            
            # Generate and scale
            w = (features @ alpha).reshape(rows, cols)
            w *= math.sqrt(2.0 / (rows + cols))
            
            self.in_proj_weight.copy_(w)


class FourierWeightAttention(nn.Module):
    """Multi-head attention where Q/K/V weights are generated on-the-fly.
    
    Instead of storing 3*d^2 parameters for Q/K/V projections, we store
    only 2*total_k Fourier coefficients per projection (6*total_k total).
    
    For d=256, k=32, n_bands=2:
    - Standard: 3 * 256^2 = 196,608 params
    - Fourier:  6 * 128 = 768 params (256x compression!)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        k: int = 32,
        n_bands: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        
        total_k = k * n_bands
        
        # Per-projection Fourier coefficients (Q, K, V)
        self.alpha_q = nn.Parameter(torch.randn(2 * total_k) * 0.01 / (1 + torch.arange(2 * total_k).float()))
        self.alpha_k = nn.Parameter(torch.randn(2 * total_k) * 0.01 / (1 + torch.arange(2 * total_k).float()))
        self.alpha_v = nn.Parameter(torch.randn(2 * total_k) * 0.01 / (1 + torch.arange(2 * total_k).float()))
        
        # Biases
        self.bias_q = nn.Parameter(torch.zeros(embed_dim))
        self.bias_k = nn.Parameter(torch.zeros(embed_dim))
        self.bias_v = nn.Parameter(torch.zeros(embed_dim))
        
        # Output projection (standard — small relative to Q/K/V)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Frozen Fourier frequencies (shared across Q/K/V for consistency)
        omega = torch.zeros(total_k, 3)
        phi = torch.zeros(total_k)
        for band in range(n_bands):
            s, e = band * k, (band + 1) * k
            omega[s:e] = torch.randn(k, 3) * (1 + band * 3)
            phi[s:e] = torch.randn(k) * math.pi
        
        # Precompute features for Q/K/V (same shape since all are d×d)
        i_idx = (torch.arange(embed_dim, dtype=torch.float32) + 1) / (embed_dim + 1)
        j_idx = (torch.arange(embed_dim, dtype=torch.float32) + 1) / (embed_dim + 1)
        ii, jj = torch.meshgrid(i_idx, j_idx, indexing='ij')
        l_val = 0.5
        pos = torch.stack([ii, jj, torch.full_like(ii, l_val)], dim=-1).reshape(-1, 3)
        
        args = pos @ omega.T + phi
        features = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (d*d, 2*total_k)
        self.register_buffer('features', features)
        self._proj_scale = math.sqrt(2.0 / (2 * embed_dim))
    
    def _gen_proj(self, alpha: torch.Tensor) -> torch.Tensor:
        """Generate a (embed_dim, embed_dim) projection matrix from Fourier alpha."""
        w = (self.features @ alpha).reshape(self.embed_dim, self.embed_dim) * self._proj_scale
        return w
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            (batch, seq_len, embed_dim)
        """
        B, S, D = x.shape
        
        # Generate Q/K/V projection weights
        W_q = self._gen_proj(self.alpha_q)
        W_k = self._gen_proj(self.alpha_k)
        W_v = self._gen_proj(self.alpha_v)
        
        # Project
        Q = F.linear(x, W_q, self.bias_q)  # (B, S, D)
        K = F.linear(x, W_k, self.bias_k)
        V = F.linear(x, W_v, self.bias_v)
        
        # Reshape to heads
        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, d_h)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, S, S)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply to values
        out = attn @ V  # (B, H, S, d_h)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        
        return self.out_proj(out)


class FourierTransformerBlock(nn.Module):
    """Transformer block with Fourier-generated attention weights.
    
    Standard transformer block:
    x → FourierAttention → Add → LayerNorm → FFN → Add → LayerNorm
    
    Args:
        embed_dim: Model dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward hidden dimension
        k: Fourier components per band
        n_bands: Number of frequency bands
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int = None,
        k: int = 32,
        n_bands: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        ff_dim = ff_dim or 4 * embed_dim
        
        self.attn = FourierWeightAttention(embed_dim, num_heads, k, n_bands, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class FourierTransformer(nn.Module):
    """Full transformer with Fourier-generated attention weights.
    
    All attention layers use FourierWeightAttention for Q/K/V projections.
    FFN layers use standard Linear (could be Fourier too — future work).
    
    Args:
        embed_dim: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        ff_dim: Feed-forward hidden dimension
        max_seq_len: Maximum sequence length
        num_classes: Number of output classes (0 = return embeddings)
        k: Fourier components per band
        n_bands: Number of frequency bands
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = None,
        max_seq_len: int = 256,
        num_classes: int = 0,
        k: int = 32,
        n_bands: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Token embedding + positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            FourierTransformerBlock(embed_dim, num_heads, ff_dim, k, n_bands, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        if num_classes > 0:
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.head = None
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim) — pre-embedded tokens
        Returns:
            If num_classes > 0: (batch, num_classes) logits
            Else: (batch, seq_len, embed_dim) embeddings
        """
        B, S, D = x.shape
        x = x + self.pos_embed[:, :S, :]
        
        for block in self.blocks:
            x = block(x, attn_mask)
        
        x = self.norm(x)
        
        if self.head is not None:
            # Global average pooling → classify
            x = x.mean(dim=1)
            return self.head(x)
        return x
    
    @property
    def fourier_params(self) -> int:
        """Count of Fourier-specific trainable parameters."""
        count = 0
        for block in self.blocks:
            attn = block.attn
            count += attn.alpha_q.numel() + attn.alpha_k.numel() + attn.alpha_v.numel()
            count += attn.bias_q.numel() + attn.bias_k.numel() + attn.bias_v.numel()
        return count
    
    @property
    def equivalent_standard_params(self) -> int:
        """Count of params if we used standard attention (3*d^2 per layer)."""
        d = self.embed_dim
        return len(self.blocks) * (3 * d * d + d * 3)
    
    @property
    def compression_ratio(self) -> float:
        """Compression ratio for attention layers."""
        return self.equivalent_standard_params / max(self.fourier_params, 1)
