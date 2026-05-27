"""Dual-path hybrid layer combining SSM and Attention pathways.

This module implements the core HybridLayer that routes tokens through either
an SSM (State Space Model) pathway or a Transformer attention pathway based
on learned routing decisions.

Key components:
- ``SSMPathway``: Simplified Mamba-style SSM with selective scan.
- ``AttentionPathway``: Multi-head self-attention with automatic flash-attention
  dispatch via ``torch.nn.functional.scaled_dot_product_attention``.
- ``HybridLayer``: Token-level routing between the two pathways.
- ``DynaRouteBlock``: Complete block with FFN and residual connections.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .router import TokenRouter

__all__ = ["SSMPathway", "AttentionPathway", "HybridLayer", "DynaRouteBlock"]


class SSMPathway(nn.Module):
    """State Space Model pathway for efficient long-range dependencies.

    Implements a simplified Mamba-style SSM with selective scan mechanism.
    The selective scan is sequential (O(n) in sequence length) and serves
    as a reference implementation.

    Args:
        d_model: Model dimension.
        state_dim: SSM state dimension.
        dt_rank: Rank for dt projection. Defaults to ``d_model // 16``.
        expand: Expansion factor for inner dimension.
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 64,
        dt_rank: Optional[int] = None,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.dt_rank = dt_rank or max(1, d_model // 16)
        self.d_inner = d_model * expand

        # Input projection (x and z gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # SSM parameters
        # A is initialized as a log-space parameter for numerical stability
        A = torch.arange(1, self.state_dim + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1).clone()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Selective parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + state_dim * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt bias to a reasonable range
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.bias, -dt_init_std, dt_init_std)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SSM pathway.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        # Project input
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        # Compute selective parameters
        x_dbl = self.x_proj(x_ssm)
        dt, B, C = x_dbl.split([self.dt_rank, self.state_dim, self.state_dim], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        # SSM computation (selective scan)
        A = -torch.exp(self.A_log.float())
        y = self._selective_scan(x_ssm.float(), dt.float(), A, B.float(), C.float())
        y = y.to(x.dtype)

        # Skip connection and gate
        y = y + x_ssm * self.D
        y = y * torch.sigmoid(z)
        return self.out_proj(y)

    def _selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Perform selective scan operation.

        This is a sequential reference implementation (one step at a time).
        For production, consider replacing with a parallel scan kernel.

        Args:
            x: Input tensor (batch, seq_len, d_inner).
            dt: Time step tensor (batch, seq_len, d_inner).
            A: State matrix (d_inner, state_dim).
            B: Input matrix (batch, seq_len, state_dim).
            C: Output matrix (batch, seq_len, state_dim).

        Returns:
            Output tensor (batch, seq_len, d_inner).
        """
        batch, seq_len, d_inner = x.shape
        state_dim = A.shape[1]

        # Initialize state
        h = torch.zeros(batch, d_inner, state_dim, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            # Discretized state update:
            #   h_t = exp(A * dt_t) * h_{t-1} + dt_t * B_t * x_t
            dt_t = dt[:, t, :].unsqueeze(-1)  # (batch, d_inner, 1)
            B_t = B[:, t, :].unsqueeze(1)  # (batch, 1, state_dim)
            x_t = x[:, t, :].unsqueeze(-1)  # (batch, d_inner, 1)

            h = torch.exp(A * dt_t) * h + dt_t * B_t * x_t

            # Output: y_t = sum_j C_{t,j} * h_{t,j}
            C_t = C[:, t, :].unsqueeze(1)  # (batch, 1, state_dim)
            y_t = (h * C_t).sum(dim=-1)  # (batch, d_inner)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class AttentionPathway(nn.Module):
    """Multi-head attention pathway for precise local context modeling.

    Uses ``torch.nn.functional.scaled_dot_product_attention`` which
    automatically dispatches to FlashAttention / Memory-efficient attention
    when the hardware and PyTorch version support it.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
        bias: Whether to use bias in linear layers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # Fused QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through attention pathway.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional attention mask. Boolean tensor where ``True`` indicates
                positions that **should** attend to each other. Shape should be
                broadcastable to ``(batch, n_heads, seq_len, seq_len)``.

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        # Fused QKV projection
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (batch, n_heads, seq_len, head_dim)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (auto-dispatches to flash attention)
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=dropout_p,
        )

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)


class HybridLayer(nn.Module):
    """Hybrid layer with dual SSM and Attention pathways.

    Routes each token through either the SSM pathway (for efficient long-range
    processing) or the Attention pathway (for precise local reasoning) based
    on learned routing decisions.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        ssm_state_dim: SSM state dimension.
        temperature: Routing temperature (higher = softer routing).
        router_hidden_dim: Hidden dimension for the router network.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        ssm_state_dim: int = 64,
        temperature: float = 1.0,
        router_hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature

        # Dual pathways
        self.ssm_path = SSMPathway(d_model, state_dim=ssm_state_dim)
        self.attn_path = AttentionPathway(d_model, n_heads=n_heads, dropout=dropout)

        # Router
        self.router = TokenRouter(
            d_model=d_model,
            hidden_dim=router_hidden_dim,
            temperature=temperature,
        )

        # Layer norms (pre-norm style for each pathway)
        self.norm_ssm = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_routing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Forward pass through hybrid layer.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional attention mask.
            return_routing: Whether to return routing information.

        Returns:
            Tuple of (output_tensor, routing_info_dict or None).
        """
        # Get routing decisions
        routing_weights, routing_logits = self.router(x)  # (batch, seq_len, 2)

        # Process through both pathways (pre-norm)
        ssm_out = self.norm_ssm(self.ssm_path(x))
        attn_out = self.norm_attn(self.attn_path(x, mask=mask))

        # Combine based on routing weights
        ssm_weight = routing_weights[..., 0:1]  # (batch, seq_len, 1)
        attn_weight = routing_weights[..., 1:2]  # (batch, seq_len, 1)

        output = ssm_weight * ssm_out + attn_weight * attn_out
        output = self.norm_out(output)

        # Collect routing info
        routing_info = None
        if return_routing:
            with torch.no_grad():
                ssm_ratio = routing_weights[..., 0].mean().item()
            routing_info = {
                "routing_weights": routing_weights,
                "routing_logits": routing_logits,
                "ssm_ratio": ssm_ratio,
                "attn_ratio": 1.0 - ssm_ratio,
            }

        return output, routing_info


class DynaRouteBlock(nn.Module):
    """Complete DynaRoute block with feedforward network.

    Combines HybridLayer with a feedforward network and residual connections
    following the pre-norm Transformer architecture.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feedforward dimension.
        ssm_state_dim: SSM state dimension.
        temperature: Routing temperature.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        ssm_state_dim: int = 64,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hybrid = HybridLayer(
            d_model=d_model,
            n_heads=n_heads,
            ssm_state_dim=ssm_state_dim,
            temperature=temperature,
            dropout=dropout,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_routing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Forward pass through DynaRoute block.

        Args:
            x: Input tensor (batch, seq_len, d_model).
            mask: Optional attention mask.
            return_routing: Whether to return routing information.

        Returns:
            Tuple of (output_tensor, routing_info_dict or None).
        """
        # Hybrid layer with residual
        residual = x
        h, routing_info = self.hybrid(x, mask=mask, return_routing=return_routing)
        h = h + residual

        # FFN with residual (pre-norm)
        h = h + self.ffn(self.norm(h))

        return h, routing_info
