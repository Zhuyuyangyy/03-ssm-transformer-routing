"""Dual-path hybrid layer combining SSM and Attention pathways.

This module implements the core HybridLayer that routes tokens through either
an SSM (State Space Model) pathway or a Transformer attention pathway based
on learned routing decisions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .router import TokenRouter


class SSMPathway(nn.Module):
    """State Space Model pathway for efficient long-range dependencies.

    Implements a simplified Mamba-style SSM with selective scan mechanism.
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 64,
        dt_rank: Optional[int] = None,
        expand: int = 2,
    ):
        """Initialize SSM pathway.

        Args:
            d_model: Model dimension.
            state_dim: SSM state dimension.
            dt_rank: Rank for dt projection. Defaults to d_model // 16.
            expand: Expansion factor for inner dimension.
        """
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.dt_rank = dt_rank or d_model // 16
        self.d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.randn(self.d_inner, state_dim).abs()))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Selective parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + state_dim * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

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

        # SSM computation (simplified selective scan)
        A = -torch.exp(self.A_log)
        y = self._selective_scan(x_ssm, dt, A, B, C)

        # Skip connection and output
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
            # Update state: h = exp(A * dt) * h + dt * B * x
            h = torch.exp(A * dt[:, t, :].unsqueeze(-1)) * h + \
                dt[:, t, :].unsqueeze(-1) * B[:, t, :].unsqueeze(1) * x[:, t, :].unsqueeze(-1)
            # Output: y = C * h
            y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class AttentionPathway(nn.Module):
    """Multi-head attention pathway for precise local context modeling.

    Implements standard multi-head self-attention with optional flash attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        """Initialize attention pathway.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # QKV projection
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through attention pathway.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional attention mask.

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        # Project QKV
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # Apply attention to values
        out = torch.matmul(attn, v)
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

        # Layer norms
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

        # Process through both pathways
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
            routing_info = {
                "routing_weights": routing_weights,
                "routing_logits": routing_logits,
                "ssm_ratio": routing_weights[..., 0].mean().item(),
                "attn_ratio": routing_weights[..., 1].mean().item(),
            }

        return output, routing_info


class DynaRouteBlock(nn.Module):
    """Complete DynaRoute block with feedforward network.

    Combines HybridLayer with a feedforward network and residual connections.

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

        # FFN with residual
        h = h + self.ffn(self.norm(h))

        return h, routing_info
