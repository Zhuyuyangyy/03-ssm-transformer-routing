"""Token-level router for dynamic pathway selection.

This module implements the routing mechanism that decides whether each token
should be processed through the SSM pathway or the Attention pathway.
Two router variants are provided:

- ``TokenRouter``: Standard router with Gumbel-Softmax training and optional
  context aggregation.
- ``AdaptiveRouter``: Router with learnable temperature and capacity constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

__all__ = ["TokenRouter", "AdaptiveRouter"]


class TokenRouter(nn.Module):
    """Token-level router for dynamic pathway selection.

    Uses a lightweight MLP to compute routing probabilities for each token,
    determining whether it should be processed by the SSM or Attention pathway.

    The router uses Gumbel-Softmax during training for differentiable discrete
    selection and argmax during inference for hard routing.

    Args:
        d_model: Input model dimension.
        hidden_dim: Hidden dimension of the router MLP.
        temperature: Gumbel-Softmax temperature. Higher values produce softer routing.
        num_paths: Number of routing paths (default: 2 for SSM and Attention).
        use_context: Whether to use contextual information for routing.
    """

    def __init__(
        self,
        d_model: int = 512,
        hidden_dim: int = 256,
        temperature: float = 1.0,
        num_paths: int = 2,
        use_context: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.num_paths = num_paths
        self.use_context = use_context

        # Context aggregation (optional)
        context_dim = 0
        if use_context:
            context_dim = d_model // 4
            self.context_proj = nn.Linear(d_model, context_dim, bias=False)

        # Router MLP
        input_dim = d_model + context_dim
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_paths),
        )

        # Learnable bias for balanced routing
        self.routing_bias = nn.Parameter(torch.zeros(num_paths))

        self._init_weights()

    def _init_weights(self):
        """Initialize router weights with small values for stable early training."""
        for module in self.router:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _aggregate_context(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate contextual information from the sequence.

        Uses mean-pooled sequence representation as context, broadcast to
        every position so that each token sees the global picture.

        Args:
            x: Input tensor (batch, seq_len, d_model).

        Returns:
            Context tensor (batch, seq_len, d_model // 4).
        """
        context = x.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        context = context.expand_as(x)  # (batch, seq_len, d_model)
        return self.context_proj(context)

    def forward(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute routing decisions for each token.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            hard: Whether to use hard routing. Defaults to ``not self.training``.

        Returns:
            Tuple of:
                - routing_weights: Soft/hard routing weights (batch, seq_len, num_paths).
                - routing_logits: Raw routing logits (batch, seq_len, num_paths).
        """
        if hard is None:
            hard = not self.training

        # Prepare input
        if self.use_context:
            context = self._aggregate_context(x)
            router_input = torch.cat([x, context], dim=-1)
        else:
            router_input = x

        # Compute routing logits
        logits = self.router(router_input) + self.routing_bias

        if hard:
            # Hard routing during inference
            indices = logits.argmax(dim=-1)
            weights = F.one_hot(indices, self.num_paths).float()
        else:
            # Gumbel-Softmax during training
            weights = F.gumbel_softmax(
                logits, tau=max(self.temperature, 1e-8), hard=False
            )

        return weights, logits

    def get_routing_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy of routing distribution.

        Higher entropy indicates more balanced routing; lower entropy
        indicates the router is more decisive.

        Args:
            logits: Routing logits (batch, seq_len, num_paths).

        Returns:
            Entropy tensor (batch, seq_len).
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

    def get_load_balance_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss to encourage balanced routing.

        Penalizes routing imbalances to prevent one pathway from being
        underutilized. Uses a numerically stable formulation.

        Args:
            routing_weights: Routing weights (batch, seq_len, num_paths).

        Returns:
            Scalar load balancing loss.
        """
        # Average routing probability per path
        avg_probs = routing_weights.mean(dim=[0, 1])  # (num_paths,)

        # Target: uniform distribution
        target = torch.ones_like(avg_probs) / self.num_paths

        # KL divergence from uniform (numerically stable via log_softmax)
        loss = F.kl_div(
            F.log_softmax(avg_probs.log(), dim=-1),
            target,
            reduction="batchmean",
        )
        return loss


class AdaptiveRouter(nn.Module):
    """Adaptive router with learned temperature and capacity constraints.

    Extends TokenRouter with:
    - Learnable temperature per layer
    - Capacity factor to limit tokens per path
    - Auxiliary loss for balanced utilization

    Args:
        d_model: Input model dimension.
        hidden_dim: Hidden dimension.
        num_paths: Number of routing paths.
        capacity_factor: Maximum tokens per path as ratio of total tokens.
        temperature_init: Initial temperature value.
    """

    def __init__(
        self,
        d_model: int = 512,
        hidden_dim: int = 256,
        num_paths: int = 2,
        capacity_factor: float = 1.25,
        temperature_init: float = 1.0,
    ):
        super().__init__()
        self.num_paths = num_paths
        self.capacity_factor = capacity_factor

        # Learnable temperature
        self.log_temperature = nn.Parameter(
            torch.tensor(float(temperature_init)).log()
        )

        # Router network
        self.router = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_paths),
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Current routing temperature."""
        return self.log_temperature.exp()

    def forward(
        self,
        x: torch.Tensor,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute routing with capacity constraints.

        Args:
            x: Input tensor (batch, seq_len, d_model).
            hard: Whether to use hard routing.

        Returns:
            Tuple of (routing_weights, routing_logits).
        """
        logits = self.router(x)

        if hard:
            # Hard routing with capacity constraint
            return self._capacity_routing(logits)
        else:
            # Soft routing
            tau = self.temperature.clamp(min=1e-8)
            weights = F.gumbel_softmax(logits, tau=tau, hard=False)
            return weights, logits

    def _capacity_routing(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply capacity-constrained routing.

        Limits the number of tokens routed to each path based on
        the capacity factor.

        Args:
            logits: Routing logits (batch, seq_len, num_paths).

        Returns:
            Tuple of (routing_weights, routing_logits).
        """
        batch, seq_len, num_paths = logits.shape
        capacity = max(1, int(seq_len * self.capacity_factor / num_paths))

        probs = F.softmax(logits, dim=-1)

        # Build hard routing via top-k per path
        weights = torch.zeros_like(probs)
        path_indicator = torch.eye(
            num_paths, device=logits.device, dtype=logits.dtype
        )  # (num_paths, num_paths)

        for path_idx in range(num_paths):
            path_probs = probs[..., path_idx]  # (batch, seq_len)
            _, top_indices = path_probs.topk(capacity, dim=-1)
            # Scatter the one-hot vector for this path at the selected positions
            # top_indices: (batch, capacity) -> expand for scatter
            idx_expanded = top_indices.unsqueeze(-1).expand(-1, -1, num_paths)
            one_hot = path_indicator[path_idx].expand(batch, capacity, -1)
            weights.scatter_(1, idx_expanded, one_hot)

        return weights, logits
