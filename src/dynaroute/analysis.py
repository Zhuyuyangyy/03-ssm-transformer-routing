"""Routing pattern analysis for DynaRoute models.

This module provides tools for analyzing and visualizing the routing patterns
learned by the hybrid SSM-Transformer model.

Key capabilities:
- Per-layer routing statistics
- Token-type routing preferences
- Sequence position routing patterns
- Routing stability analysis
- Gradient-based token importance attribution
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
import json
from pathlib import Path
from torch.utils.data import DataLoader

__all__ = ["RoutingAnalyzer"]


class RoutingAnalyzer:
    """Analyzer for routing patterns in DynaRoute models.

    Provides methods for analyzing how tokens are routed between SSM and
    Attention pathways, including:
    - Per-layer routing statistics
    - Token-type routing preferences
    - Sequence position routing patterns
    - Routing stability analysis
    - Gradient-based token importance

    Args:
        model: Trained DynaRoute model.
        device: Analysis device.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.device = device
        self.model = model.to(device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    @torch.no_grad()
    def analyze_routing(
        self,
        input_ids: torch.Tensor,
        token_types: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Analyze routing patterns for given inputs.

        Args:
            input_ids: Input token IDs (batch, seq_len).
            token_types: Optional token type labels (batch, seq_len).
                E.g., 0=content, 1=function, 2=stopword.

        Returns:
            Dictionary with routing analysis results.

        Raises:
            ValueError: If the model does not return routing information.
        """
        input_ids = input_ids.to(self.device)

        # Forward pass to collect routing info
        _, routing_info = self.model(input_ids, return_routing=True)

        if routing_info is None:
            raise ValueError(
                "Model did not return routing information. "
                "Ensure the model was called with return_routing=True."
            )

        analysis: Dict[str, Any] = {
            "routing_weights": routing_info["routing_weights"].cpu().numpy(),
            "routing_logits": routing_info["routing_logits"].cpu().numpy(),
            "ssm_ratio": routing_info["ssm_ratio"],
            "attn_ratio": routing_info["attn_ratio"],
        }

        # Per-token-type analysis
        if token_types is not None:
            analysis["per_type"] = self._analyze_per_type(
                routing_info["routing_weights"], token_types.to(self.device)
            )

        # Position-based analysis
        analysis["per_position"] = self._analyze_per_position(
            routing_info["routing_weights"]
        )

        # Routing entropy
        analysis["entropy"] = self._compute_entropy(
            routing_info["routing_logits"]
        ).cpu().numpy()

        return analysis

    # ------------------------------------------------------------------
    # Sub-analyses
    # ------------------------------------------------------------------

    def _analyze_per_type(
        self,
        routing_weights: torch.Tensor,
        token_types: torch.Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """Analyze routing preferences per token type.

        Args:
            routing_weights: Routing weights (batch, seq_len, num_paths).
            token_types: Token type labels (batch, seq_len).

        Returns:
            Dictionary mapping token type to routing statistics.
        """
        results = {}
        weights = routing_weights.cpu().numpy()
        types = token_types.cpu().numpy()

        for t in np.unique(types):
            mask = types == t
            type_weights = weights[mask]

            results[f"type_{int(t)}"] = {
                "ssm_ratio": float(type_weights[..., 0].mean()),
                "attn_ratio": float(type_weights[..., 1].mean()),
                "count": int(mask.sum()),
                "std": float(type_weights[..., 0].std()),
            }

        return results

    def _analyze_per_position(
        self,
        routing_weights: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """Analyze routing patterns by sequence position.

        Args:
            routing_weights: Routing weights (batch, seq_len, num_paths).

        Returns:
            Dictionary with position-based routing statistics.
        """
        weights = routing_weights.cpu().numpy()

        return {
            "mean_ssm_by_position": weights[..., 0].mean(axis=0),
            "std_ssm_by_position": weights[..., 0].std(axis=0),
            "mean_attn_by_position": weights[..., 1].mean(axis=0),
            "std_attn_by_position": weights[..., 1].std(axis=0),
        }

    @staticmethod
    def _compute_entropy(logits: torch.Tensor) -> torch.Tensor:
        """Compute routing entropy from logits.

        Args:
            logits: Routing logits (batch, seq_len, num_paths).

        Returns:
            Entropy tensor (batch, seq_len).
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

    # ------------------------------------------------------------------
    # Pattern discovery
    # ------------------------------------------------------------------

    def find_routing_patterns(
        self,
        dataloader: DataLoader,
        num_batches: int = 100,
    ) -> Dict[str, Any]:
        """Discover common routing patterns across a dataset.

        Args:
            dataloader: Data loader for analysis. Batches can be dicts
                (with ``"input_ids"`` key) or tuples.
            num_batches: Number of batches to analyze.

        Returns:
            Dictionary with discovered patterns sorted by frequency.
        """
        pattern_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "positions": []}
        )

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            input_ids = self._extract_input_ids(batch)
            analysis = self.analyze_routing(input_ids)

            # Extract routing decisions (hard)
            routing_weights = torch.tensor(analysis["routing_weights"])
            hard_routing = routing_weights.argmax(dim=-1)  # (batch, seq_len)

            # Find contiguous routing blocks
            for b in range(hard_routing.size(0)):
                self._extract_patterns(
                    hard_routing[b].numpy(),
                    pattern_stats,
                )

        return self._aggregate_patterns(pattern_stats)

    @staticmethod
    def _extract_input_ids(
        batch: Union[Dict[str, torch.Tensor], tuple, list],
    ) -> torch.Tensor:
        """Extract input_ids from a batch regardless of format.

        Args:
            batch: A batch from a data loader.

        Returns:
            Input tensor.
        """
        if isinstance(batch, dict):
            return batch["input_ids"]
        if isinstance(batch, (tuple, list)):
            return batch[0]
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    @staticmethod
    def _extract_patterns(
        routing: np.ndarray,
        pattern_stats: Dict[str, Dict[str, Any]],
    ):
        """Extract contiguous routing patterns from a sequence.

        Args:
            routing: Hard routing decisions (seq_len,).
            pattern_stats: Dictionary to accumulate pattern statistics.
        """
        if len(routing) == 0:
            return

        current_path = routing[0]
        start = 0

        for i in range(1, len(routing)):
            if routing[i] != current_path:
                pattern_key = f"path_{int(current_path)}_len_{i - start}"
                pattern_stats[pattern_key]["count"] += 1
                pattern_stats[pattern_key]["positions"].append(start)
                current_path = routing[i]
                start = i

        # Record last pattern
        pattern_key = f"path_{int(current_path)}_len_{len(routing) - start}"
        pattern_stats[pattern_key]["count"] += 1
        pattern_stats[pattern_key]["positions"].append(start)

    @staticmethod
    def _aggregate_patterns(
        pattern_stats: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate routing pattern statistics.

        Args:
            pattern_stats: Raw pattern statistics.

        Returns:
            Aggregated pattern analysis sorted by frequency.
        """
        aggregated = {}

        for pattern, stats in pattern_stats.items():
            positions = np.array(stats["positions"])
            aggregated[pattern] = {
                "count": stats["count"],
                "avg_position": float(positions.mean()) if len(positions) > 0 else 0.0,
                "std_position": float(positions.std()) if len(positions) > 0 else 0.0,
            }

        # Sort by frequency (descending)
        aggregated = dict(
            sorted(aggregated.items(), key=lambda x: x[1]["count"], reverse=True)
        )

        return aggregated

    # ------------------------------------------------------------------
    # Stability analysis
    # ------------------------------------------------------------------

    def compute_routing_stability(
        self,
        input_ids: torch.Tensor,
        num_trials: int = 10,
    ) -> Dict[str, float]:
        """Compute routing stability across multiple forward passes.

        Measures how consistent routing decisions are when the model
        is evaluated multiple times on the same input.

        Args:
            input_ids: Input token IDs (batch, seq_len).
            num_trials: Number of forward passes to compare.

        Returns:
            Dictionary with stability metrics:
            - ``agreement_rate``: Fraction of tokens with consistent routing.
            - ``position_variance``: Average per-position routing variance.
            - ``stability_score``: Combined metric (higher is more stable).
        """
        input_ids = input_ids.to(self.device)

        # Enable stochastic routing
        self.model.train()

        all_routes = []
        for _ in range(num_trials):
            with torch.no_grad():
                _, routing_info = self.model(input_ids, return_routing=True)
                hard_routing = routing_info["routing_weights"].argmax(dim=-1)
                all_routes.append(hard_routing.cpu().numpy())

        self.model.eval()

        all_routes = np.array(all_routes)  # (num_trials, batch, seq_len)

        # Compute mode (most common route) per position
        flat_routes = all_routes.reshape(num_trials, -1)
        mode_route = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=flat_routes
        )
        mode_route = mode_route.reshape(all_routes.shape[1:])

        # Agreement rate
        agreement = float((all_routes == mode_route).mean())

        # Per-position variance
        position_var = float(all_routes.var(axis=0).mean())

        return {
            "agreement_rate": agreement,
            "position_variance": position_var,
            "stability_score": agreement * (1 - position_var),
        }

    # ------------------------------------------------------------------
    # Token importance
    # ------------------------------------------------------------------

    def compute_token_importance(
        self,
        input_ids: torch.Tensor,
        token_idx: int,
        embedding_module: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """Compute importance of a specific token to routing decisions.

        Uses gradient-based attribution to measure how much each input
        token influences the routing of a target token. The gradient flows
        through the embedding layer, so ``input_ids`` are converted to
        embeddings first.

        Args:
            input_ids: Input token IDs (1, seq_len).
            token_idx: Index of target token whose routing logits we
                differentiate through.
            embedding_module: The embedding layer to use for converting
                token IDs to embeddings. If None, the method looks for
                ``model.embedding`` or the first ``nn.Embedding`` child.

        Returns:
            Dictionary with token importance scores and metadata.

        Raises:
            ValueError: If no embedding module can be found.
        """
        self.model.eval()
        input_ids = input_ids.to(self.device)

        # Find embedding module
        if embedding_module is None:
            embedding_module = self._find_embedding_module()
        if embedding_module is None:
            raise ValueError(
                "Could not find an embedding module. Pass one explicitly."
            )

        # Get embeddings with gradient
        with torch.enable_grad():
            embeddings = embedding_module(input_ids)  # (1, seq_len, d_model)
            embeddings.retain_grad()

            # Forward through the rest of the model
            _, routing_info = self.model(embeddings, return_routing=True)

            if routing_info is None:
                raise ValueError("Model did not return routing information.")

            # Backprop from target token's routing logits
            target_logits = routing_info["routing_logits"][0, token_idx]
            target_logits.sum().backward()

            # Importance = gradient magnitude
            importance = embeddings.grad.abs()[0].detach().cpu().numpy()

        return {
            "token_importance": importance,
            "target_token_idx": token_idx,
            "routing_logits": target_logits.detach().cpu().numpy(),
        }

    def _find_embedding_module(self) -> Optional[nn.Module]:
        """Try to find an embedding module in the model.

        Returns:
            The embedding module, or None if not found.
        """
        # Common attribute names
        for attr in ("embedding", "embed", "token_embedding", "wte"):
            mod = getattr(self.model, attr, None)
            if isinstance(mod, nn.Embedding):
                return mod

        # Search children
        for module in self.model.modules():
            if isinstance(module, nn.Embedding):
                return module

        return None

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def export_analysis(
        analysis: Dict[str, Any],
        output_path: str,
    ):
        """Export analysis results to JSON.

        Numpy arrays are recursively converted to nested lists.

        Args:
            analysis: Analysis results dictionary.
            output_path: Path to output JSON file.
        """
        def _convert(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            return obj

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(_convert(analysis), f, indent=2)
