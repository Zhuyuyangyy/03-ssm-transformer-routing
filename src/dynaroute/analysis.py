"""Routing pattern analysis for DynaRoute models.

This module provides tools for analyzing and visualizing the routing patterns
learned by the hybrid SSM-Transformer model.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import json
from pathlib import Path


class RoutingAnalyzer:
    """Analyzer for routing patterns in DynaRoute models.

    Provides methods for analyzing how tokens are routed between SSM and
    Attention pathways, including:
    - Per-layer routing statistics
    - Token-type routing preferences
    - Sequence position routing patterns
    - Routing stability analysis

    Args:
        model: Trained DynaRoute model.
        device: Analysis device.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

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
        """
        input_ids = input_ids.to(self.device)

        # Forward pass to collect routing info
        outputs, routing_info = self.model(input_ids, return_routing=True)

        if routing_info is None:
            raise ValueError("Model did not return routing information")

        analysis = {
            "routing_weights": routing_info["routing_weights"].cpu().numpy(),
            "routing_logits": routing_info["routing_logits"].cpu().numpy(),
            "ssm_ratio": routing_info["ssm_ratio"],
            "attn_ratio": routing_info["attn_ratio"],
        }

        # Per-token-type analysis
        if token_types is not None:
            analysis["per_type"] = self._analyze_per_type(
                routing_info["routing_weights"], token_types
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

        unique_types = np.unique(types)
        for t in unique_types:
            mask = types == t
            type_weights = weights[mask]

            results[f"type_{t}"] = {
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
        }

    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute routing entropy.

        Args:
            logits: Routing logits (batch, seq_len, num_paths).

        Returns:
            Entropy tensor (batch, seq_len).
        """
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

    def find_routing_patterns(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 100,
    ) -> Dict[str, Any]:
        """Discover common routing patterns across a dataset.

        Args:
            dataloader: Data loader for analysis.
            num_batches: Number of batches to analyze.

        Returns:
            Dictionary with discovered patterns.
        """
        pattern_stats = defaultdict(lambda: {"count": 0, "positions": []})

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            analysis = self.analyze_routing(input_ids)

            # Extract routing decisions (hard)
            routing_weights = torch.tensor(analysis["routing_weights"])
            hard_routing = routing_weights.argmax(dim=-1)  # (batch, seq_len)

            # Find contiguous routing blocks
            for b in range(hard_routing.size(0)):
                self._extract_patterns(
                    hard_routing[b].cpu().numpy(),
                    pattern_stats,
                )

        # Aggregate patterns
        return self._aggregate_patterns(pattern_stats)

    def _extract_patterns(
        self,
        routing: np.ndarray,
        pattern_stats: Dict,
    ):
        """Extract contiguous routing patterns from a sequence.

        Args:
            routing: Hard routing decisions (seq_len,).
            pattern_stats: Dictionary to accumulate pattern statistics.
        """
        # Find runs of same routing decision
        current_path = routing[0]
        start = 0

        for i in range(1, len(routing)):
            if routing[i] != current_path:
                # Record pattern
                pattern_key = f"path_{current_path}_len_{i - start}"
                pattern_stats[pattern_key]["count"] += 1
                pattern_stats[pattern_key]["positions"].append(start)

                current_path = routing[i]
                start = i

        # Record last pattern
        pattern_key = f"path_{current_path}_len_{len(routing) - start}"
        pattern_stats[pattern_key]["count"] += 1
        pattern_stats[pattern_key]["positions"].append(start)

    def _aggregate_patterns(
        self, pattern_stats: Dict
    ) -> Dict[str, Any]:
        """Aggregate routing pattern statistics.

        Args:
            pattern_stats: Raw pattern statistics.

        Returns:
            Aggregated pattern analysis.
        """
        aggregated = {}

        for pattern, stats in pattern_stats.items():
            positions = np.array(stats["positions"])
            aggregated[pattern] = {
                "count": stats["count"],
                "avg_position": float(positions.mean()) if len(positions) > 0 else 0,
                "std_position": float(positions.std()) if len(positions) > 0 else 0,
            }

        # Sort by frequency
        aggregated = dict(
            sorted(aggregated.items(), key=lambda x: x[1]["count"], reverse=True)
        )

        return aggregated

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
            Dictionary with stability metrics.
        """
        self.model.train()  # Enable stochastic routing

        all_routes = []
        for _ in range(num_trials):
            with torch.no_grad():
                _, routing_info = self.model(input_ids.to(self.device), return_routing=True)
                hard_routing = routing_info["routing_weights"].argmax(dim=-1)
                all_routes.append(hard_routing.cpu().numpy())

        self.model.eval()

        all_routes = np.array(all_routes)  # (num_trials, batch, seq_len)

        # Compute agreement rate
        mode_route = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=all_routes.reshape(num_trials, -1)
        )
        mode_route = mode_route.reshape(all_routes.shape[1:])
        agreement = (all_routes == mode_route).mean()

        # Compute per-position variance
        position_var = all_routes.var(axis=0).mean()

        return {
            "agreement_rate": float(agreement),
            "position_variance": float(position_var),
            "stability_score": float(agreement * (1 - position_var)),
        }

    def export_analysis(
        self,
        analysis: Dict[str, Any],
        output_path: str,
    ):
        """Export analysis results to JSON.

        Args:
            analysis: Analysis results dictionary.
            output_path: Path to output JSON file.
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable = {}
        for key, value in analysis.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, dict):
                serializable[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                serializable[key] = value

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)

    def compute_token_importance(
        self,
        input_ids: torch.Tensor,
        token_idx: int,
    ) -> Dict[str, float]:
        """Compute importance of a specific token to routing decisions.

        Uses gradient-based attribution to measure how much each input
        token influences the routing of a target token.

        Args:
            input_ids: Input token IDs (1, seq_len).
            token_idx: Index of target token.

        Returns:
            Dictionary with token importance scores.
        """
        self.model.eval()
        input_tensor = input_ids.to(self.device).float().requires_grad_(True)

        # Forward pass
        _, routing_info = self.model(input_tensor, return_routing=True)

        # Get routing logits for target token
        target_logits = routing_info["routing_logits"][0, token_idx]

        # Compute gradients
        target_logits.sum().backward()

        # Importance = gradient magnitude
        importance = input_tensor.grad.abs()[0].cpu().numpy()

        return {
            "token_importance": importance,
            "target_token_idx": token_idx,
            "routing_logits": target_logits.detach().cpu().numpy(),
        }
