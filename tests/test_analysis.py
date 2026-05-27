"""Tests for the routing analysis module."""

import json
import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from dynaroute.analysis import RoutingAnalyzer
from dynaroute.hybrid_layer import DynaRouteBlock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(vocab_size=32, d_model=16, n_heads=2, ssm_state_dim=4):
    """Create a minimal model that returns routing info."""

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.block = DynaRouteBlock(
                d_model=d_model, n_heads=n_heads,
                d_ff=32, ssm_state_dim=ssm_state_dim, dropout=0.0,
            )
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, input_ids, return_routing=False):
            x = self.embedding(input_ids)
            x, routing_info = self.block(x, return_routing=return_routing)
            logits = self.head(x)
            return logits, routing_info

    return TestModel()


def _make_float_model(d_model=16, n_heads=2, ssm_state_dim=4):
    """Create a model that accepts float inputs (for gradient tests)."""

    class FloatModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(32, d_model)
            self.block = DynaRouteBlock(
                d_model=d_model, n_heads=n_heads,
                d_ff=32, ssm_state_dim=ssm_state_dim, dropout=0.0,
            )
            self.head = nn.Linear(d_model, 32, bias=False)

        def forward(self, x, return_routing=False):
            # If x is integer, embed it; otherwise treat as float embeddings
            if x.dtype in (torch.long, torch.int64):
                x = self.embedding(x)
            x, routing_info = self.block(x, return_routing=return_routing)
            logits = self.head(x)
            return logits, routing_info

    return FloatModel()


# ======================================================================
# RoutingAnalyzer
# ======================================================================


class TestRoutingAnalyzer:
    """Tests for RoutingAnalyzer."""

    def test_analyze_routing_output_keys(self):
        model = _make_model()
        analyzer = RoutingAnalyzer(model, device="cpu")
        input_ids = torch.randint(0, 32, (2, 8))

        result = analyzer.analyze_routing(input_ids)

        assert "routing_weights" in result
        assert "routing_logits" in result
        assert "ssm_ratio" in result
        assert "attn_ratio" in result
        assert "per_position" in result
        assert "entropy" in result

    def test_ratios_sum_to_one(self):
        model = _make_model()
        analyzer = RoutingAnalyzer(model, device="cpu")
        input_ids = torch.randint(0, 32, (2, 8))

        result = analyzer.analyze_routing(input_ids)
        assert result["ssm_ratio"] + result["attn_ratio"] == pytest.approx(1.0, abs=1e-4)

    def test_entropy_shape(self):
        model = _make_model()
        analyzer = RoutingAnalyzer(model, device="cpu")
        input_ids = torch.randint(0, 32, (2, 8))

        result = analyzer.analyze_routing(input_ids)
        assert result["entropy"].shape == (2, 8)

    def test_per_type_analysis(self):
        model = _make_model()
        analyzer = RoutingAnalyzer(model, device="cpu")
        input_ids = torch.randint(0, 32, (2, 8))
        token_types = torch.randint(0, 3, (2, 8))

        result = analyzer.analyze_routing(input_ids, token_types=token_types)
        assert "per_type" in result
        assert len(result["per_type"]) <= 3

    def test_per_position_keys(self):
        model = _make_model()
        analyzer = RoutingAnalyzer(model, device="cpu")
        input_ids = torch.randint(0, 32, (2, 8))

        result = analyzer.analyze_routing(input_ids)
        pos = result["per_position"]
        assert "mean_ssm_by_position" in pos
        assert "std_ssm_by_position" in pos
        assert "mean_attn_by_position" in pos

    def test_stability_metrics(self):
        model = _make_model()
        analyzer = RoutingAnalyzer(model, device="cpu")
        input_ids = torch.randint(0, 32, (1, 8))

        result = analyzer.compute_routing_stability(input_ids, num_trials=3)

        assert "agreement_rate" in result
        assert "position_variance" in result
        assert "stability_score" in result
        assert 0 <= result["agreement_rate"] <= 1
        assert result["stability_score"] >= 0

    def test_find_routing_patterns(self):
        model = _make_model()
        analyzer = RoutingAnalyzer(model, device="cpu")

        input_ids = torch.randint(0, 32, (4, 8))
        labels = torch.randint(0, 32, (4, 8))
        dataset = TensorDataset(input_ids, labels)
        loader = DataLoader(dataset, batch_size=2)

        patterns = analyzer.find_routing_patterns(loader, num_batches=2)
        assert isinstance(patterns, dict)
        # Each pattern key should start with "path_"
        for key in patterns:
            assert key.startswith("path_")

    def test_find_routing_patterns_dict_batch(self):
        """Test with dict-style batches."""
        model = _make_model()
        analyzer = RoutingAnalyzer(model, device="cpu")

        input_ids = torch.randint(0, 32, (4, 8))
        labels = torch.randint(0, 32, (4, 8))

        class DictDataset(TensorDataset):
            def __getitem__(self, idx):
                tensors = super().__getitem__(idx)
                return {"input_ids": tensors[0], "labels": tensors[1]}

        dataset = DictDataset(input_ids, labels)
        loader = DataLoader(dataset, batch_size=2)

        patterns = analyzer.find_routing_patterns(loader, num_batches=2)
        assert isinstance(patterns, dict)

    def test_export_analysis(self, tmp_path):
        model = _make_model()
        analyzer = RoutingAnalyzer(model, device="cpu")
        input_ids = torch.randint(0, 32, (1, 8))
        result = analyzer.analyze_routing(input_ids)

        out_path = str(tmp_path / "analysis.json")
        analyzer.export_analysis(result, out_path)

        with open(out_path) as f:
            loaded = json.load(f)

        assert "ssm_ratio" in loaded
        assert "attn_ratio" in loaded
        assert isinstance(loaded["routing_weights"], list)

    def test_compute_token_importance(self):
        model = _make_float_model()
        analyzer = RoutingAnalyzer(model, device="cpu")

        input_ids = torch.randint(0, 32, (1, 8))
        result = analyzer.compute_token_importance(
            input_ids, token_idx=3, embedding_module=model.embedding,
        )

        assert "token_importance" in result
        assert "target_token_idx" in result
        assert "routing_logits" in result
        assert result["target_token_idx"] == 3
        assert result["token_importance"].shape == (1, 8, 16)

    def test_compute_token_importance_auto_find(self):
        """Test that the method auto-discovers the embedding module."""
        model = _make_float_model()
        # Delete the explicit .embedding attribute to test fallback
        # Actually the model has self.embedding, so auto-find should work
        analyzer = RoutingAnalyzer(model, device="cpu")
        input_ids = torch.randint(0, 32, (1, 8))

        result = analyzer.compute_token_importance(input_ids, token_idx=0)
        assert "token_importance" in result
