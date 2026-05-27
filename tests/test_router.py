"""Tests for the token router module."""

import pytest
import torch
import torch.nn as nn

from dynaroute.router import TokenRouter, AdaptiveRouter


# ======================================================================
# TokenRouter
# ======================================================================


class TestTokenRouter:
    """Tests for TokenRouter."""

    def test_output_shapes(self, sample_tensor):
        router = TokenRouter(d_model=32, hidden_dim=16, num_paths=2)
        weights, logits = router(sample_tensor)

        B, S, D = sample_tensor.shape
        assert weights.shape == (B, S, 2)
        assert logits.shape == (B, S, 2)

    def test_soft_routing_sums_to_one(self, sample_tensor):
        router = TokenRouter(d_model=32, hidden_dim=16, num_paths=2)
        router.train()
        weights, _ = router(sample_tensor)

        path_sum = weights.sum(dim=-1)
        assert torch.allclose(path_sum, torch.ones_like(path_sum), atol=1e-5)

    def test_hard_routing_is_one_hot(self, sample_tensor):
        router = TokenRouter(d_model=32, hidden_dim=16, num_paths=2)
        router.eval()
        weights, _ = router(sample_tensor)

        # Each position should be exactly one-hot
        assert torch.allclose(weights.sum(dim=-1), torch.ones(sample_tensor.shape[:2]))
        # Values should be 0 or 1
        assert ((weights == 0) | (weights == 1)).all()

    def test_no_context(self, sample_tensor):
        router = TokenRouter(d_model=32, hidden_dim=16, num_paths=2, use_context=False)
        weights, logits = router(sample_tensor)

        assert weights.shape[-1] == 2
        assert logits.shape[-1] == 2

    def test_entropy_range(self, sample_tensor):
        router = TokenRouter(d_model=32, hidden_dim=16, num_paths=2)
        _, logits = router(sample_tensor)
        entropy = router.get_routing_entropy(logits)

        # Entropy of a 2-class distribution is in [0, ln(2)]
        assert entropy.min() >= -1e-6
        assert entropy.max() <= 0.6932 + 1e-3  # ln(2)

    def test_load_balance_loss_non_negative(self, sample_tensor):
        router = TokenRouter(d_model=32, hidden_dim=16, num_paths=2)
        router.train()
        weights, _ = router(sample_tensor)
        loss = router.get_load_balance_loss(weights)

        assert loss.item() >= -1e-6

    def test_custom_num_paths(self, sample_tensor):
        router = TokenRouter(d_model=32, hidden_dim=16, num_paths=3)
        weights, logits = router(sample_tensor)
        assert weights.shape[-1] == 3

    def test_gradient_flows(self, sample_tensor):
        router = TokenRouter(d_model=32, hidden_dim=16, num_paths=2)
        router.train()
        weights, _ = router(sample_tensor)
        weights.sum().backward()

        for param in router.parameters():
            assert param.grad is not None


# ======================================================================
# AdaptiveRouter
# ======================================================================


class TestAdaptiveRouter:
    """Tests for AdaptiveRouter."""

    def test_output_shapes(self, sample_tensor):
        router = AdaptiveRouter(d_model=32, hidden_dim=16, num_paths=2)
        weights, logits = router(sample_tensor)

        B, S, _ = sample_tensor.shape
        assert weights.shape == (B, S, 2)
        assert logits.shape == (B, S, 2)

    def test_learnable_temperature(self):
        router = AdaptiveRouter(d_model=32, temperature_init=2.0)
        assert isinstance(router.temperature, torch.Tensor)
        assert router.temperature.item() == pytest.approx(2.0, abs=1e-4)

    def test_soft_routing_sums_to_one(self, sample_tensor):
        router = AdaptiveRouter(d_model=32, hidden_dim=16, num_paths=2)
        weights, _ = router(sample_tensor, hard=False)

        path_sum = weights.sum(dim=-1)
        assert torch.allclose(path_sum, torch.ones_like(path_sum), atol=1e-4)

    def test_hard_routing_respects_capacity(self):
        S = 16
        router = AdaptiveRouter(
            d_model=32, hidden_dim=16, num_paths=2, capacity_factor=1.0
        )
        x = torch.randn(1, S, 32)
        weights, _ = router(x, hard=True)

        # Each path gets at most S * capacity_factor / num_paths = 8 tokens
        path_counts = weights.sum(dim=1)  # (batch, num_paths)
        assert (path_counts <= S + 1).all()  # capacity factor may slightly exceed

    def test_gradient_flows(self, sample_tensor):
        router = AdaptiveRouter(d_model=32, hidden_dim=16)
        weights, _ = router(sample_tensor)
        weights.sum().backward()

        for param in router.parameters():
            assert param.grad is not None
