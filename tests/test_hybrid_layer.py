"""Tests for the hybrid layer module."""

import pytest
import torch
import torch.nn as nn

from dynaroute.hybrid_layer import (
    SSMPathway,
    AttentionPathway,
    HybridLayer,
    DynaRouteBlock,
)


# ======================================================================
# SSMPathway
# ======================================================================


class TestSSMPathway:
    """Tests for SSMPathway."""

    def test_output_shape(self, sample_tensor):
        ssm = SSMPathway(d_model=32, state_dim=8)
        out = ssm(sample_tensor)
        assert out.shape == sample_tensor.shape

    def test_different_state_dims(self, sample_tensor):
        for state_dim in (4, 8, 16):
            ssm = SSMPathway(d_model=32, state_dim=state_dim)
            out = ssm(sample_tensor)
            assert out.shape == sample_tensor.shape

    def test_different_expand_factors(self, sample_tensor):
        for expand in (1, 2, 4):
            ssm = SSMPathway(d_model=32, state_dim=8, expand=expand)
            out = ssm(sample_tensor)
            assert out.shape == sample_tensor.shape

    def test_gradient_flows(self, sample_tensor):
        ssm = SSMPathway(d_model=32, state_dim=8)
        out = ssm(sample_tensor)
        out.sum().backward()
        for p in ssm.parameters():
            assert p.grad is not None

    def test_deterministic_eval(self, sample_tensor):
        ssm = SSMPathway(d_model=32, state_dim=8)
        ssm.eval()
        out1 = ssm(sample_tensor)
        out2 = ssm(sample_tensor)
        assert torch.allclose(out1, out2)

    def test_sequence_length_one(self):
        ssm = SSMPathway(d_model=32, state_dim=8)
        x = torch.randn(2, 1, 32)
        out = ssm(x)
        assert out.shape == (2, 1, 32)


# ======================================================================
# AttentionPathway
# ======================================================================


class TestAttentionPathway:
    """Tests for AttentionPathway."""

    def test_output_shape(self, sample_tensor):
        attn = AttentionPathway(d_model=32, n_heads=4)
        out = attn(sample_tensor)
        assert out.shape == sample_tensor.shape

    def test_with_mask(self, sample_tensor):
        B, S, _ = sample_tensor.shape
        attn = AttentionPathway(d_model=32, n_heads=4)
        # Causal mask: lower triangular
        mask = torch.tril(torch.ones(S, S, dtype=torch.bool))
        out = attn(sample_tensor, mask=mask)
        assert out.shape == sample_tensor.shape

    def test_gradient_flows(self, sample_tensor):
        attn = AttentionPathway(d_model=32, n_heads=4)
        out = attn(sample_tensor)
        out.sum().backward()
        for p in attn.parameters():
            assert p.grad is not None

    def test_deterministic_eval(self, sample_tensor):
        attn = AttentionPathway(d_model=32, n_heads=4)
        attn.eval()
        out1 = attn(sample_tensor)
        out2 = attn(sample_tensor)
        assert torch.allclose(out1, out2)

    def test_invalid_heads(self):
        with pytest.raises(AssertionError):
            AttentionPathway(d_model=32, n_heads=5)  # 32 not divisible by 5


# ======================================================================
# HybridLayer
# ======================================================================


class TestHybridLayer:
    """Tests for HybridLayer."""

    def test_output_shape(self, sample_tensor):
        layer = HybridLayer(d_model=32, n_heads=4, ssm_state_dim=8)
        out, info = layer(sample_tensor, return_routing=True)
        assert out.shape == sample_tensor.shape
        assert info is not None

    def test_routing_info_keys(self, sample_tensor):
        layer = HybridLayer(d_model=32, n_heads=4, ssm_state_dim=8)
        _, info = layer(sample_tensor, return_routing=True)

        expected_keys = {"routing_weights", "routing_logits", "ssm_ratio", "attn_ratio"}
        assert expected_keys == set(info.keys())

    def test_ratios_sum_to_one(self, sample_tensor):
        layer = HybridLayer(d_model=32, n_heads=4, ssm_state_dim=8)
        _, info = layer(sample_tensor, return_routing=True)
        assert info["ssm_ratio"] + info["attn_ratio"] == pytest.approx(1.0, abs=1e-5)

    def test_no_routing_info_by_default(self, sample_tensor):
        layer = HybridLayer(d_model=32, n_heads=4, ssm_state_dim=8)
        _, info = layer(sample_tensor, return_routing=False)
        assert info is None

    def test_gradient_flows(self, sample_tensor):
        layer = HybridLayer(d_model=32, n_heads=4, ssm_state_dim=8)
        out, _ = layer(sample_tensor)
        out.sum().backward()
        for p in layer.parameters():
            assert p.grad is not None

    def test_with_mask(self, sample_tensor):
        B, S, _ = sample_tensor.shape
        layer = HybridLayer(d_model=32, n_heads=4, ssm_state_dim=8)
        mask = torch.tril(torch.ones(S, S, dtype=torch.bool))
        out, _ = layer(sample_tensor, mask=mask)
        assert out.shape == sample_tensor.shape


# ======================================================================
# DynaRouteBlock
# ======================================================================


class TestDynaRouteBlock:
    """Tests for DynaRouteBlock."""

    def test_output_shape(self, sample_tensor):
        block = DynaRouteBlock(d_model=32, n_heads=4, d_ff=64, ssm_state_dim=8)
        out, info = block(sample_tensor, return_routing=True)
        assert out.shape == sample_tensor.shape
        assert info is not None

    def test_residual_connection(self):
        """Output should differ from input (residual + path contributions)."""
        block = DynaRouteBlock(d_model=32, n_heads=4, d_ff=64, ssm_state_dim=8)
        x = torch.randn(1, 4, 32)
        block.eval()
        out, _ = block(x)
        # Should not be identical (path + FFN + norms change the value)
        assert not torch.allclose(out, x, atol=1e-6)

    def test_gradient_flows(self, sample_tensor):
        block = DynaRouteBlock(d_model=32, n_heads=4, d_ff=64, ssm_state_dim=8)
        out, _ = block(sample_tensor)
        out.sum().backward()
        for p in block.parameters():
            assert p.grad is not None

    def test_parameter_count(self):
        block = DynaRouteBlock(d_model=32, n_heads=4, d_ff=64, ssm_state_dim=8)
        n_params = sum(p.numel() for p in block.parameters())
        assert n_params > 0
