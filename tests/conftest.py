"""Shared fixtures for dynaroute tests."""

import sys
from pathlib import Path

import pytest
import torch

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Dimensions used across tests (small for speed)
# ---------------------------------------------------------------------------
BATCH = 2
SEQ_LEN = 8
D_MODEL = 32
N_HEADS = 4
VOCAB_SIZE = 64
SSM_STATE_DIM = 8


@pytest.fixture()
def d_model():
    return D_MODEL


@pytest.fixture()
def n_heads():
    return N_HEADS


@pytest.fixture()
def batch():
    return BATCH


@pytest.fixture()
def seq_len():
    return SEQ_LEN


@pytest.fixture()
def sample_tensor(batch, seq_len, d_model):
    """Random float tensor (batch, seq_len, d_model)."""
    return torch.randn(batch, seq_len, d_model)


@pytest.fixture()
def sample_input_ids(batch, seq_len, vocab_size=VOCAB_SIZE):
    """Random integer token IDs (batch, seq_len)."""
    return torch.randint(0, vocab_size, (batch, seq_len))


@pytest.fixture()
def device():
    return "cpu"
