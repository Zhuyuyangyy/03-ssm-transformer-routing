"""Tests for the training module."""

import math
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dynaroute.training import TrainingConfig, TemperatureScheduler, ProgressiveTrainer


# ======================================================================
# TrainingConfig
# ======================================================================


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.total_epochs == 100
        assert cfg.warmup_epochs == 10
        assert cfg.temperature_schedule == "cosine"

    def test_validation_bad_epochs(self):
        with pytest.raises(AssertionError):
            TrainingConfig(total_epochs=0)

    def test_validation_warmup_too_large(self):
        with pytest.raises(AssertionError):
            TrainingConfig(total_epochs=10, warmup_epochs=10)

    def test_validation_bad_schedule(self):
        with pytest.raises(AssertionError):
            TrainingConfig(temperature_schedule="unknown")

    def test_validation_temp_order(self):
        with pytest.raises(AssertionError):
            TrainingConfig(temperature_start=0.5, temperature_end=5.0)


# ======================================================================
# TemperatureScheduler
# ======================================================================


class TestTemperatureScheduler:
    """Tests for TemperatureScheduler."""

    def test_starts_at_start_temp(self):
        sched = TemperatureScheduler(start_temp=5.0, end_temp=0.5, total_steps=100)
        assert sched.get_temperature() == 5.0

    def test_ends_at_end_temp(self):
        sched = TemperatureScheduler(start_temp=5.0, end_temp=0.5, total_steps=100)
        for _ in range(100):
            temp = sched.step()
        assert temp == pytest.approx(0.5, abs=1e-4)

    def test_linear_decreasing(self):
        sched = TemperatureScheduler(
            start_temp=10.0, end_temp=1.0, total_steps=10, schedule="linear"
        )
        prev = sched.get_temperature()
        for _ in range(10):
            curr = sched.step()
            assert curr <= prev + 1e-6
            prev = curr

    def test_cosine_decreasing(self):
        sched = TemperatureScheduler(
            start_temp=10.0, end_temp=1.0, total_steps=10, schedule="cosine"
        )
        temps = [sched.step() for _ in range(10)]
        # Should be monotonically decreasing
        for i in range(1, len(temps)):
            assert temps[i] <= temps[i - 1] + 1e-6

    def test_exponential_decreasing(self):
        sched = TemperatureScheduler(
            start_temp=10.0, end_temp=1.0, total_steps=10, schedule="exponential"
        )
        temps = [sched.step() for _ in range(10)]
        for i in range(1, len(temps)):
            assert temps[i] <= temps[i - 1] + 1e-6

    def test_invalid_schedule(self):
        with pytest.raises(ValueError):
            TemperatureScheduler(schedule="bad")

    def test_invalid_total_steps(self):
        with pytest.raises(ValueError):
            TemperatureScheduler(total_steps=0)

    def test_step_advances_counter(self):
        sched = TemperatureScheduler(total_steps=100)
        assert sched.current_step == 0
        sched.step()
        assert sched.current_step == 1

    def test_get_temperature_does_not_advance(self):
        sched = TemperatureScheduler(total_steps=100)
        sched.step()
        sched.get_temperature()
        assert sched.current_step == 1

    def test_clamps_at_end_temp(self):
        sched = TemperatureScheduler(
            start_temp=10.0, end_temp=0.5, total_steps=5, schedule="linear"
        )
        for _ in range(20):  # overshoot
            temp = sched.step()
        assert temp >= 0.5 - 1e-6


# ======================================================================
# ProgressiveTrainer
# ======================================================================


def _make_dummy_loader(num_samples=16, seq_len=8, vocab_size=32, batch_size=4):
    """Create a dummy data loader for testing."""
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = torch.randint(0, vocab_size, (num_samples, seq_len))
    dataset = TensorDataset(input_ids, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _make_simple_model(vocab_size=32, d_model=16, n_heads=2, ssm_state_dim=4):
    """Create a minimal model for testing the trainer."""
    from dynaroute.hybrid_layer import DynaRouteBlock

    class SimpleModel(nn.Module):
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

    return SimpleModel()


class TestProgressiveTrainer:
    """Tests for ProgressiveTrainer."""

    def test_basic_training_runs(self, tmp_path):
        model = _make_simple_model()
        train_loader = _make_dummy_loader()
        cfg = TrainingConfig(
            total_epochs=2, warmup_epochs=1, checkpoint_dir=str(tmp_path),
        )
        trainer = ProgressiveTrainer(
            model=model, config=cfg, train_loader=train_loader, device="cpu",
        )
        results = trainer.train()

        assert "history" in results
        assert "best_val_loss" in results
        assert len(results["history"]) == 2

    def test_training_with_validation(self, tmp_path):
        model = _make_simple_model()
        train_loader = _make_dummy_loader()
        val_loader = _make_dummy_loader(num_samples=8)
        cfg = TrainingConfig(
            total_epochs=2, warmup_epochs=1, checkpoint_dir=str(tmp_path),
        )
        trainer = ProgressiveTrainer(
            model=model, config=cfg, train_loader=train_loader,
            val_loader=val_loader, device="cpu",
        )
        results = trainer.train()

        assert results["best_val_loss"] < float("inf")

    def test_phases_correct(self, tmp_path):
        model = _make_simple_model()
        train_loader = _make_dummy_loader()
        cfg = TrainingConfig(
            total_epochs=10, warmup_epochs=3, finetune_epochs=3,
            checkpoint_dir=str(tmp_path),
        )
        trainer = ProgressiveTrainer(
            model=model, config=cfg, train_loader=train_loader, device="cpu",
        )

        phases = []
        def callback(metrics):
            phases.append(metrics["phase"])

        trainer.train(progress_callback=callback)

        assert phases[0] == "warmup"
        assert phases[4] == "annealing"
        assert phases[-1] == "finetuning"

    def test_progress_callback_called(self, tmp_path):
        model = _make_simple_model()
        train_loader = _make_dummy_loader()
        cfg = TrainingConfig(total_epochs=3, warmup_epochs=1, checkpoint_dir=str(tmp_path))
        trainer = ProgressiveTrainer(
            model=model, config=cfg, train_loader=train_loader, device="cpu",
        )

        called = []
        trainer.train(progress_callback=lambda m: called.append(m["epoch"]))

        assert called == [0, 1, 2]

    def test_unpack_batch_dict(self):
        batch = {"input_ids": torch.zeros(2), "labels": torch.ones(2)}
        inputs, labels = ProgressiveTrainer._unpack_batch(batch)
        assert torch.equal(inputs, torch.zeros(2))
        assert torch.equal(labels, torch.ones(2))

    def test_unpack_batch_tuple(self):
        batch = (torch.zeros(2), torch.ones(2))
        inputs, labels = ProgressiveTrainer._unpack_batch(batch)
        assert torch.equal(inputs, torch.zeros(2))
        assert torch.equal(labels, torch.ones(2))

    def test_loss_decreases(self, tmp_path):
        """Sanity check: training loss should decrease over epochs."""
        torch.manual_seed(42)
        model = _make_simple_model()
        train_loader = _make_dummy_loader(num_samples=32)
        cfg = TrainingConfig(
            total_epochs=5, warmup_epochs=1, checkpoint_dir=str(tmp_path),
        )
        trainer = ProgressiveTrainer(
            model=model, config=cfg, train_loader=train_loader, device="cpu",
        )
        results = trainer.train()

        first_loss = results["history"][0]["loss"]
        last_loss = results["history"][-1]["loss"]
        assert last_loss < first_loss
