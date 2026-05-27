"""Progressive training with temperature annealing for DynaRoute.

This module implements training strategies for the hybrid SSM-Transformer model,
including progressive training schedules and temperature annealing.

Training phases:
1. **Warmup**: High temperature for soft routing exploration.
2. **Annealing**: Gradually reduce temperature for harder routing.
3. **Fine-tuning**: Low temperature with frozen router.
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, Any, Union, List
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["TrainingConfig", "TemperatureScheduler", "ProgressiveTrainer"]


@dataclass
class TrainingConfig:
    """Configuration for progressive training.

    Attributes:
        total_epochs: Total number of training epochs.
        warmup_epochs: Number of warmup epochs.
        temperature_start: Initial routing temperature.
        temperature_end: Final routing temperature.
        temperature_schedule: Temperature annealing schedule type.
        learning_rate: Peak learning rate.
        weight_decay: Weight decay for optimizer.
        grad_clip: Gradient clipping value.
        balance_loss_weight: Weight for load balancing loss.
        aux_loss_weight: Weight for auxiliary routing loss.
        checkpoint_dir: Directory for saving checkpoints.
        finetune_epochs: Number of fine-tuning epochs at the end.
    """
    total_epochs: int = 100
    warmup_epochs: int = 10
    temperature_start: float = 5.0
    temperature_end: float = 0.5
    temperature_schedule: str = "cosine"  # "linear", "cosine", "exponential"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    balance_loss_weight: float = 0.01
    aux_loss_weight: float = 0.001
    checkpoint_dir: str = "checkpoints"
    finetune_epochs: int = 10

    def __post_init__(self):
        """Validate configuration."""
        assert self.total_epochs > 0, "total_epochs must be positive"
        assert 0 <= self.warmup_epochs < self.total_epochs, (
            f"warmup_epochs ({self.warmup_epochs}) must be in [0, {self.total_epochs})"
        )
        assert self.temperature_start >= self.temperature_end > 0, (
            "temperature_start must be >= temperature_end > 0"
        )
        assert self.temperature_schedule in ("linear", "cosine", "exponential"), (
            f"Unknown schedule: {self.temperature_schedule}"
        )


class TemperatureScheduler:
    """Temperature annealing scheduler for routing decisions.

    Supports multiple annealing strategies:
    - **linear**: Linear decay from start to end temperature.
    - **cosine**: Cosine annealing (smooth deceleration near the end).
    - **exponential**: Exponential decay for rapid cooling.

    Args:
        start_temp: Starting temperature.
        end_temp: Ending temperature.
        total_steps: Total number of training steps.
        schedule: Annealing schedule type.
    """

    SCHEDULES = ("linear", "cosine", "exponential")

    def __init__(
        self,
        start_temp: float = 5.0,
        end_temp: float = 0.5,
        total_steps: int = 1000,
        schedule: str = "cosine",
    ):
        if schedule not in self.SCHEDULES:
            raise ValueError(
                f"Unknown schedule: {schedule!r}. Choose from {self.SCHEDULES}"
            )
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")

        self.start_temp = start_temp
        self.end_temp = end_temp
        self.total_steps = total_steps
        self.schedule = schedule
        self.current_step = 0

        # Pre-compute decay rate for exponential schedule
        self._exp_decay_rate: Optional[float] = None
        if schedule == "exponential":
            self._exp_decay_rate = math.log(end_temp / start_temp) / total_steps

    def _compute_temperature(self, step: int) -> float:
        """Compute temperature for a given step (pure function).

        Args:
            step: Current step index.

        Returns:
            Temperature value.
        """
        progress = min(step / self.total_steps, 1.0)

        if self.schedule == "linear":
            return self.start_temp + (self.end_temp - self.start_temp) * progress

        if self.schedule == "cosine":
            return (
                self.end_temp
                + 0.5 * (self.start_temp - self.end_temp)
                * (1 + math.cos(math.pi * progress))
            )

        # exponential
        return self.start_temp * math.exp(self._exp_decay_rate * step)

    def step(self) -> float:
        """Advance one step and return the new temperature.

        Returns:
            Current temperature value.
        """
        self.current_step += 1
        temp = self._compute_temperature(self.current_step)
        return max(temp, self.end_temp)

    def get_temperature(self) -> float:
        """Get current temperature without stepping.

        Returns:
            Current temperature value.
        """
        temp = self._compute_temperature(self.current_step)
        return max(temp, self.end_temp)


class ProgressiveTrainer:
    """Progressive trainer for DynaRoute models.

    Implements a multi-phase training strategy:
    1. **Warmup** phase: High temperature for soft routing exploration.
    2. **Annealing** phase: Gradually reduce temperature for harder routing.
    3. **Fine-tuning** phase: Low temperature with frozen router.

    The trainer accepts batches as either:
    - A dict with ``"input_ids"`` and ``"labels"`` keys, or
    - A tuple/list of ``(input_ids, labels)``.

    Args:
        model: The DynaRoute model to train.
        config: Training configuration.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: PyTorch optimizer. If None, creates AdamW.
        device: Training device.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        device: str = "cuda",
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Move model to device
        self.model = model.to(device)

        # Setup optimizer
        self.optimizer = optimizer or optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup temperature scheduler
        total_steps = config.total_epochs * len(train_loader)
        self.temp_scheduler = TemperatureScheduler(
            start_temp=config.temperature_start,
            end_temp=config.temperature_end,
            total_steps=total_steps,
            schedule=config.temperature_schedule,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.training_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run the complete training process.

        Args:
            progress_callback: Optional callback invoked after each epoch with
                a metrics dictionary.

        Returns:
            Dictionary with training history and final metrics.
        """
        for epoch in range(self.config.total_epochs):
            self.current_epoch = epoch

            # Determine training phase
            phase = self._get_phase(epoch)

            # Train one epoch
            train_metrics = self._train_epoch(phase)

            # Validate
            val_metrics = self._validate() if self.val_loader else {}

            # Current temperature
            current_temp = self.temp_scheduler.get_temperature()

            # Log metrics
            epoch_metrics = {
                "epoch": epoch,
                "phase": phase,
                "temperature": current_temp,
                "global_step": self.global_step,
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.training_history.append(epoch_metrics)

            # Progress callback
            if progress_callback:
                progress_callback(epoch_metrics)

            # Save best model
            if val_metrics and val_metrics.get("loss", float("inf")) < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self._save_checkpoint("best_model.pt")

        return {
            "history": self.training_history,
            "best_val_loss": self.best_val_loss,
            "total_epochs": self.config.total_epochs,
        }

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def _get_phase(self, epoch: int) -> str:
        """Determine the current training phase.

        Args:
            epoch: Current epoch index.

        Returns:
            One of ``"warmup"``, ``"annealing"``, ``"finetuning"``.
        """
        finetune_start = self.config.total_epochs - self.config.finetune_epochs
        if epoch < self.config.warmup_epochs:
            return "warmup"
        if epoch < finetune_start:
            return "annealing"
        return "finetuning"

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_batch(
        batch: Union[Dict[str, torch.Tensor], tuple, list],
    ) -> tuple:
        """Unpack a batch into (input_ids, labels).

        Supports both dict-style ``{"input_ids": ..., "labels": ...}``
        and tuple-style ``(input_ids, labels)`` batches.

        Args:
            batch: A batch from the data loader.

        Returns:
            Tuple of (input_ids, labels).
        """
        if isinstance(batch, dict):
            return batch["input_ids"], batch["labels"]
        if isinstance(batch, (tuple, list)):
            return batch[0], batch[1]
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    # ------------------------------------------------------------------
    # Core training / validation loops
    # ------------------------------------------------------------------

    def _train_epoch(self, phase: str) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            phase: Current training phase.

        Returns:
            Dictionary with epoch training metrics.
        """
        self.model.train()

        # Freeze router during fine-tuning
        if phase == "finetuning":
            self._freeze_router()

        total_loss = 0.0
        total_main_loss = 0.0
        total_balance_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            inputs, targets = self._unpack_batch(batch)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs, routing_info = self.model(inputs, return_routing=True)

            # Compute losses
            vocab_size = outputs.size(-1)
            main_loss = self.criterion(
                outputs.reshape(-1, vocab_size), targets.reshape(-1)
            )

            balance_loss = torch.tensor(0.0, device=self.device)
            if routing_info and "routing_weights" in routing_info:
                balance_loss = self._compute_balance_loss(routing_info["routing_weights"])

            loss = main_loss + self.config.balance_loss_weight * balance_loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            self.optimizer.step()

            # Update temperature (only during annealing phase)
            if phase == "annealing":
                self.temp_scheduler.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_balance_loss += balance_loss.item()
            num_batches += 1
            self.global_step += 1

        return {
            "loss": total_loss / max(num_batches, 1),
            "main_loss": total_main_loss / max(num_batches, 1),
            "balance_loss": total_balance_loss / max(num_batches, 1),
        }

    def _validate(self) -> Dict[str, float]:
        """Validate the model.

        Returns:
            Dictionary with validation metrics.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = self._unpack_batch(batch)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs, _ = self.model(inputs, return_routing=False)
                vocab_size = outputs.size(-1)
                loss = self.criterion(
                    outputs.reshape(-1, vocab_size), targets.reshape(-1)
                )

                total_loss += loss.item()
                num_batches += 1

        return {"loss": total_loss / max(num_batches, 1)}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _compute_balance_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss.

        Args:
            routing_weights: Routing weights (batch, seq_len, num_paths).

        Returns:
            Scalar balance loss.
        """
        num_paths = routing_weights.size(-1)
        avg_probs = routing_weights.mean(dim=[0, 1])
        target = torch.ones_like(avg_probs) / num_paths
        return F.kl_div(
            avg_probs.log(), target, reduction="batchmean"
        )

    def _freeze_router(self):
        """Freeze router parameters for fine-tuning phase."""
        for name, param in self.model.named_parameters():
            if "router" in name:
                param.requires_grad = False

    def _unfreeze_router(self):
        """Unfreeze router parameters (useful between phases)."""
        for name, param in self.model.named_parameters():
            if "router" in name:
                param.requires_grad = True

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename (relative to checkpoint_dir).
        """
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
        }, path)

    def load_checkpoint(self, filename: str) -> Dict[str, Any]:
        """Load a model checkpoint.

        Args:
            filename: Checkpoint filename (relative to checkpoint_dir).

        Returns:
            Checkpoint metadata dictionary.
        """
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.training_history = checkpoint.get("training_history", [])
        return checkpoint


# Re-export F for _compute_balance_loss
import torch.nn.functional as F  # noqa: E402
