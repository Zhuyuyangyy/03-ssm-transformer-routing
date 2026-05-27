"""Progressive training with temperature annealing for DynaRoute.

This module implements training strategies for the hybrid SSM-Transformer model,
including progressive training schedules and temperature annealing.
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass


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


class TemperatureScheduler:
    """Temperature annealing scheduler for routing decisions.

    Supports multiple annealing strategies:
    - Linear: Linear decay from start to end temperature
    - Cosine: Cosine annealing with warm restarts
    - Exponential: Exponential decay

    Args:
        start_temp: Starting temperature.
        end_temp: Ending temperature.
        total_steps: Total number of training steps.
        schedule: Annealing schedule type.
    """

    def __init__(
        self,
        start_temp: float = 5.0,
        end_temp: float = 0.5,
        total_steps: int = 1000,
        schedule: str = "cosine",
    ):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.total_steps = total_steps
        self.schedule = schedule
        self.current_step = 0

    def step(self) -> float:
        """Update and return current temperature.

        Returns:
            Current temperature value.
        """
        self.current_step += 1
        progress = min(self.current_step / self.total_steps, 1.0)

        if self.schedule == "linear":
            temp = self.start_temp + (self.end_temp - self.start_temp) * progress
        elif self.schedule == "cosine":
            temp = self.end_temp + 0.5 * (self.start_temp - self.end_temp) * \
                   (1 + math.cos(math.pi * progress))
        elif self.schedule == "exponential":
            decay_rate = math.log(self.end_temp / self.start_temp) / self.total_steps
            temp = self.start_temp * math.exp(decay_rate * self.current_step)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return max(temp, self.end_temp)

    def get_temperature(self) -> float:
        """Get current temperature without stepping.

        Returns:
            Current temperature value.
        """
        progress = min(self.current_step / self.total_steps, 1.0)

        if self.schedule == "linear":
            return self.start_temp + (self.end_temp - self.start_temp) * progress
        elif self.schedule == "cosine":
            return self.end_temp + 0.5 * (self.start_temp - self.end_temp) * \
                   (1 + math.cos(math.pi * progress))
        elif self.schedule == "exponential":
            decay_rate = math.log(self.end_temp / self.start_temp) / self.total_steps
            return self.start_temp * math.exp(decay_rate * self.current_step)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")


class ProgressiveTrainer:
    """Progressive trainer for DynaRoute models.

    Implements a multi-phase training strategy:
    1. Warmup phase: High temperature for soft routing exploration
    2. Annealing phase: Gradually reduce temperature for harder routing
    3. Fine-tuning phase: Low temperature with frozen router

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
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

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

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.training_history = []

    def train(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run the complete training process.

        Args:
            progress_callback: Optional callback for progress updates.

        Returns:
            Dictionary with training history and final metrics.
        """
        for epoch in range(self.config.total_epochs):
            self.current_epoch = epoch

            # Determine training phase
            if epoch < self.config.warmup_epochs:
                phase = "warmup"
            elif epoch < self.config.total_epochs - 10:
                phase = "annealing"
            else:
                phase = "finetuning"

            # Train one epoch
            train_metrics = self._train_epoch(phase)

            # Validate
            val_metrics = self._validate() if self.val_loader else {}

            # Update temperature
            current_temp = self.temp_scheduler.get_temperature()

            # Log metrics
            epoch_metrics = {
                "epoch": epoch,
                "phase": phase,
                "temperature": current_temp,
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

    def _train_epoch(self, phase: str) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            phase: Current training phase ("warmup", "annealing", "finetuning").

        Returns:
            Dictionary with epoch training metrics.
        """
        self.model.train()
        total_loss = 0.0
        total_main_loss = 0.0
        total_balance_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # Move to device
            inputs = batch["input_ids"].to(self.device)
            targets = batch["labels"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs, routing_info = self.model(inputs, return_routing=True)

            # Compute losses
            main_loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            balance_loss = 0.0
            if routing_info and "routing_weights" in routing_info:
                balance_loss = self._compute_balance_loss(routing_info["routing_weights"])

            # Total loss
            loss = main_loss + self.config.balance_loss_weight * balance_loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            self.optimizer.step()

            # Update temperature
            if phase == "annealing":
                self.temp_scheduler.step()

            # Freeze router in finetuning phase
            if phase == "finetuning":
                self._freeze_router()

            # Accumulate metrics
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_balance_loss += balance_loss if isinstance(balance_loss, float) else balance_loss.item()
            num_batches += 1
            self.global_step += 1

        return {
            "loss": total_loss / num_batches,
            "main_loss": total_main_loss / num_batches,
            "balance_loss": total_balance_loss / num_batches,
        }

    def _validate(self) -> Dict[str, float]:
        """Validate the model.

        Returns:
            Dictionary with validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["input_ids"].to(self.device)
                targets = batch["labels"].to(self.device)

                outputs, _ = self.model(inputs, return_routing=False)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

                total_loss += loss.item()
                num_batches += 1

        return {"loss": total_loss / num_batches}

    def _compute_balance_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss.

        Args:
            routing_weights: Routing weights (batch, seq_len, num_paths).

        Returns:
            Scalar balance loss.
        """
        # Average routing probability per path
        avg_probs = routing_weights.mean(dim=[0, 1])
        target = torch.ones_like(avg_probs) / routing_weights.size(-1)
        return torch.nn.functional.kl_div(avg_probs.log(), target, reduction="batchmean")

    def _freeze_router(self):
        """Freeze router parameters for fine-tuning phase."""
        for name, param in self.model.named_parameters():
            if "router" in name:
                param.requires_grad = False

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename.
        """
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
        }, filename)
