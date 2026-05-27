"""Main routing experiment for DynaRoute.

This script trains and evaluates the DynaRoute hybrid model on
standard language modeling benchmarks.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dynaroute import HybridLayer, TokenRouter, ProgressiveTrainer, RoutingAnalyzer


def create_dummy_dataset(
    vocab_size: int = 1000,
    seq_len: int = 128,
    num_samples: int = 1000,
) -> TensorDataset:
    """Create dummy dataset for testing.

    Args:
        vocab_size: Vocabulary size.
        seq_len: Sequence length.
        num_samples: Number of samples.

    Returns:
        TensorDataset with input_ids and labels.
    """
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    return TensorDataset(input_ids, labels)


def build_model(args: argparse.Namespace) -> nn.Module:
    """Build DynaRoute model from arguments.

    Args:
        args: Command line arguments.

    Returns:
        DynaRoute model.
    """
    from dynaroute.hybrid_layer import DynaRouteBlock

    class DynaRouteModel(nn.Module):
        """Complete DynaRoute model for language modeling."""

        def __init__(self, vocab_size, d_model, n_layers, n_heads, ssm_state_dim, temperature):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([
                DynaRouteBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ssm_state_dim=ssm_state_dim,
                    temperature=temperature,
                )
                for _ in range(n_layers)
            ])
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, input_ids, return_routing=False):
            x = self.embedding(input_ids)
            routing_infos = []

            for layer in self.layers:
                x, routing_info = layer(x, return_routing=return_routing)
                if routing_info:
                    routing_infos.append(routing_info)

            x = self.norm(x)
            logits = self.head(x)

            if return_routing:
                # Aggregate routing info from all layers
                avg_info = {
                    "routing_weights": torch.stack([r["routing_weights"] for r in routing_infos]).mean(0),
                    "routing_logits": torch.stack([r["routing_logits"] for r in routing_infos]).mean(0),
                    "ssm_ratio": sum(r["ssm_ratio"] for r in routing_infos) / len(routing_infos),
                    "attn_ratio": sum(r["attn_ratio"] for r in routing_infos) / len(routing_infos),
                }
                return logits, avg_info
            return logits, None

    return DynaRouteModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ssm_state_dim=args.ssm_state_dim,
        temperature=args.temperature,
    )


def main(args: argparse.Namespace):
    """Run routing experiment.

    Args:
        args: Command line arguments.
    """
    print(f"Running routing experiment with config: {vars(args)}")

    # Setup device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = create_dummy_dataset(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        num_samples=args.num_train,
    )
    val_dataset = create_dummy_dataset(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        num_samples=args.num_val,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Build model
    model = build_model(args)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup trainer
    from dynaroute.training import TrainingConfig

    config = TrainingConfig(
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        learning_rate=args.lr,
        balance_loss_weight=args.balance_loss_weight,
    )

    trainer = ProgressiveTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # Train
    def progress_callback(metrics):
        print(f"Epoch {metrics['epoch']}: loss={metrics['loss']:.4f}, "
              f"temp={metrics['temperature']:.3f}, phase={metrics['phase']}")

    results = trainer.train(progress_callback=progress_callback)

    # Analyze routing patterns
    print("\nAnalyzing routing patterns...")
    analyzer = RoutingAnalyzer(model, device=device)

    # Get a batch for analysis
    sample_batch = next(iter(val_loader))
    sample_input = sample_batch[0][:4]  # First 4 samples

    analysis = analyzer.analyze_routing(sample_input)
    print(f"Average SSM ratio: {analysis['ssm_ratio']:.2%}")
    print(f"Average Attention ratio: {analysis['attn_ratio']:.2%}")

    # Export results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer.export_analysis(analysis, str(output_dir / "routing_analysis.json"))

    with open(output_dir / "training_results.json", "w") as f:
        json.dump({
            "best_val_loss": results["best_val_loss"],
            "total_epochs": results["total_epochs"],
            "config": vars(args),
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DynaRoute routing experiment")

    # Model arguments
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--ssm-state-dim", type=int, default=32, help="SSM state dimension")
    parser.add_argument("--temperature", type=float, default=1.0, help="Initial routing temperature")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--temperature-start", type=float, default=5.0, help="Starting temperature")
    parser.add_argument("--temperature-end", type=float, default=0.5, help="Ending temperature")
    parser.add_argument("--balance-loss-weight", type=float, default=0.01, help="Balance loss weight")

    # Data arguments
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--num-train", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--num-val", type=int, default=200, help="Number of validation samples")

    # Other arguments
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--output-dir", type=str, default="results/routing", help="Output directory")

    args = parser.parse_args()
    main(args)
