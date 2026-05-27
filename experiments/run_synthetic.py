"""Synthetic data experiments for DynaRoute.

This script evaluates DynaRoute on synthetic tasks designed to test
the model's ability to route tokens appropriately based on task demands.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dynaroute import HybridLayer, TokenRouter, RoutingAnalyzer


def generate_copy_task(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
) -> TensorDataset:
    """Generate copy task data.

    Model must copy the first half of the sequence to the second half.

    Args:
        num_samples: Number of samples.
        seq_len: Sequence length (must be even).
        vocab_size: Vocabulary size.

    Returns:
        TensorDataset with input and target sequences.
    """
    assert seq_len % 2 == 0, "Sequence length must be even"

    half_len = seq_len // 2
    inputs = torch.randint(1, vocab_size, (num_samples, half_len))
    separator = torch.zeros(num_samples, 1, dtype=torch.long)
    targets = inputs.clone()

    # Input: [pattern] [sep] [zeros]
    # Target: [pattern] [sep] [pattern]
    input_seq = torch.cat([inputs, separator, torch.zeros(num_samples, half_len - 1, dtype=torch.long)], dim=1)
    target_seq = torch.cat([inputs, separator, inputs[:, :half_len - 1]], dim=1)

    return TensorDataset(input_seq, target_seq)


def generate_sorting_task(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
) -> TensorDataset:
    """Generate sorting task data.

    Model must sort the input sequence.

    Args:
        num_samples: Number of samples.
        seq_len: Sequence length.
        vocab_size: Vocabulary size.

    Returns:
        TensorDataset with unsorted input and sorted target.
    """
    inputs = torch.randint(0, vocab_size, (num_samples, seq_len))
    targets, _ = inputs.sort(dim=1)

    return TensorDataset(inputs, targets)


def generate_lookup_task(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
) -> TensorDataset:
    """Generate key-value lookup task data.

    Model must retrieve values based on keys in the sequence.

    Args:
        num_samples: Number of samples.
        seq_len: Sequence length (must be divisible by 4).
        vocab_size: Vocabulary size.

    Returns:
        TensorDataset with key-value pairs and query.
    """
    assert seq_len % 4 == 0, "Sequence length must be divisible by 4"

    num_pairs = seq_len // 4
    keys = torch.randint(1, vocab_size // 2, (num_samples, num_pairs))
    values = torch.randint(vocab_size // 2, vocab_size, (num_samples, num_pairs))

    # Build sequence: [k1, v1, k2, v2, ..., query]
    kv_pairs = torch.stack([keys, values], dim=2).reshape(num_samples, -1)
    query_idx = torch.randint(0, num_pairs, (num_samples,))
    query = keys[torch.arange(num_samples), query_idx]

    inputs = torch.cat([kv_pairs, query.unsqueeze(1)], dim=1)
    targets = values[torch.arange(num_samples), query_idx].unsqueeze(1).expand(-1, seq_len)

    return TensorDataset(inputs, targets)


TASK_GENERATORS = {
    "copy": generate_copy_task,
    "sort": generate_sorting_task,
    "lookup": generate_lookup_task,
}


def main(args: argparse.Namespace):
    """Run synthetic data experiment.

    Args:
        args: Command line arguments.
    """
    print(f"Running synthetic experiment: {args.pattern}")
    print(f"Config: seq_len={args.seq_len}, vocab_size={args.vocab_size}")

    # Generate data
    generator = TASK_GENERATORS[args.pattern]
    dataset = generator(args.num_samples, args.seq_len, args.vocab_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Build simple model for testing
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
        nn.Embedding(args.vocab_size, args.d_model),
        HybridLayer(d_model=args.d_model, n_heads=args.n_heads),
        nn.Linear(args.d_model, args.vocab_size),
    ).to(device)

    # Analyze routing before training
    print("\nAnalyzing initial routing patterns...")
    analyzer = RoutingAnalyzer(model[1], device=device)

    sample_input, _ = dataset[0]
    sample_input = sample_input.unsqueeze(0).to(device)

    initial_analysis = analyzer.analyze_routing(sample_input)
    print(f"Initial SSM ratio: {initial_analysis['ssm_ratio']:.2%}")

    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining...")
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            # Forward pass through embedding and hybrid layer
            x = model[0](batch_inputs)  # Embedding
            x, routing_info = model[1](x, return_routing=True)  # HybridLayer
            logits = model[2](x)  # Linear head

            # Compute loss
            loss = criterion(logits.view(-1, args.vocab_size), batch_targets.view(-1))

            # Add balance loss
            if routing_info:
                balance_loss = model[1].router.get_load_balance_loss(
                    routing_info["routing_weights"]
                )
                loss = loss + 0.01 * balance_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}")

    # Analyze routing after training
    print("\nAnalyzing trained routing patterns...")
    final_analysis = analyzer.analyze_routing(sample_input)
    print(f"Final SSM ratio: {final_analysis['ssm_ratio']:.2%}")
    print(f"Final Attention ratio: {final_analysis['attn_ratio']:.2%}")

    # Compute routing stability
    print("\nComputing routing stability...")
    stability = analyzer.compute_routing_stability(sample_input, num_trials=5)
    print(f"Routing agreement rate: {stability['agreement_rate']:.2%}")
    print(f"Stability score: {stability['stability_score']:.4f}")

    # Export results
    output_dir = Path(args.output_dir) / args.pattern
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer.export_analysis(final_analysis, str(output_dir / "routing_analysis.json"))

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DynaRoute synthetic experiments")

    parser.add_argument("--pattern", type=str, default="copy",
                       choices=["copy", "sort", "lookup"],
                       help="Synthetic task pattern")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=100, help="Vocabulary size")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--output-dir", type=str, default="results/synthetic", help="Output directory")

    args = parser.parse_args()
    main(args)
