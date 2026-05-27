# SSM-Transformer Routing (DynaRoute)

A hybrid architecture that dynamically routes tokens between SSM (State Space Model) and Transformer attention pathways, enabling efficient sequence modeling with adaptive computation.

## Overview

DynaRoute introduces a token-level routing mechanism that selectively directs each token through either an SSM pathway (for efficient long-range dependency capture) or an Attention pathway (for precise local context modeling). This approach achieves:

- **Adaptive Computation**: Tokens are routed based on their contextual needs
- **Efficiency**: SSM pathway handles bulk processing with O(n) complexity
- **Quality**: Attention pathway focuses on tokens requiring fine-grained reasoning

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from dynaroute import HybridLayer, TokenRouter

# Create a hybrid layer
layer = HybridLayer(
    d_model=512,
    n_heads=8,
    ssm_state_dim=64,
    temperature=1.0
)

# Forward pass with routing
output, routing_info = layer(input_ids, return_routing=True)
print(f"SSM ratio: {routing_info['ssm_ratio']:.2%}")
```

## Project Structure

```
src/dynaroute/
├── hybrid_layer.py      # Dual-path hybrid layer (SSM + Attention)
├── router.py            # Token-level router
├── training.py          # Progressive training with temperature annealing
└── analysis.py          # Routing pattern analysis
experiments/
├── run_routing.py       # Main routing experiment
└── run_synthetic.py     # Synthetic data experiments
```

## Experiments

```bash
# Run routing experiment
python experiments/run_routing.py --model-size base --epochs 50

# Run synthetic data experiment
python experiments/run_synthetic.py --seq-len 4096 --pattern copy
```

## Citation

```bibtex
@article{dynaroute2024,
  title={DynaRoute: Dynamic SSM-Transformer Routing for Efficient Sequence Modeling},
  author={Research Team},
  year={2024}
}
```
