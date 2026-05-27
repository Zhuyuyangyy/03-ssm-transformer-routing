# DynaRoute: Dynamic SSM-Transformer Routing -- Innovation Report

## 1. Problem Statement

Modern sequence models face a fundamental trade-off:

| Architecture | Strength | Weakness |
|---|---|---|
| Transformer (Attention) | Precise local reasoning | O(n^2) time and memory |
| SSM (e.g., Mamba) | O(n) long-range capture | Less precise on local patterns |

Existing approaches force every token through the same computational pathway, regardless of the token's contextual needs. This wastes computation on tokens that could be handled efficiently by SSM, and under-serves tokens that require fine-grained attention.

## 2. Core Innovation: Token-Level Dynamic Routing

DynaRoute introduces a **learned, per-token routing mechanism** that adaptively directs each token through either an SSM pathway or an Attention pathway.

### 2.1 What Makes This Novel

1. **Token-granularity decisions**: Unlike layer-level mixture-of-experts that routes entire sequences, DynaRoute makes routing decisions at the individual token level. A single sequence may have some tokens routed through SSM and others through Attention.

2. **Differentiable discrete routing**: The router uses Gumbel-Softmax during training for fully differentiable routing, switching to hard argmax at inference for zero routing overhead.

3. **Context-aware routing**: The router aggregates global sequence context (via mean pooling + projection) before making per-token decisions, allowing it to reason about a token's role in the broader sequence.

4. **Adaptive temperature annealing**: A three-phase training strategy (warmup -> annealing -> fine-tuning) gradually transitions from soft exploration to hard exploitation of routing decisions.

### 2.2 Key Technical Contributions

| Contribution | Description |
|---|---|
| Dual-pathway hybrid layer | Combines a Mamba-style selective SSM scan with standard multi-head attention in a single layer |
| Gumbel-Softmax router | Lightweight MLP router with learnable bias for balanced routing |
| Capacity-constrained routing | AdaptiveRouter variant limits tokens per path to prevent degenerate collapse |
| Progressive temperature scheduling | Cosine/linear/exponential annealing with dedicated warmup and fine-tuning phases |
| Load-balancing auxiliary loss | KL-divergence from uniform distribution prevents one pathway from dominating |
| Routing analysis toolkit | Post-hoc tools for entropy analysis, pattern discovery, stability measurement, and gradient-based token importance |

## 3. Architecture Overview

```
Input (batch, seq_len, d_model)
        |
   [TokenRouter] --> routing_weights (batch, seq_len, 2)
        |
   +----+----+
   |         |
[SSM Path]  [Attention Path]
   |         |
   +----+----+
        |
  weighted combination
        |
   LayerNorm + FFN
        |
     Output
```

### 3.1 SSM Pathway

Implements a simplified Mamba-style selective state space model:

- Input projection with gating (x, z split)
- Selective parameters: dt (time step), B (input matrix), C (output matrix) are input-dependent
- Sequential selective scan: h_t = exp(A * dt_t) * h_{t-1} + dt_t * B_t * x_t
- Skip connection via learnable D parameter
- Output gating via sigmoid(z)

### 3.2 Attention Pathway

Standard multi-head self-attention with modern optimizations:

- Fused QKV projection (single linear layer)
- Automatic dispatch to FlashAttention / memory-efficient attention via `torch.nn.functional.scaled_dot_product_attention`
- Optional causal masking

### 3.3 Router

Two variants provided:

- **TokenRouter**: Standard router with Gumbel-Softmax training, context aggregation, and learnable routing bias.
- **AdaptiveRouter**: Adds learnable temperature (log-parameterized) and capacity-constrained hard routing.

## 4. Training Strategy

Three-phase progressive training:

| Phase | Temperature | Router | Purpose |
|---|---|---|---|
| Warmup (epochs 0..W) | High (start_temp) | Trainable | Explore routing space broadly |
| Annealing (epochs W..T-F) | Decaying | Trainable | Gradually commit to routing decisions |
| Fine-tuning (epochs T-F..T) | Low (end_temp) | Frozen | Refine pathway parameters |

## 5. Analysis Toolkit

The `RoutingAnalyzer` class provides:

- **Per-type analysis**: How different token types (content, function, stopword) are routed
- **Per-position analysis**: Routing patterns along the sequence dimension
- **Pattern discovery**: Finding contiguous routing blocks across a dataset
- **Stability analysis**: Measuring routing consistency across stochastic forward passes
- **Token importance**: Gradient-based attribution of routing decisions to input tokens

## 6. Comparison to Related Work

| Method | Routing Granularity | Pathways | Capacity Control |
|---|---|---|---|
| Switch Transformer | Layer-level | N experts | Yes |
| Mixture of Experts | Layer-level | N experts | Yes |
| Jamba (AI21) | Layer-level | SSM or Attention | No |
| **DynaRoute** | **Token-level** | **SSM + Attention** | **Yes (AdaptiveRouter)** |

DynaRoute is, to our knowledge, the first architecture to perform **token-level** routing between SSM and Transformer pathways within a single layer.

## 7. Potential Applications

- **Long-context language modeling**: Route most tokens through SSM for O(n) processing; reserve attention for tokens requiring precise reasoning.
- **Efficient inference**: Hard routing at inference eliminates attention computation for SSM-routed tokens.
- **Interpretable models**: Routing patterns reveal which tokens the model considers "attention-worthy."
- **Adaptive compute**: Naturally allocates more FLOPs to harder tokens without explicit budget mechanisms.
