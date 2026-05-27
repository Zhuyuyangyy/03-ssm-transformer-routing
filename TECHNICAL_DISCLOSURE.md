# Technical Disclosure: DynaRoute

## Title

**DynaRoute: Dynamic Token-Level Routing Between State Space Model and Transformer Attention Pathways for Efficient Sequence Modeling**

## Inventors

Research Team

## Date of Disclosure

2024 (updated 2025)

---

## 1. Field of the Invention

The present invention relates to neural network architectures for sequence modeling, and more particularly to hybrid architectures that dynamically route individual tokens between a State Space Model (SSM) pathway and a Transformer self-attention pathway within a single layer.

## 2. Background

### 2.1 Prior Art

Sequence modeling architectures have evolved through several generations:

1. **Recurrent Neural Networks (RNNs)**: Process tokens sequentially with O(n) complexity but suffer from vanishing gradients and limited parallelism.

2. **Transformers**: Process all tokens in parallel via self-attention with O(n^2) complexity. Achieved state-of-the-art results across NLP, vision, and multimodal tasks.

3. **State Space Models (SSMs)**: Recent architectures such as Mamba (Gu & Dao, 2023) achieve O(n) complexity through selective state space mechanisms, competitive with Transformers on long sequences.

4. **Hybrid SSM-Transformer models**: Architectures like Jamba (AI21 Labs, 2024) alternate between SSM and Transformer layers at the layer level.

### 2.2 Limitations of Prior Art

- Layer-level alternation (Jamba) forces all tokens in a layer through the same pathway.
- No mechanism exists for token-level routing between SSM and Attention within a single layer.
- Existing mixture-of-experts (MoE) architectures route between identical expert types, not between fundamentally different computational paradigms.

## 3. Summary of the Invention

DynaRoute introduces a **token-level dynamic routing mechanism** that selectively directs each token in a sequence through either:

- **Path A (SSM)**: A selective state space model pathway optimized for efficient long-range dependency capture with O(n) complexity, or
- **Path B (Attention)**: A multi-head self-attention pathway optimized for precise local context modeling.

The routing decision is made by a lightweight learned router network that operates on the input tokens augmented with contextual information.

## 4. Detailed Description

### 4.1 System Architecture

The DynaRoute system comprises the following components:

#### 4.1.1 HybridLayer

A single processing layer containing:

- An **SSMPathway** implementing a selective state space scan
- An **AttentionPathway** implementing multi-head self-attention
- A **TokenRouter** that computes per-token routing weights
- Layer normalization modules for each pathway and the output

The forward computation is:

```
routing_weights, logits = Router(x)
ssm_out = LayerNorm_SSM(SSMPath(x))
attn_out = LayerNorm_Attn(AttentionPath(x))
output = routing_weights[:, :, 0] * ssm_out + routing_weights[:, :, 1] * attn_out
output = LayerNorm_Out(output)
```

#### 4.1.2 SSMPathway

Implements a Mamba-style selective state space model:

1. **Input projection**: Linear projection producing (x_ssm, z) via chunking
2. **Selective parameters**: Input-dependent dt (time step), B (input matrix), C (output matrix)
3. **Discretization**: A_bar = exp(A * dt)
4. **State update**: h_t = A_bar_t * h_{t-1} + dt_t * B_t * x_t
5. **Output**: y_t = C_t * h_t + D * x_t (skip connection)
6. **Gating**: output = y * sigmoid(z)

#### 4.1.3 AttentionPathway

Implements standard multi-head self-attention with the following optimizations:

1. **Fused QKV projection**: Single linear layer producing Q, K, V simultaneously
2. **Flash attention dispatch**: Automatic use of `torch.nn.functional.scaled_dot_product_attention` which dispatches to hardware-optimized kernels (FlashAttention-2, memory-efficient attention) when available
3. **Optional causal masking** for autoregressive generation

#### 4.1.4 TokenRouter

A lightweight MLP router that takes each token's hidden state (optionally augmented with sequence-level context) and produces routing logits:

1. **Context aggregation**: Mean-pool the sequence, project to lower dimension, concatenate with each token
2. **Router MLP**: Two hidden layers with SiLU activation, producing logits for each pathway
3. **Learnable bias**: Added to logits to allow the model to learn an a priori routing preference
4. **Training mode**: Gumbel-Softmax for differentiable soft routing
5. **Inference mode**: Argmax for hard one-hot routing

#### 4.1.5 AdaptiveRouter (Alternative)

An enhanced router with:

1. **Learnable temperature**: log-parameterized temperature that is optimized end-to-end
2. **Capacity-constrained routing**: Limits the maximum number of tokens routed to each path, preventing degenerate collapse where all tokens route to one path

### 4.2 Training Methodology

#### 4.2.1 Three-Phase Progressive Training

| Phase | Epochs | Temperature | Router Status | Purpose |
|---|---|---|---|---|
| Warmup | 0 to W | High (start_temp) | Trainable | Explore diverse routing configurations |
| Annealing | W to T-F | Decaying | Trainable | Gradually commit to routing decisions |
| Fine-tuning | T-F to T | Low (end_temp) | Frozen | Refine pathway parameters with fixed routing |

#### 4.2.2 Temperature Annealing Schedules

Three annealing schedules are provided:

- **Linear**: tau(t) = tau_start + (tau_end - tau_start) * (t / T)
- **Cosine**: tau(t) = tau_end + 0.5 * (tau_start - tau_end) * (1 + cos(pi * t / T))
- **Exponential**: tau(t) = tau_start * exp(log(tau_end/tau_start) * t / T)

#### 4.2.3 Load Balancing Loss

An auxiliary loss prevents routing collapse:

```
L_balance = KL(avg_routing_probs || uniform_distribution)
L_total = L_task + lambda_balance * L_balance
```

where `avg_routing_probs` is the mean routing probability per pathway across all tokens and positions.

### 4.3 Analysis Toolkit

Post-hoc analysis capabilities include:

1. **Per-token-type routing analysis**: How different categories of tokens (content words, function words, punctuation) are routed
2. **Per-position routing analysis**: How routing patterns vary along the sequence dimension (e.g., do first/last tokens prefer attention?)
3. **Routing pattern discovery**: Identifying contiguous blocks of same-pathway routing
4. **Routing stability measurement**: Measuring consistency of routing decisions across stochastic forward passes
5. **Gradient-based token importance**: Attribution of routing decisions to input tokens via gradient backpropagation through the embedding layer

## 5. Claims

### Claim 1 (Independent)

A neural network layer for processing a sequence of tokens, comprising:

- a first pathway implementing a state space model with selective scan mechanism;
- a second pathway implementing multi-head self-attention;
- a routing network that computes, for each token in the sequence, a routing weight indicating the relative contribution of the first pathway and the second pathway to that token's output representation;
- wherein the output for each token is a weighted combination of the outputs of the first pathway and the second pathway, weighted by the routing weights for that token.

### Claim 2

The neural network layer of Claim 1, wherein the routing network:

- aggregates contextual information from the sequence via pooling;
- concatenates the contextual information with each token's representation;
- processes the concatenated representation through a multi-layer perceptron to produce routing logits.

### Claim 3

The neural network layer of Claim 1, wherein:

- during training, routing weights are computed using Gumbel-Softmax for differentiable discrete selection;
- during inference, routing weights are computed using argmax for hard one-hot selection.

### Claim 4

The neural network layer of Claim 1, further comprising a load balancing loss computed as the Kullback-Leibler divergence between the average routing probability distribution and a uniform distribution.

### Claim 5

A training method for the neural network layer of Claim 1, comprising:

- a warmup phase with high routing temperature for soft routing exploration;
- an annealing phase with gradually decreasing routing temperature;
- a fine-tuning phase with low routing temperature and frozen routing parameters.

### Claim 6

The training method of Claim 5, wherein the annealing schedule is selected from the group consisting of linear, cosine, and exponential schedules.

### Claim 7

The neural network layer of Claim 1, wherein the routing network further comprises:

- a learnable temperature parameter;
- a capacity constraint that limits the maximum number of tokens routed to each pathway.

### Claim 8

A system for analyzing routing patterns in the neural network layer of Claim 1, comprising:

- means for computing per-token-type routing statistics;
- means for computing per-position routing statistics;
- means for discovering contiguous routing patterns;
- means for measuring routing stability across multiple forward passes;
- means for computing gradient-based token importance for routing decisions.

## 6. Advantages Over Prior Art

1. **Token-granularity adaptivity**: Unlike layer-level routing, each token is individually assessed for its computational needs.

2. **Paradigm-level routing**: Routes between fundamentally different computational paradigms (SSM vs. Attention) rather than between identical experts.

3. **Interpretability**: Routing patterns reveal the model's implicit assessment of which tokens require attention vs. SSM processing.

4. **Efficiency**: At inference, hard routing eliminates unnecessary attention computation for SSM-routed tokens, achieving near-SSM efficiency with Transformer-quality outputs where needed.

5. **Stability**: Progressive training with temperature annealing and load balancing loss prevents routing collapse.

## 7. Implementations

### 7.1 Software Implementation

The invention is implemented in Python using the PyTorch framework. Key implementation details:

- `TokenRouter`: ~60 lines of PyTorch module code
- `SSMPathway`: ~80 lines implementing selective scan
- `AttentionPathway`: ~40 lines using `F.scaled_dot_product_attention`
- `HybridLayer`: ~50 lines combining pathways with routing
- `ProgressiveTrainer`: ~200 lines implementing three-phase training
- `RoutingAnalyzer`: ~250 lines implementing analysis toolkit

### 7.2 Hardware Considerations

- The SSM pathway uses sequential scanning (O(n) in sequence length) and is suitable for CPU and GPU execution.
- The Attention pathway leverages hardware-optimized kernels (FlashAttention) when available on CUDA devices.
- The router is a lightweight MLP with minimal computational overhead compared to the pathways.

## 8. Definitions

| Term | Definition |
|---|---|
| SSM | State Space Model, a class of sequence models with O(n) complexity |
| Token | A discrete unit in a sequence (e.g., a word, subword, or patch) |
| Routing weight | A scalar indicating the relative contribution of a pathway to a token's output |
| Gumbel-Softmax | A differentiable approximation to discrete categorical sampling |
| Capacity factor | A multiplier limiting the maximum tokens routed to each pathway |
| Temperature | A parameter controlling the sharpness of routing probability distributions |
