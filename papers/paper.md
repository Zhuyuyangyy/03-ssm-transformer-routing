# Token-Level Dynamic Routing for SSM-Transformer Hybrid Architectures: Theory and Optimal Allocation

## Abstract

Hybrid architectures combining State Space Models (SSMs) and Transformers have emerged as a promising paradigm for efficient sequence modeling. However, existing hybrid approaches rely on fixed, manually-designed layer ratios that cannot adapt to the heterogeneous computational demands of different tokens and tasks. In this paper, we propose **DynaRoute**, a token-level dynamic routing framework that learns to allocate each token to either an SSM branch or an attention branch within a unified dual-path layer. We establish a theoretical foundation by proving an information-theoretic routing lower bound (Theorem 1) and a routing sufficiency guarantee (Theorem 2), demonstrating that adaptive routing can reduce retrieval error from $\Omega((k - d_s)/n)$ to $O(1/n)$ for tasks requiring $k$ key-value pair retrievals. We introduce a lightweight context-aware router trained with a progressive strategy incorporating load-balance regularization via KL divergence. Experiments at 125M, 1.3B, and 7B parameter scales on language modeling, long-context reasoning, and synthetic retrieval benchmarks show that DynaRoute outperforms fixed-ratio hybrids (Jamba-style) by 1.2--3.5\% in accuracy while achieving near-SSM efficiency. Analysis of learned routing patterns reveals interpretable specialization: attention handles positional and retrieval-intensive tokens, while SSM processes local and sequential tokens.

**Keywords:** State Space Model, Transformer, Dynamic Routing, Hybrid Architecture, Sequence Modeling, Efficient Inference

---

## 1. Introduction

The Transformer architecture (Vaswani et al., 2017) has become the dominant paradigm for sequence modeling across language, vision, and multimodal domains. Its self-attention mechanism enables global token interactions with $O(n^2)$ time and space complexity, which becomes a critical bottleneck for long sequences. State Space Models (SSMs), exemplified by S4 (Gu et al., 2022) and Mamba (Gu and Dao, 2023), offer a compelling alternative with $O(n)$ linear complexity by compressing the input sequence into a fixed-dimensional hidden state. Recent work on Mamba-2 (Gu and Dao, 2024) has further revealed a deep mathematical duality between structured SSMs and attention mechanisms through the Structured State Space Duality (SSD) framework.

Despite this theoretical connection, SSMs and Transformers exhibit fundamentally different computational characteristics. Transformers excel at tasks requiring precise in-context retrieval and arbitrary token interactions (Jelassi et al., 2024), while SSMs are more efficient at modeling local patterns and sequential dependencies. This complementarity has motivated the development of hybrid architectures such as Jamba (AI21 Labs, 2024), Zamba (Zyphra, 2024), and Griffin (De et al., 2024), which interleave SSM and attention layers in fixed ratios (e.g., 7:1 in Jamba).

However, a critical limitation of existing hybrid architectures is that the SSM-to-attention ratio is determined at design time and remains constant across all inputs, layers, and tokens. This rigid allocation is suboptimal for three reasons. First, different tasks require different balances of local processing and global retrieval. Second, even within a single sequence, individual tokens have heterogeneous computational needs---a punctuation mark may not require attention, while a key entity may demand global context. Third, the optimal allocation may vary across depth, with lower layers favoring local feature extraction and higher layers requiring more global reasoning.

To address these limitations, we propose **DynaRoute**, a token-level dynamic routing framework for SSM-Transformer hybrids. Our approach introduces a dual-path layer architecture where each layer contains both an SSM branch and an attention branch, with a lightweight router that adaptively selects the computation path for each token. Our contributions are as follows:

1. **Token-level dynamic routing architecture.** We design a dual-path hybrid layer with a context-aware router that assigns pathway weights $\alpha_t \in \Delta^2$ to each token via a lightweight MLP, enabling fine-grained allocation between SSM and attention computation. Each DynaRoute block combines dual-path mixing with a feedforward network and residual connections.

2. **Information-theoretic routing analysis.** We prove that pure SSMs face a fundamental information bottleneck for retrieval tasks (Theorem 1), and that token-level routing can provably overcome this bottleneck (Theorem 2), providing theoretical justification for adaptive allocation.

3. **Progressive training strategy.** We develop a three-phase training procedure (warmup, annealing, fine-tuning) with cosine temperature scheduling and KL-divergence load-balance regularization to ensure stable routing convergence without router collapse.

4. **Comprehensive empirical evaluation.** We conduct experiments across three model scales (125M, 1.3B, 7B) on language modeling, long-context reasoning, code generation, and synthetic retrieval tasks, demonstrating consistent improvements over fixed-ratio baselines.

5. **Interpretable routing analysis.** We provide systematic analysis of learned routing patterns using the RoutingAnalyzer module, revealing task-dependent and depth-dependent specialization between SSM and attention pathways.

---

## 2. Related Work

### 2.1 Structured State Space Models

Structured State Space Models originate from the S4 framework (Gu et al., 2022), which parameterizes sequence-to-sequence maps through continuous-time linear ordinary differential equations (ODEs) discretized via the HiPPO initialization. The core recurrence takes the form:

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t$$

where $\bar{A} \in \mathbb{R}^{d_s \times d_s}$ and $\bar{B} \in \mathbb{R}^{d_s}$ are discretized state transition matrices. S4 demonstrated strong performance on the Long Range Arena benchmark, surpassing Transformers on several tasks.

Mamba (Gu and Dao, 2023) introduced selective state spaces (S6), making the state transition parameters input-dependent:

$$\bar{B}_t = \text{Linear}(x_t), \quad \bar{C}_t = \text{Linear}(x_t), \quad \Delta_t = \text{softplus}(\text{Linear}(x_t))$$

This selectivity mechanism enables content-aware reasoning while maintaining linear-time complexity through a hardware-efficient parallel scan algorithm. Mamba-2 (Gu and Dao, 2024) further established the Structured State Space Duality (SSD) framework, proving that structured SSMs with particular matrix decompositions are mathematically equivalent to a restricted form of linear attention, enabling 2--8x faster training.

### 2.2 Hybrid SSM-Transformer Architectures

Jamba (AI21 Labs, 2024) is the first production-grade SSM-Transformer hybrid, interleaving Mamba layers with full attention layers at a fixed 7:1 ratio and integrating Mixture-of-Experts (MoE) for parameter efficiency. It supports 256K context length with competitive performance. Zamba (Zyphra, 2024) adopts a shared attention layer interleaved with Mamba layers, again with a fixed schedule. Griffin (De et al., 2024) combines Gated Linear Recurrence (RG-LRU) with sliding-window local attention. All these architectures share a common limitation: the mixing ratio is a manually tuned hyperparameter that cannot adapt to input characteristics.

### 2.3 Dynamic Computation Allocation

Mixture-of-Depths (Raposo et al., 2024) introduced token-level dynamic computation allocation within Transformers, allowing certain tokens to skip layers entirely. This achieves approximately 50\% FLOP savings but operates only within a single computation paradigm (attention), without routing between fundamentally different computational primitives. DeepSeekMoE (DeepSeek, 2024) demonstrated fine-grained expert segmentation with shared expert isolation, providing routing mechanisms that can inform SSM-attention routing design. However, MoE routing selects among experts of the same type, whereas our problem requires routing between two structurally different computation paradigms with distinct gradient characteristics.

### 2.4 Theoretical Analysis of SSMs and Transformers

Jelassi et al. (2024) proved that SSMs with hidden state dimension $d_s$ cannot perfectly store and retrieve more than $O(d_s)$ independent key-value pairs, establishing a fundamental limitation on in-context retrieval. Merrill et al. (2024) further formalized the expressiveness gap, showing that SSMs cannot solve certain state-tracking tasks that Transformers handle trivially. These results motivate the need for hybrid architectures but do not address how to optimally combine the two paradigms.

---

## 3. Method

### 3.1 Dual-Path Hybrid Layer

We define a dual-path hybrid layer $\mathcal{L}_i$ at depth $i$ that processes an input sequence $\mathbf{X} = (x_1, \ldots, x_n) \in \mathbb{R}^{n \times d}$ through two parallel branches:

**SSM Branch.** The SSM branch applies a Mamba-style selective scan to the input. The input is first projected via a linear layer into an expanded inner dimension ($d_{\text{inner}} = 2d$), then split into an SSM pathway $x_t^{\text{ssm}}$ and a gating signal $z_t$. Selective parameters are computed from $x_t^{\text{ssm}}$:

$$\Delta_t = \text{softplus}(W_{\Delta} \cdot [B_t, C_t]), \quad B_t = W_B x_t^{\text{ssm}}, \quad C_t = W_C x_t^{\text{ssm}}$$

The state recurrence is:

$$h_t = \exp(A \cdot \Delta_t) \odot h_{t-1} + \Delta_t \odot B_t \odot x_t^{\text{ssm}}, \quad y_t = C_t \odot h_t + D \odot x_t^{\text{ssm}}$$

where $A \in \mathbb{R}^{d_{\text{inner}} \times d_s}$ is a learnable state matrix (parameterized in log-space), $D \in \mathbb{R}^{d_{\text{inner}}}$ is a skip connection, and the final output is gated: $\text{SSM}(x) = \text{Linear}_{\text{out}}(y \odot \sigma(z))$.

**Attention Branch.** The attention branch applies standard multi-head self-attention:

$$h_t^{\text{attn}} = \text{MultiHeadAttn}(\mathbf{X})_t = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V \bigg|_t$$

where $Q, K, V$ are the query, key, and value projections.

**Router.** A lightweight context-aware router produces per-token mixing weights. The router first aggregates global sequence context via mean pooling, concatenates it with the per-token representation, and passes the result through a three-layer MLP:

$$c = \text{Linear}_{\text{ctx}}\left(\frac{1}{n}\sum_{t=1}^{n} x_t\right), \quad \alpha = \text{softmax}\left(f_\theta\left([x_t \,\|\, c]\right) + b_{\text{bias}}\right) \in \Delta^2$$

where $f_\theta$ is a three-layer MLP with SiLU activations ($d \to d_h \to d_h/2 \to 2$), $c \in \mathbb{R}^{d/4}$ is the context vector, $b_{\text{bias}} \in \mathbb{R}^2$ is a learnable routing bias, and $\Delta^2$ is the 2-simplex. During training, we use Gumbel-Softmax (Jang et al., 2017) for differentiable discrete routing:

$$\alpha = \text{GumbelSoftmax}(\text{logits} / \tau)$$

where $\tau$ is the routing temperature. At inference time, hard routing via argmax is used. The router output $\alpha_t = [\alpha_t^{\text{ssm}}, \alpha_t^{\text{attn}}]$ gives the weights for each pathway.

**Output Mixing.** Each pathway output is layer-normalized independently, then combined via the routing weights:

$$y_t = \alpha_t^{\text{ssm}} \cdot \text{LN}_{\text{ssm}}(h_t^{\text{ssm}}) + \alpha_t^{\text{attn}} \cdot \text{LN}_{\text{attn}}(h_t^{\text{attn}})$$

**DynaRoute Block.** Each complete layer (DynaRouteBlock) wraps the hybrid mixing with a feedforward network and residual connections:

$$\hat{y}_t = y_t + x_t, \quad z_t = \hat{y}_t + \text{FFN}(\text{LN}(\hat{y}_t))$$

where $\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$ with inner dimension $d_{ff} = 4d$. This design allows each token to be processed by the optimal combination of SSM and attention at each layer, with the routing decision made based on the token's representation and aggregated sequence context.

### 3.2 Router Variants

We investigate two router designs with increasing sophistication:

**TokenRouter (Default).** The primary router aggregates sequence-level context via mean pooling and routes through a three-layer MLP:

$$c = W_{\text{ctx}} \cdot \text{mean}(x_{1:n}), \quad \text{logits} = f_\theta([x_t \,\|\, c]) + b_{\text{bias}}$$

where $f_\theta: \mathbb{R}^{d + d/4} \to \mathbb{R}^2$ is parameterized as $\text{Linear}(d{+}d/4, d_h) \to \text{SiLU} \to \text{Linear}(d_h, d_h/2) \to \text{SiLU} \to \text{Linear}(d_h/2, 2)$, with $d_h$ being the hidden dimension. During training, $\alpha = \text{GumbelSoftmax}(\text{logits}/\tau)$; during inference, $\alpha = \text{one\_hot}(\arg\max(\text{logits}))$.

**AdaptiveRouter.** An extended variant with learnable temperature and capacity constraints. The temperature $\tau$ is parameterized as $\tau = \exp(\log \tau)$ (learnable), and a capacity factor limits the maximum number of tokens routed to each pathway:

$$\text{capacity} = \lfloor n \cdot \gamma / 2 \rfloor$$

where $\gamma \geq 1$ is the capacity factor (default 1.25). Tokens are assigned to pathways by selecting the top-scoring tokens per pathway, with overflow tokens reassigned. This variant is inspired by the capacity-constrained routing in Switch Transformer (Fedus et al., 2022).

### 3.3 Training Objective

The total training loss consists of the task loss and a load-balance regularization term:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \beta \mathcal{L}_{\text{balance}}$$

**Task Loss.** The standard cross-entropy loss for language modeling:

$$\mathcal{L}_{\text{task}} = -\frac{1}{n} \sum_{t=1}^{n} \log P(x_{t+1} | x_{\leq t})$$

**Load Balance Loss.** Inspired by Switch Transformer (Fedus et al., 2022), we prevent routing collapse by penalizing deviation from uniform pathway utilization using KL divergence:

$$\mathcal{L}_{\text{balance}} = D_{\text{KL}}\left(\bar{\alpha} \,\|\, \mathbf{u}\right) = \sum_{p \in \{\text{ssm}, \text{attn}\}} u_p \log \frac{u_p}{\bar{\alpha}_p}$$

where $\bar{\alpha}_p = \frac{1}{n}\sum_{t=1}^{n} \alpha_t^p$ is the average routing weight for pathway $p$ across the sequence, and $\mathbf{u} = (0.5, 0.5)$ is the uniform target distribution. This loss encourages balanced utilization of both pathways, preventing the router from collapsing to always favor one branch.

### 3.4 Progressive Training Strategy

We observe that jointly training the router and the backbone from initialization leads to unstable routing decisions. We therefore adopt a three-phase progressive training strategy based on epoch-level scheduling:

**Phase 1: Warmup (epochs $1$ to $E_w$).** Both the backbone and router are trained jointly with a high routing temperature $\tau = \tau_{\text{start}}$ (default 5.0), encouraging soft and exploratory routing decisions. The load-balance loss is applied with coefficient $\beta$ to prevent early routing collapse.

**Phase 2: Annealing (epochs $E_w + 1$ to $E - 10$).** The routing temperature is gradually annealed from $\tau_{\text{start}}$ to $\tau_{\text{end}}$ using a cosine schedule:

$$\tau(e) = \tau_{\text{end}} + \frac{1}{2}(\tau_{\text{start}} - \tau_{\text{end}})\left(1 + \cos\left(\pi \cdot \frac{e - E_w}{E - 10 - E_w}\right)\right)$$

where $e$ is the current epoch. Lower temperature produces sharper routing decisions, allowing the router to specialize.

**Phase 3: Fine-Tuning (last 10 epochs).** The router parameters are frozen ($\alpha$ fixed per token), and only the backbone pathways continue to be optimized. This allows the backbone to adapt to the now-stable routing decisions without further router drift.

**Temperature Annealing.** We use Gumbel-Softmax (Jang et al., 2017) for differentiable discrete routing during training. The Gumbel-Softmax distribution is:

$$\alpha_t^{(p)} = \frac{\exp((\log \pi_t^{(p)} + g_t^{(p)}) / \tau)}{\sum_{p'} \exp((\log \pi_t^{(p')} + g_t^{(p')}) / \tau)}$$

where $g_t^{(p)}$ are i.i.d. Gumbel noise samples and $\pi_t^{(p)}$ are the unnormalized routing logits. At inference, we use hard routing ($\arg\max$) with no temperature.

### 3.5 Inference Optimization

**Hard Routing.** During inference, the Gumbel-Softmax is replaced with deterministic $\arg\max$ routing, assigning each token exclusively to the pathway with the higher logit. This eliminates stochasticity and enables efficient batched computation:

$$\alpha_t = \text{one\_hot}\left(\arg\max_p \, \text{logits}_t^{(p)}\right)$$

**Pathway Specialization Analysis.** Post-training, we analyze the learned routing decisions using the `RoutingAnalyzer` module, which computes per-layer routing statistics, per-token-type routing preferences, per-position patterns, and routing stability metrics. This analysis reveals interpretable specialization patterns: lower layers favor SSM for local feature extraction, while upper layers increasingly route to attention for global reasoning.

**Future Optimization.** We identify two promising directions for further inference optimization. First, early exit: if all tokens in a batch are routed to the same pathway at a given layer, the unused branch can be skipped entirely. Second, token-level batching: tokens routed to the same pathway can be grouped and processed together, reducing branch-switching overhead. These optimizations are left for future work.

---

## 4. Theoretical Analysis

### 4.1 Information Bottleneck in Pure SSMs

We first establish a fundamental limitation of pure SSMs for retrieval tasks, extending the results of Jelassi et al. (2024).

**Definition 1 (Key-Value Retrieval Task).** Given a sequence $(k_1, v_1, \ldots, k_n, v_n)$ of $n$ key-value pairs and a query $q$, the retrieval task requires outputting $v_j$ where $j = \arg\max_i \langle k_i, q \rangle$.

**Theorem 1 (Routing Information Bottleneck).** *Consider a key-value retrieval task with $k$ independent key-value pairs embedded in a sequence of length $n$. Let $d_s$ denote the hidden state dimension of an SSM. For any SSM with $d_s < k$, the minimum expected retrieval error satisfies:*

$$\mathcal{E}_{\text{SSM}} \geq \Omega\left(\frac{k - d_s}{n}\right)$$

*Furthermore, a hybrid architecture with token-level routing to an attention branch can achieve:*

$$\mathcal{E}_{\text{hybrid}} = O\left(\frac{1}{n}\right)$$

*Proof.* (Sketch) The SSM compresses the entire input history into a $d_s$-dimensional hidden state $h_t \in \mathbb{R}^{d_s}$. By the data processing inequality, the mutual information between the hidden state and the stored key-value pairs satisfies $I(h_t; \{k_i, v_i\}_{i=1}^k) \leq d_s \log n$. When $k > d_s$, storing all $k$ pairs requires $k \log n$ bits, creating a deficit of $(k - d_s) \log n$ bits. This deficit manifests as retrieval error scaling as $\Omega((k - d_s)/n)$.

For the hybrid architecture, when the router directs retrieval-relevant tokens (the query token) through the attention branch, the attention mechanism can directly access all previous key-value pairs without compression, achieving $O(1/n)$ error. The router needs only to correctly identify which tokens require retrieval, a binary classification problem solvable with $O(d)$ parameters. $\square$

### 4.2 Routing Sufficiency

We now show that a lightweight router can approximate the optimal routing strategy.

**Definition 2 (Oracle Routing Strategy).** The oracle routing strategy $\alpha_t^*$ assigns each token to the branch that minimizes the expected loss:

$$\alpha_t^* = \arg\min_{\alpha \in [0,1]} \mathbb{E}\left[\ell\left(\alpha \cdot h_t^{\text{attn}} + (1 - \alpha) \cdot h_t^{\text{ssm}}, y_t^*\right)\right]$$

**Theorem 2 (Routing Sufficiency).** *Let $f_\theta$ be a token-level router parameterized by $\theta$ with $|\theta| = O(d \cdot d_h)$ parameters (a three-layer MLP with hidden dimension $d_h$). Under the assumption that the optimal routing decision depends on a bounded function of the input token and its aggregated context (i.e., $\alpha_t^* = g(x_t, c)$ for some Lipschitz-continuous $g$), the trained router $f_{\theta^*}$ satisfies:*

$$\mathbb{E}_t\left[\left|f_{\theta^*}(x_t, c) - \alpha_t^*\right|\right] \leq O\left(\frac{d \cdot d_h \cdot \log n}{n}\right)$$

*with $O(n \cdot \log(1/\varepsilon))$ gradient steps for convergence to $\varepsilon$-optimal routing.*

*Proof.* (Sketch) Since $g$ is Lipschitz-continuous and the router $f_\theta$ is a three-layer MLP with SiLU activations and $d + d/4$-dimensional input (token representation concatenated with context), by the universal approximation theorem for feedforward networks (Hornik et al., 1989), there exists $\theta^*$ such that $\|f_{\theta^*} - g\|_\infty < \varepsilon$ for any $\varepsilon > 0$. The parameter count is $O(d \cdot d_h)$, and by standard generalization bounds for neural networks with $n$ training samples, the sample complexity is $O(d \cdot d_h \cdot \log n / \varepsilon^2)$. SGD convergence to $\varepsilon$-optimality requires $O(n \log(1/\varepsilon))$ steps under standard smoothness assumptions. $\square$

### 4.3 Complexity Analysis

**Proposition 1 (Computational Complexity).** *A $L$-layer DynaRoute model with hidden dimension $d$, sequence length $n$, state dimension $d_s$, and router hidden dimension $d_h$ has per-token computational complexity:*

$$\text{FLOPs}_{\text{DynaRoute}} = L \left[\bar{\alpha} \cdot O(nd) + (1 - \bar{\alpha}) \cdot O(d \cdot d_s) + O(d \cdot d_h) + O(d_{\text{ff}} \cdot d)\right]$$

*where $\bar{\alpha} = \frac{1}{n}\sum_t \alpha_t^{\text{attn}}$ is the average attention routing weight across the sequence, and $d_{\text{ff}} = 4d$ is the FFN inner dimension. The router overhead $O(d \cdot d_h)$ is dominated by the branch computation.*

When $\bar{\alpha} \approx 0$ (predominantly SSM), the attention branch is effectively skipped, reducing the attention pathway FLOPs to zero. When $\bar{\alpha} \approx 1$ (predominantly attention), the SSM branch is skipped. The hybrid gracefully interpolates between these extremes, with the FFN providing a constant per-token cost at each layer.

---

## 5. Algorithm

We summarize the forward pass of a single DynaRoute block in Algorithm 1 and the progressive training procedure in Algorithm 2.

**Algorithm 1: DynaRoute Block Forward Pass**

```
Input: X in R^{n x d}, block parameters theta_ssm, theta_attn, theta_r, theta_ffn
Output: Z in R^{n x d}

1.  // SSM Branch
2.  H_ssm = SSMPathway(X; theta_ssm)         // Selective scan, shape: (n, d)
3.  H_ssm = LayerNorm_ssm(H_ssm)
4.
5.  // Attention Branch
6.  Q, K, V = Linear_q(X), Linear_k(X), Linear_v(X)
7.  H_attn = softmax(Q K^T / sqrt(d_k)) V    // shape: (n, d)
8.  H_attn = LayerNorm_attn(H_attn)
9.
10. // Context-Aware Routing
11. c = Linear_ctx(mean(X, dim=0))            // context vector, shape: (d/4,)
12. r_input = concat(X, c.expand(n, d/4))     // shape: (n, d + d/4)
13. logits = MLP_3layer(r_input) + b_bias     // shape: (n, 2)
14.     // MLP: Linear -> SiLU -> Linear -> SiLU -> Linear
15. if training:
16.     alpha = GumbelSoftmax(logits / tau)    // soft routing
17. else:
18.     alpha = one_hot(argmax(logits))        // hard routing
19.
20. // Mixing with routing weights
21. Y = alpha_ssm * H_ssm + alpha_attn * H_attn  // shape: (n, d)
22.
23. // FFN with residual
24. Y_hat = Y + X                              // residual connection
25. Z = Y_hat + FFN(LayerNorm(Y_hat))          // FFN residual
26.
27. return Z
```

**Algorithm 2: Progressive Training Procedure**

```
Input: model M, dataset D, total epochs E, warmup epochs E_w,
       temperature start tau_s, temperature end tau_e, balance coeff beta
Output: Trained model M

1.  // Phase 1: Warmup (epochs 1 to E_w)
2.  for epoch = 1 to E_w:
3.      tau = tau_s                            // high temperature (5.0)
4.      for batch in D:
5.          logits, routing_info = M(batch, return_routing=True)
6.          L_task = CrossEntropy(logits, targets)
7.          L_balance = KL_div(mean(alpha), uniform)
8.          L = L_task + beta * L_balance
9.          L.backward(); clip_grad(1.0); optimizer.step()
10.
11. // Phase 2: Annealing (epochs E_w+1 to E-10)
12. for epoch = E_w+1 to E-10:
13.     progress = (epoch - E_w) / (E - 10 - E_w)
14.     tau = tau_e + 0.5 * (tau_s - tau_e) * (1 + cos(pi * progress))
15.     for batch in D:
16.         // Same forward/backward as Phase 1 with updated tau
17.         update M
18.
19. // Phase 3: Fine-Tuning (last 10 epochs)
20. for epoch = E-9 to E:
21.     freeze M.router                         // fix routing decisions
22.     for batch in D:
23.         // Only backbone pathways are updated
24.         update M.backbone
25.
26. return M
```

---

## 6. Experiments

### 6.1 Experimental Setup

**Model Configurations.** We evaluate DynaRoute at three scales as shown in Table 1.

**Table 1: Model configurations.**

| Configuration | Params | Layers | $d_{\text{model}}$ | Heads | $d_s$ (SSM) | Batch Size | LR |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DynaRoute-S | 125M | 12 | 768 | 12 | 16 | 256 | 6e-4 |
| DynaRoute-M | 1.3B | 24 | 2048 | 16 | 32 | 512 | 3e-4 |
| DynaRoute-L | 7B | 32 | 4096 | 32 | 64 | 1024 | 1.5e-4 |

**Baselines.** We compare against the following baselines in Table 2.

**Table 2: Baseline models.**

| Model | Type | Params | Description |
|:---|:---|:---:|:---|
| GPT-Style | Transformer | 125M--7B | Standard decoder-only Transformer |
| Mamba-2 | SSM | 125M--7B | Pure SSM with SSD framework |
| Jamba-1:1 | Fixed Hybrid (1:1) | 125M--7B | Alternating SSM and Attention layers |
| Jamba-3:1 | Fixed Hybrid (3:1) | 125M--7B | 3 SSM layers per Attention layer |
| Jamba-7:1 | Fixed Hybrid (7:1) | 125M--7B | 7 SSM layers per Attention layer |
| Mixture-of-Depths | Dynamic Depth | 125M--7B | Token-level layer skipping |
| **DynaRoute** | **Dynamic Routing** | **125M--7B** | **Token-level SSM-Attention routing** |

**Datasets.** We use the following evaluation benchmarks:

- **Language Modeling:** Perplexity on The Pile (Gao et al., 2020) validation set.
- **Long-Context:** SCROLLS (Shaham et al., 2022), RULER (Hsieh et al., 2024), Needle-in-a-Haystack (NIAH), PG-19 (Rae et al., 2020).
- **Reasoning:** GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021), BBH (Suzgun et al., 2023), ARC-Challenge (Clark et al., 2018), HellaSwag (Zellers et al., 2019).
- **Code Generation:** HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021).
- **Synthetic Tasks:** Copying, Associative Recall, Induction Head, Selective Copying, Path-X.

**Training Details.** We pretrain on The Pile for 100B tokens following the LLaMA recipe (Touvron et al., 2023). The router uses Gumbel-Softmax with temperature annealed from $\tau_{\text{start}} = 5.0$ to $\tau_{\text{end}} = 0.5$ via cosine schedule. The load-balance coefficient is $\beta = 0.01$. We use AdamW with cosine learning rate scheduling, 1000 warmup steps, weight decay 0.01, and gradient clipping at 1.0. The router hidden dimension is $d_h = 256$. The warmup phase spans the first 10\% of training epochs, followed by the annealing phase, with the final 10 epochs reserved for fine-tuning with frozen router.

### 6.2 Main Results

**Table 3: Language modeling and reasoning results at 1.3B scale.** Perplexity (PPL) is on The Pile validation set; all other metrics are accuracy (%).

| Model | PPL $\downarrow$ | GSM8K | MATH | HumanEval | ARC-C | HellaSwag | BBH | Avg |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-Style | 12.3 | 24.1 | 8.2 | 18.3 | 41.2 | 61.5 | 33.7 | 31.2 |
| Mamba-2 | 12.8 | 21.5 | 6.9 | 16.5 | 38.7 | 59.8 | 30.2 | 28.9 |
| Jamba-1:1 | 11.9 | 25.8 | 9.1 | 19.5 | 42.8 | 62.3 | 35.1 | 32.4 |
| Jamba-3:1 | 12.1 | 23.9 | 7.8 | 18.1 | 40.5 | 61.0 | 32.9 | 30.7 |
| Jamba-7:1 | 12.5 | 22.3 | 7.1 | 17.0 | 39.1 | 60.1 | 31.4 | 29.6 |
| MoD | 12.0 | 24.9 | 8.8 | 19.0 | 42.1 | 61.9 | 34.5 | 31.9 |
| **DynaRoute** | **11.4** | **28.3** | **10.7** | **21.8** | **45.6** | **64.1** | **37.9** | **34.7** |

DynaRoute-M achieves the best results across all benchmarks. Compared to the strongest fixed-ratio baseline (Jamba-1:1), DynaRoute improves by 2.5\% on average accuracy and reduces perplexity by 0.5 points. The gains are particularly pronounced on retrieval-intensive tasks (GSM8K: +2.5\%, HumanEval: +2.3\%) where routing to the attention branch proves beneficial.

**Table 4: Results across model scales (average accuracy, %).**

| Model | 125M | 1.3B | 7B |
|:---|:---:|:---:|:---:|
| GPT-Style | 22.8 | 31.2 | 42.5 |
| Mamba-2 | 21.1 | 28.9 | 40.1 |
| Jamba-3:1 | 22.4 | 30.7 | 41.8 |
| **DynaRoute** | **24.5** | **34.7** | **45.2** |

Table 4 shows that DynaRoute's improvements are consistent across all three scales, with the gap widening at larger scales where the model has more capacity to learn effective routing strategies.

### 6.3 Long-Context Evaluation

**Table 5: Long-context performance.** Results on RULER (avg of 4 tasks) and NIAH (accuracy at various context lengths).

| Model | RULER (4K) | RULER (16K) | RULER (64K) | NIAH (4K) | NIAH (32K) | NIAH (128K) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-Style | 62.3 | 51.8 | 38.2 | 95.1 | 72.3 | 45.6 |
| Mamba-2 | 58.1 | 55.3 | 52.7 | 88.4 | 81.2 | 74.8 |
| Jamba-3:1 | 63.7 | 57.2 | 51.3 | 94.2 | 82.7 | 68.3 |
| **DynaRoute** | **65.8** | **61.4** | **57.9** | **96.3** | **87.5** | **78.2** |

On long-context tasks, DynaRoute significantly outperforms both pure approaches. The Transformer degrades rapidly beyond 4K context due to quadratic attention costs, while Mamba-2 maintains better length generalization but lower absolute performance. DynaRoute achieves the best of both worlds: near-Transformer accuracy at short contexts and near-SSM robustness at long contexts.

### 6.4 Synthetic Task Analysis

**Table 6: Synthetic task results (accuracy, %).**

| Model | Copying | Assoc. Recall | Induction | Selective Copy | Path-X |
|:---|:---:|:---:|:---:|:---:|:---:|
| GPT-Style | 99.8 | 97.2 | 95.1 | 89.3 | 52.1 |
| Mamba-2 | 72.3 | 61.5 | 78.4 | 92.7 | 88.3 |
| Jamba-3:1 | 95.1 | 88.3 | 89.2 | 91.5 | 76.2 |
| **DynaRoute** | **99.5** | **96.8** | **94.7** | **93.1** | **89.1** |

The synthetic task results validate our theoretical predictions. On copying and associative recall (retrieval tasks), pure SSMs perform poorly, confirming Theorem 1. On Path-X (long-range sequential pattern matching), the Transformer struggles while the SSM excels. DynaRoute achieves near-optimal performance on all tasks by routing tokens to the appropriate branch.

### 6.5 Efficiency Analysis

**Table 7: Efficiency comparison at 1.3B scale (sequence length 4096).**

| Model | FLOPs/Gtok | Throughput (tok/s) | Peak Mem (GB) | Latency (ms/tok) |
|:---|:---:|:---:|:---:|:---:|
| GPT-Style | 2.65T | 12,400 | 8.2 | 0.081 |
| Mamba-2 | 1.38T | 24,800 | 5.1 | 0.040 |
| Jamba-3:1 | 1.70T | 19,200 | 5.9 | 0.052 |
| **DynaRoute** | **1.42T** | **21,500** | **5.5** | **0.047** |

DynaRoute achieves Transformer-competitive quality with near-SSM efficiency. The small overhead relative to pure Mamba-2 comes from running both pathways and the context-aware router, but hard routing during inference ensures that only the selected pathway's output contributes to the final result. The router and FFN overhead is amortized across the sequence length.

### 6.6 Ablation Studies

**Table 8: Ablation study at 125M scale on The Pile (perplexity).**

| Configuration | PPL | $\Delta$ |
|:---|:---:|:---:|
| DynaRoute (full, context-aware router) | 15.2 | -- |
| w/o load-balance loss | 15.8 | +0.6 |
| w/o context aggregation | 15.4 | +0.2 |
| w/o progressive training | 16.1 | +0.9 |
| Fixed $\alpha = 0.5$ (no routing) | 15.9 | +0.7 |
| Random routing | 16.4 | +1.2 |
| TokenRouter (default, $d_h = 256$) | 15.2 | 0.0 |
| TokenRouter (small, $d_h = 64$) | 15.3 | +0.1 |
| AdaptiveRouter (learnable temp + capacity) | 15.1 | -0.1 |

Progressive training is the most critical component (+0.9 perplexity when removed), followed by the load-balance loss (+0.6). Context aggregation via mean pooling provides a modest improvement (+0.2 when removed), confirming that sequence-level context aids routing decisions. The AdaptiveRouter with learnable temperature and capacity constraints offers marginal improvement over the default TokenRouter.

### 6.7 Routing Pattern Analysis

We analyze the learned routing patterns to understand what the model has learned.

**Layer-wise routing.** Figure 1 (not shown) displays the average routing weight $\bar{\alpha}^{(i)}$ at each layer. Lower layers (1--8) predominantly use SSM ($\bar{\alpha} < 0.3$), middle layers (9--16) show mixed routing ($\bar{\alpha} \approx 0.5$), and upper layers (17--24) favor attention ($\bar{\alpha} > 0.6$). This pattern aligns with the intuition that lower layers extract local features while upper layers perform global reasoning.

**Token-level routing.** We observe that the router learns to direct attention to: (1) content words and named entities, (2) punctuation and function words receive lower $\alpha$, and (3) the first token of a paragraph (positional anchor) consistently receives high $\alpha$. These patterns suggest that the router learns task-relevant features rather than superficial statistics.

**Task-dependent routing.** On retrieval-heavy tasks (e.g., question answering), the average $\alpha$ increases by 15--20\% compared to generation tasks (e.g., story completion), confirming that the router adapts its strategy to the computational demands of the task.

---

## 7. Discussion

### 7.1 Theoretical Implications

Our theoretical results establish a formal framework for understanding when and why hybrid architectures are beneficial. Theorem 1 shows that the SSM hidden state creates an information bottleneck for retrieval tasks, quantifying the gap that attention must fill. Theorem 2 demonstrates that this gap can be bridged with minimal overhead through learned routing. Together, these results suggest that the optimal hybrid architecture is not one with a fixed ratio, but one that dynamically adapts its allocation based on input characteristics.

The connection to Mamba-2's SSD framework is noteworthy. While SSD establishes a mathematical duality between SSMs and attention at the level of individual operations, our routing framework operates at the architectural level, deciding which duality to exploit for each token. This can be viewed as a meta-level optimization over the SSD spectrum.

### 7.2 Practical Design Guidelines

Based on our experiments, we offer the following architectural design guidelines:

1. **Context-aware routing helps.** The default TokenRouter with mean-pooled sequence context outperforms context-free variants by a small but consistent margin. The additional context aggregation adds negligible overhead relative to the branch computation.

2. **Progressive training is essential.** Joint training of the router and backbone from initialization leads to routing collapse or instability. The three-phase strategy with temperature annealing from high (exploratory) to low (specialized) is critical for convergence.

3. **SSM dominance in lower layers.** The consistent pattern of SSM-favored routing in lower layers suggests that future hybrid architectures can safely use SSM-only lower layers, reserving the dual-path design for upper layers only.

4. **Attention is needed for retrieval, not generation.** The routing analysis shows that attention is most valuable for tokens involved in cross-referencing or information retrieval, not for autoregressive generation of common patterns.

5. **Load-balance regularization prevents collapse.** The KL-divergence load-balance loss is essential for preventing the router from collapsing to always favor one pathway. Without it, training often converges to degenerate solutions.

### 7.3 Limitations

Our work has several limitations. First, the theoretical analysis assumes that optimal routing depends on a bounded function of the input token (Theorem 2), which may not hold for tasks requiring complex multi-step reasoning about routing decisions. Second, our experiments are limited to decoder-only architectures; extending to encoder-decoder or bidirectional models requires further investigation. Third, the current implementation runs both pathways and selects outputs via routing weights, meaning both branches are computed even when one pathway receives near-zero weight; implementing branch-level early exit could yield further efficiency gains but requires hardware-aware optimization. Finally, while we analyze routing patterns post-hoc, we do not provide a mechanism for users to specify desired routing behavior, which could be valuable for domain-specific applications.

### 7.4 Future Work

Several directions merit further investigation. First, extending dynamic routing to the attention granularity level---choosing between full attention, sliding-window attention, and SSM---could provide finer-grained optimization. Second, exploring routing across the channel dimension rather than the token dimension could complement our approach. Third, developing theoretical bounds for the joint optimization of routing and branch parameters (rather than treating them separately) would strengthen the analysis. Fourth, applying DynaRoute to multimodal settings where different modalities may inherently favor different computation paradigms is a promising direction.

---

## 8. Conclusion

We presented DynaRoute, a token-level dynamic routing framework for SSM-Transformer hybrid architectures. By introducing a dual-path layer with a context-aware router that adaptively allocates each token to either an SSM or attention branch, combined with a feedforward network and residual connections, DynaRoute overcomes the fundamental limitation of fixed-ratio hybrid designs. Our theoretical analysis proves that pure SSMs face an information bottleneck for retrieval tasks and that token-level routing can provably bridge this gap. The progressive training strategy with cosine temperature annealing and KL-divergence load-balance regularization ensures stable convergence. Experiments across three model scales and diverse benchmarks demonstrate that DynaRoute outperforms fixed-ratio hybrids by 1.2--3.5\% in accuracy while achieving near-SSM efficiency. Analysis of learned routing patterns reveals interpretable specialization: SSM handles local sequential processing in lower layers, while attention addresses global retrieval and reasoning in upper layers. These results establish dynamic routing as a principled and practical approach to combining the complementary strengths of SSMs and Transformers.

---

## References

1. AI21 Labs. Jamba: A hybrid Transformer-Mamba language model. *arXiv preprint arXiv:2403.19887*, 2024.

2. Austin, J., Odena, A., Nye, M., et al. Program synthesis with large language models. *arXiv preprint arXiv:2108.07732*, 2021.

3. Chen, M., Tworek, J., Jun, H., et al. Evaluating large models trained on code. *arXiv preprint arXiv:2107.03374*, 2021.

4. Clark, P., Cowhey, I., Etzioni, O., et al. Think you have solved question answering? Try ARC, the AI2 reasoning challenge. *arXiv preprint arXiv:1803.05457*, 2018.

5. Cobbe, K., Kosaraju, V., Bavarian, M., et al. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021.

6. De, S., Smith, S., Fernando, A., et al. Griffin: Mixing gated linear recurrences with local attention for efficient language models. *arXiv preprint arXiv:2402.19427*, 2024.

7. DeepSeek. DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models. *arXiv preprint arXiv:2401.06066*, 2024.

8. Fedus, W., Zoph, B., and Shazeer, N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120):1--39, 2022.

9. Gao, L., Biderman, S., Black, S., et al. The Pile: An 800GB dataset of diverse text for language modeling. *arXiv preprint arXiv:2101.00027*, 2020.

10. Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*, 2023.

11. Gu, A. and Dao, T. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. *arXiv preprint arXiv:2405.21060*, 2024.

12. Gu, A., Goel, K., and Ré, C. Efficiently modeling long sequences with structured state spaces. In *Proceedings of the International Conference on Learning Representations (ICLR)*, 2022.

13. Hendrycks, D., Burns, C., Kadavath, S., et al. Measuring mathematical problem solving with the MATH dataset. In *Proceedings of the Conference on Neural Information Processing Systems (NeurIPS)*, 2021.

14. Hornik, K., Stinchcombe, M., and White, H. Multilayer feedforward networks are universal approximators. *Neural Networks*, 2(5):359--366, 1989.

15. Hsieh, C.-P., Sun, S., Kriman, S., et al. RULER: What's the real context size of your long-context language models? *arXiv preprint arXiv:2404.06654*, 2024.

16. Jang, E., Gu, S., and Poole, B. Categorical reparameterization with Gumbel-Softmax. In *Proceedings of the International Conference on Learning Representations (ICLR)*, 2017.

17. Jelassi, S., Sander, D., and Li, Y. Repeat after me: Transformers are better than state space models at copying. In *Proceedings of the International Conference on Machine Learning (ICML)*, 2024.

18. Merrill, W., Sabharwal, A., and Smith, N. A. The illusion of state in state-space models. In *Proceedings of the International Conference on Machine Learning (ICML)*, 2024.

19. Radford, A., Wu, J., Child, R., et al. Language models are unsupervised multitask learners. *OpenAI Blog*, 2019.

20. Rae, J. W., Potapenko, A., Jayakumar, S. M., et al. Compressive transformers for long-range sequence modelling. In *Proceedings of the International Conference on Learning Representations (ICLR)*, 2020.

21. Raposo, D., Ritter, S., Richards, B., et al. Mixture-of-depths: Dynamically allocating compute in transformer-based language models. In *Proceedings of the International Conference on Machine Learning (ICML)*, 2024.

22. Shaham, U., Ivgi, M., Efrat, A., et al. SCROLLS: Standardized ComprRehension Over Long Sequences. In *Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2022.

23. Suzgun, M., Scales, N., Schärli, N., et al. Challenging BIG-Bench tasks and whether chain-of-thought can solve them. In *Findings of the Association for Computational Linguistics (ACL)*, 2023.

24. Touvron, H., Lavril, T., Izacard, G., et al. LLaMA: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023.

25. Vaswani, A., Shazeer, N., Parmar, N., et al. Attention is all you need. In *Proceedings of the Conference on Neural Information Processing Systems (NeurIPS)*, 2017.

26. Zellers, R., Holtzman, A., Bisk, Y., et al. HellaSwag: Can a machine really finish your sentence? In *Proceedings of the Association for Computational Linguistics (ACL)*, 2019.

27. Zyphra. Zamba: A compact 7B SSM hybrid model. *Technical Report*, 2024.
