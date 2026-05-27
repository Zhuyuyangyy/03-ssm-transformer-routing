# Token-Level Dynamic Routing for SSM-Transformer Hybrid Architectures: Theory and Optimal Allocation

## Abstract

Hybrid architectures combining State Space Models (SSMs) and Transformers have emerged as a promising paradigm for efficient sequence modeling. However, existing hybrid approaches rely on fixed, manually-designed layer ratios that cannot adapt to the heterogeneous computational demands of different tokens and tasks. In this paper, we propose **DynaRoute**, a token-level dynamic routing framework that learns to allocate each token to either an SSM branch or an attention branch within a unified dual-path layer. We establish a theoretical foundation by proving an information-theoretic routing lower bound (Theorem 1) and a routing sufficiency guarantee (Theorem 2), demonstrating that adaptive routing can reduce retrieval error from $\Omega((k - d_s)/n)$ to $O(1/n)$ for tasks requiring $k$ key-value pair retrievals. We introduce a lightweight router with fewer than 0.01\% additional parameters, trained with a progressive strategy incorporating load-balance regularization and entropy annealing. Experiments at 125M, 1.3B, and 7B parameter scales on language modeling, long-context reasoning, and synthetic retrieval benchmarks show that DynaRoute outperforms fixed-ratio hybrids (Jamba-style) by 1.2--3.5\% in accuracy while reducing FLOPs by 18\% through early-exit optimization. Analysis of learned routing patterns reveals interpretable specialization: attention handles positional and retrieval-intensive tokens, while SSM processes local and sequential tokens.

**Keywords:** State Space Model, Transformer, Dynamic Routing, Hybrid Architecture, Sequence Modeling, Efficient Inference

---

## 1. Introduction

The Transformer architecture (Vaswani et al., 2017) has become the dominant paradigm for sequence modeling across language, vision, and multimodal domains. Its self-attention mechanism enables global token interactions with $O(n^2)$ time and space complexity, which becomes a critical bottleneck for long sequences. State Space Models (SSMs), exemplified by S4 (Gu et al., 2022) and Mamba (Gu and Dao, 2023), offer a compelling alternative with $O(n)$ linear complexity by compressing the input sequence into a fixed-dimensional hidden state. Recent work on Mamba-2 (Gu and Dao, 2024) has further revealed a deep mathematical duality between structured SSMs and attention mechanisms through the Structured State Space Duality (SSD) framework.

Despite this theoretical connection, SSMs and Transformers exhibit fundamentally different computational characteristics. Transformers excel at tasks requiring precise in-context retrieval and arbitrary token interactions (Jelassi et al., 2024), while SSMs are more efficient at modeling local patterns and sequential dependencies. This complementarity has motivated the development of hybrid architectures such as Jamba (AI21 Labs, 2024), Zamba (Zyphra, 2024), and Griffin (De et al., 2024), which interleave SSM and attention layers in fixed ratios (e.g., 7:1 in Jamba).

However, a critical limitation of existing hybrid architectures is that the SSM-to-attention ratio is determined at design time and remains constant across all inputs, layers, and tokens. This rigid allocation is suboptimal for three reasons. First, different tasks require different balances of local processing and global retrieval. Second, even within a single sequence, individual tokens have heterogeneous computational needs---a punctuation mark may not require attention, while a key entity may demand global context. Third, the optimal allocation may vary across depth, with lower layers favoring local feature extraction and higher layers requiring more global reasoning.

To address these limitations, we propose **DynaRoute**, a token-level dynamic routing framework for SSM-Transformer hybrids. Our approach introduces a dual-path layer architecture where each layer contains both an SSM branch and an attention branch, with a lightweight router that adaptively selects the computation path for each token. Our contributions are as follows:

1. **Token-level dynamic routing architecture.** We design a dual-path hybrid layer with a router that assigns a continuous mixing weight $\alpha_t \in [0, 1]$ to each token, enabling fine-grained allocation between SSM and attention computation. The router adds fewer than 0.01\% additional parameters.

2. **Information-theoretic routing analysis.** We prove that pure SSMs face a fundamental information bottleneck for retrieval tasks (Theorem 1), and that token-level routing can provably overcome this bottleneck (Theorem 2), providing theoretical justification for adaptive allocation.

3. **Progressive training strategy.** We develop a three-phase training procedure with load-balance regularization, entropy annealing, and temperature scheduling to ensure stable routing convergence without router collapse.

4. **Comprehensive empirical evaluation.** We conduct experiments across three model scales (125M, 1.3B, 7B) on language modeling, long-context reasoning, code generation, and synthetic retrieval tasks, demonstrating consistent improvements over fixed-ratio baselines.

5. **Interpretable routing analysis.** We provide systematic analysis of learned routing patterns, revealing task-dependent and depth-dependent specialization between SSM and attention pathways.

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

**SSM Branch.** The SSM branch applies a Mamba-2 block to the input:

$$h_t^{\text{ssm}} = \text{MambaBlock}(\mathbf{X})_t$$

where $\text{MambaBlock}$ consists of selective state space processing with input-dependent parameterization as defined in Gu and Dao (2024).

**Attention Branch.** The attention branch applies standard multi-head self-attention:

$$h_t^{\text{attn}} = \text{MultiHeadAttn}(\mathbf{X})_t = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V \bigg|_t$$

where $Q, K, V$ are the query, key, and value projections.

**Router.** A lightweight router produces a per-token mixing weight:

$$\alpha_t = \sigma\left(W_r \cdot \text{LN}(x_t) + b_r\right) \in (0, 1)$$

where $W_r \in \mathbb{R}^{1 \times d}$, $b_r \in \mathbb{R}$, $\sigma$ is the sigmoid function, and LN denotes layer normalization. The total parameter count of the router is $d + 1$, which is negligible compared to the branch parameters.

**Output Mixing.** The layer output is a convex combination:

$$y_t = \alpha_t \cdot h_t^{\text{attn}} + (1 - \alpha_t) \cdot h_t^{\text{ssm}}$$

This design allows each token to be processed by the optimal combination of SSM and attention at each layer, with the routing decision made based on the token's representation.

### 3.2 Router Variants

We investigate three router designs with increasing sophistication:

**Variant A: Token-Independent Router.** The base router uses only the current token representation:

$$\alpha_t = \sigma(W_r \cdot \text{LN}(x_t) + b_r)$$

**Variant B: Context-Aware Router.** To incorporate local context, we use a small convolution over neighboring tokens:

$$\alpha_t = \sigma\left(W_r \cdot \text{LN}\left(\text{Conv1D}(x_{t-k:t+k})\right) + b_r\right)$$

where $k$ is the context window size (default $k = 3$).

**Variant C: Hierarchical Adaptive Router.** Each layer has an independent router combined with a global routing prior to prevent excessive per-layer variance:

$$\alpha_t^{(i)} = \lambda \cdot \alpha_t^{\text{global}} + (1 - \lambda) \cdot \sigma\left(W_r^{(i)} \cdot \text{LN}(x_t^{(i)}) + b_r^{(i)}\right)$$

where $\alpha_t^{\text{global}}$ is a global routing signal computed at the first layer, and $\lambda \in [0, 1]$ is a learnable mixing coefficient.

### 3.3 Training Objective

The total training loss consists of the task loss and two regularization terms:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \beta_1 \mathcal{L}_{\text{balance}} + \beta_2 \mathcal{L}_{\text{entropy}}$$

**Task Loss.** The standard cross-entropy loss for language modeling:

$$\mathcal{L}_{\text{task}} = -\frac{1}{n} \sum_{t=1}^{n} \log P(x_{t+1} | x_{\leq t})$$

**Load Balance Loss.** Inspired by Switch Transformer (Fedus et al., 2022), we prevent routing collapse by encouraging uniform load distribution:

$$\mathcal{L}_{\text{balance}} = \sum_{i=1}^{L} \left(\frac{1}{n} \sum_{t=1}^{n} \alpha_t^{(i)} - 0.5\right)^2$$

where $L$ is the number of layers. This loss penalizes layers that route excessively to one branch.

**Entropy Regularization.** To prevent premature routing determinism, we maximize the entropy of the routing distribution:

$$\mathcal{L}_{\text{entropy}} = -\frac{1}{nL} \sum_{i=1}^{L} \sum_{t=1}^{n} \left[\alpha_t^{(i)} \log \alpha_t^{(i)} + (1 - \alpha_t^{(i)}) \log (1 - \alpha_t^{(i)})\right]$$

### 3.4 Progressive Training Strategy

We observe that jointly training the router and the backbone from initialization leads to unstable routing decisions. We therefore adopt a three-phase progressive training strategy:

**Phase 1: Backbone Warmup (0\%--10\% of total steps).** The router is frozen with $\alpha_t = 0.5$ for all tokens, allowing both branches to learn useful representations.

**Phase 2: Router Awakening (10\%--50\% of total steps).** The router is unfrozen with a large load-balance coefficient $\beta_1$ to encourage exploration of diverse routing patterns.

**Phase 3: Fine-Grained Routing (50\%--100\% of total steps).** The load-balance coefficient $\beta_1$ is linearly annealed, and the entropy regularization $\beta_2$ is reduced, allowing the router to specialize.

**Temperature Annealing.** During training, we use Gumbel-Softmax (Jang et al., 2017) for differentiable discrete routing with temperature $\tau$ annealed from 1.0 to 0.1:

$$\alpha_t = \text{sigmoid}\left(\frac{\log \pi_1 - \log \pi_0 + g_1 - g_0}{\tau}\right)$$

where $g_1, g_0$ are i.i.d. Gumbel noise samples and $\pi_1, \pi_0$ are the unnormalized routing logits.

### 3.5 Inference Optimization

**Early Exit.** If all tokens in a batch have $\alpha^{(i)} \approx 0$ (or $\alpha^{(i)} \approx 1$), the unused attention (or SSM) branch can be skipped entirely for that layer, saving up to 50\% of per-layer computation:

$$\text{Skip}_{\text{attn}}^{(i)} = \mathbb{1}\left[\max_t \alpha_t^{(i)} < \epsilon\right], \quad \text{Skip}_{\text{ssm}}^{(i)} = \mathbb{1}\left[\min_t \alpha_t^{(i)} > 1 - \epsilon\right]$$

where $\epsilon = 0.01$ is a threshold.

**Batched Routing.** Tokens with similar routing weights can be grouped and processed together, reducing the overhead of running two branches:

$$\mathcal{G}_{\text{attn}} = \{t : \alpha_t > 0.5\}, \quad \mathcal{G}_{\text{ssm}} = \{t : \alpha_t \leq 0.5\}$$

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

**Theorem 2 (Routing Sufficiency).** *Let $f_\theta$ be a token-level router parameterized by $\theta$ with $|\theta| = O(d)$ parameters. Under the assumption that the optimal routing decision depends on a bounded function of the input token (i.e., $\alpha_t^* = g(x_t)$ for some Lipschitz-continuous $g$), the trained router $f_{\theta^*}$ satisfies:*

$$\mathbb{E}_t\left[\left|f_{\theta^*}(x_t) - \alpha_t^*\right|\right] \leq O\left(\frac{d \cdot \log n}{n}\right)$$

*with $O(n \cdot \log(1/\varepsilon))$ gradient steps for convergence to $\varepsilon$-optimal routing.*

*Proof.* (Sketch) Since $g$ is Lipschitz-continuous and the router $f_\theta$ is a single-layer sigmoid network with $d$-dimensional input, by the universal approximation theorem for sigmoid networks (Hornik et al., 1989), there exists $\theta^*$ such that $\|f_{\theta^*} - g\|_\infty < \varepsilon$ for any $\varepsilon > 0$. The parameter count is $O(d)$, and by standard generalization bounds for neural networks with $n$ training samples, the sample complexity is $O(d \log n / \varepsilon^2)$. SGD convergence to $\varepsilon$-optimality requires $O(n \log(1/\varepsilon))$ steps under standard smoothness assumptions. $\square$

### 4.3 Complexity Analysis

**Proposition 1 (Computational Complexity).** *A $L$-layer DynaRoute model with hidden dimension $d$, sequence length $n$, and state dimension $d_s$ has per-token computational complexity:*

$$\text{FLOPs}_{\text{DynaRoute}} = L \left[\bar{\alpha} \cdot O(nd) + (1 - \bar{\alpha}) \cdot O(d \cdot d_s) + O(d)\right]$$

*where $\bar{\alpha} = \frac{1}{n}\sum_t \alpha_t$ is the average routing weight across the sequence. The router overhead $O(d)$ is negligible compared to branch computation.*

When $\bar{\alpha} \approx 0$ (predominantly SSM), the complexity reduces to $O(L \cdot d \cdot d_s)$, approaching pure SSM efficiency. When $\bar{\alpha} \approx 1$ (predominantly attention), it becomes $O(L \cdot n \cdot d)$, approaching Transformer complexity. The hybrid gracefully interpolates between these extremes.

---

## 5. Algorithm

We summarize the forward pass of a single DynaRoute layer in Algorithm 1 and the progressive training procedure in Algorithm 2.

**Algorithm 1: DynaRoute Layer Forward Pass**

```
Input: X in R^{n x d}, layer parameters theta_ssm, theta_attn, theta_r
Output: Y in R^{n x d}

1.  // SSM Branch
2.  H_ssm = MambaBlock(X; theta_ssm)          // shape: (n, d)
3.
4.  // Attention Branch
5.  Q, K, V = Linear(X), Linear(X), Linear(X)
6.  H_attn = softmax(Q K^T / sqrt(d_k)) V    // shape: (n, d)
7.
8.  // Routing
9.  X_norm = LayerNorm(X)
10. logits = W_r @ X_norm + b_r              // shape: (n, 1)
11. alpha = sigmoid(logits / tau)             // shape: (n, 1)
12.     // During training, replace sigmoid with Gumbel-Softmax
13.
14. // Mixing
15. Y = alpha * H_attn + (1 - alpha) * H_ssm  // shape: (n, d)
16.
17. return Y
```

**Algorithm 2: Progressive Training Procedure**

```
Input: model M, dataset D, total steps T, coefficients beta1_init, beta2
Output: Trained model M

1.  // Phase 1: Backbone Warmup (0 to 0.1T)
2.  for step = 1 to 0.1T:
3.      freeze M.router
4.      alpha = 0.5  for all tokens
5.      L = CrossEntropy(M(X), Y)
6.      update M.backbone
7.
8.  // Phase 2: Router Awakening (0.1T to 0.5T)
9.  unfreeze M.router
10. for step = 0.1T+1 to 0.5T:
11.     beta1 = beta1_init
12.     tau = 1.0 - 0.9 * (step - 0.1T) / (0.4T)
13.     L = L_task + beta1 * L_balance + beta2 * L_entropy
14.     update M
15.
16. // Phase 3: Fine-Grained Routing (0.5T to T)
17. for step = 0.5T+1 to T:
18.     beta1 = beta1_init * (T - step) / (0.5T)
19.     tau = max(0.1, tau_prev - 0.9 * delta)
20.     L = L_task + beta1 * L_balance + beta2 * L_entropy
21.     update M
22.
23. return M
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

**Training Details.** We pretrain on The Pile for 100B tokens following the LLaMA recipe (Touvron et al., 2023). The router uses $\beta_1^{\text{init}} = 0.01$ and $\beta_2 = 0.001$. We use cosine learning rate scheduling with 1000 warmup steps, weight decay 0.1, and gradient clipping at 1.0.

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
| **DynaRoute** | **1.39T** | **22,100** | **5.4** | **0.045** |
| **DynaRoute+EE** | **1.14T** | **26,500** | **5.0** | **0.038** |

DynaRoute achieves Transformer-competitive quality with near-SSM efficiency. With early exit (DynaRoute+EE), the model surpasses even the pure Mamba-2 in throughput, as many layers route predominantly to the lightweight SSM branch, allowing the attention branch to be skipped.

### 6.6 Ablation Studies

**Table 8: Ablation study at 125M scale on The Pile (perplexity).**

| Configuration | PPL | $\Delta$ |
|:---|:---:|:---:|
| DynaRoute (full) | 15.2 | -- |
| w/o load-balance loss | 15.8 | +0.6 |
| w/o entropy regularization | 15.5 | +0.3 |
| w/o progressive training | 16.1 | +0.9 |
| Fixed $\alpha = 0.5$ (no routing) | 15.9 | +0.7 |
| Random routing | 16.4 | +1.2 |
| Router variant A (token-only) | 15.2 | 0.0 |
| Router variant B (context-aware) | 15.0 | -0.2 |
| Router variant C (hierarchical) | 15.1 | -0.1 |

Progressive training is the most critical component (+0.9 perplexity when removed), followed by the load-balance loss (+0.6). Context-aware routing (Variant B) provides a small additional improvement, suggesting that local context helps but is not essential.

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

1. **Router simplicity is sufficient.** The lightweight token-independent router (Variant A) performs nearly as well as more complex alternatives. The additional parameters and computation of context-aware or hierarchical routers provide marginal improvements that may not justify the overhead.

2. **Progressive training is essential.** Joint training of the router and backbone from initialization leads to routing collapse or instability. The three-phase strategy is critical for convergence.

3. **SSM dominance in lower layers.** The consistent pattern of SSM-favored routing in lower layers suggests that future hybrid architectures can safely use SSM-only lower layers, reserving the dual-path design for upper layers only.

4. **Attention is needed for retrieval, not generation.** The routing analysis shows that attention is most valuable for tokens involved in cross-referencing or information retrieval, not for autoregressive generation of common patterns.

### 7.3 Limitations

Our work has several limitations. First, the theoretical analysis assumes that optimal routing depends on a bounded function of the input token (Theorem 2), which may not hold for tasks requiring complex multi-step reasoning about routing decisions. Second, our experiments are limited to decoder-only architectures; extending to encoder-decoder or bidirectional models requires further investigation. Third, the early-exit optimization introduces hardware-specific efficiency gains that may not transfer across different GPU architectures. Finally, while we analyze routing patterns post-hoc, we do not provide a mechanism for users to specify desired routing behavior, which could be valuable for domain-specific applications.

### 7.4 Future Work

Several directions merit further investigation. First, extending dynamic routing to the attention granularity level---choosing between full attention, sliding-window attention, and SSM---could provide finer-grained optimization. Second, exploring routing across the channel dimension rather than the token dimension could complement our approach. Third, developing theoretical bounds for the joint optimization of routing and branch parameters (rather than treating them separately) would strengthen the analysis. Fourth, applying DynaRoute to multimodal settings where different modalities may inherently favor different computation paradigms is a promising direction.

---

## 8. Conclusion

We presented DynaRoute, a token-level dynamic routing framework for SSM-Transformer hybrid architectures. By introducing a dual-path layer with a lightweight router that adaptively allocates each token to either an SSM or attention branch, DynaRoute overcomes the fundamental limitation of fixed-ratio hybrid designs. Our theoretical analysis proves that pure SSMs face an information bottleneck for retrieval tasks and that token-level routing can provably bridge this gap. Experiments across three model scales and diverse benchmarks demonstrate that DynaRoute outperforms fixed-ratio hybrids by 1.2--3.5\% in accuracy while achieving near-SSM efficiency through early-exit optimization. Analysis of learned routing patterns reveals interpretable specialization: SSM handles local sequential processing in lower layers, while attention addresses global retrieval and reasoning in upper layers. These results establish dynamic routing as a principled and practical approach to combining the complementary strengths of SSMs and Transformers.

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
