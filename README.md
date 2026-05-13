# On-Policy Distillation (OPD) Failure Modes — Paper Collection

A curated collection of recent papers investigating failure modes and stabilization mechanisms in On-Policy Distillation for LLMs. Papers are organized into three thematic categories.

---

## Directory Structure

```
papers/
├── failure_mode_analysis/             # Direct analysis of OPD failure modes
├── stabilization_and_target_shaping/  # Target/loss modification approaches
└── opd_rl_connection_and_reward/      # OPD-RL connection & reward extrapolation
```

---

## 1. Failure Mode Analysis

| Paper | arXiv | Focus | Relevance to Our Work |
|-------|-------|-------|----------------------|
| **Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes** | [2603.25562](https://arxiv.org/abs/2603.25562) | Identifies three failure modes of sampled-token OPD: token-level signal imbalance, unreliable teacher guidance after student prefix drift, and tokenizer/special-token mismatch. Proposes teacher top-K local support matching. | Supports the claim that dense token signals in OPD are not inherently reliable, especially when teacher signal degrades along long trajectories. |
| **Demystifying OPD: Length Inflation and Stabilization Strategies for LLMs** | [2604.08527](https://arxiv.org/abs/2604.08527) | Discovers length inflation during OPD training, leading to truncation collapse, repetition saturation, biased gradients, and sharp validation degradation. Proposes StableOPD. | Directly related to the truncation/length instability phenomena in our paper. |

> **[Hypothesis]** This paper shows that length inflation in standard OPD is irreversible once repetition saturation is triggered. However, in our on-policy self-distillation experiments, inflation appears early but **self-recovers**. Possible explanation: our setup includes explicit truncation feedback (the model receives a signal that its response was truncated), which provides an implicit negative reward that suppresses repetition before it reaches the irreversible tipping point. This is a different stabilization mechanism from Stable-OPD's mixture distillation + divergence constraint. **TODO:** Ablate the truncation feedback to verify causality.

**PDF path:** `papers/failure_mode_analysis/`

---

## 2. Stabilization & Target Shaping

| Paper | arXiv | Focus | Relevance to Our Work |
|-------|-------|-------|----------------------|
| **Stable On-Policy Distillation through Adaptive Target Reformulation (Veto)** | [2601.07155](https://arxiv.org/abs/2601.07155) | Shows that when the student-teacher gap is too large, FKL produces pathological gradients while RKL causes diversity collapse. Proposes Veto, constructing intermediate targets in logit space. | Argues that a stronger teacher / larger distribution gap is not necessarily better — target shaping is needed. |
| **Entropy-Aware On-Policy Distillation of Language Models** | [2603.07079](https://arxiv.org/abs/2603.07079) | Shows that RKL's mode-seeking behavior on high-entropy teacher tokens reduces diversity and produces unstable signals. Proposes entropy-aware FKL/RKL mixture. | Connects to our concern about whether token-level advantage / all-logits supervision is reliable. |

**PDF path:** `papers/stabilization_and_target_shaping/`

---

## 3. OPD-RL Connection & Reward Extrapolation

| Paper | arXiv | Focus | Relevance to Our Work |
|-------|-------|-------|----------------------|
| **Scaling Reasoning Efficiently via Relaxed On-Policy Distillation (REOPOLD)** | [2603.11137](https://arxiv.org/abs/2603.11137) | Interprets teacher-student log-likelihood ratio as a token reward. Shows OPD is prone to instability and negative transfer. Relaxes strict imitation via reward clipping and entropy-based dynamic sampling. | Serves as related work for "OPD ≈ dense reward policy optimization, but teacher reward must be used selectively/conservatively." |
| **Learning beyond Teacher: Generalized OPD with Reward Extrapolation (ExOPD)** | [2602.12125](https://arxiv.org/abs/2602.12125) | Formalizes OPD as dense KL-constrained RL. Through reward scaling / ExOPD, enables student to potentially surpass teacher. | Fits in OPD-RL connection discussion; should be presented alongside extrapolation cliff to avoid over-selling benefits. |
| **The Extrapolation Cliff in OPD of Near-Deterministic Structured Outputs** | [2605.08737](https://arxiv.org/abs/2605.08737) | Shows that reward extrapolation can improve student performance, but beyond a threshold causes structured-output contract collapse. Provides clip-safety threshold. | Demonstrates that reward extrapolation / sharpening has a safety boundary. |

**PDF path:** `papers/opd_rl_connection_and_reward/`

---

## Key Takeaways (Cross-Paper)

- **Dense token-level signals are not inherently reliable**: Teacher guidance after student prefix drift can be misleading (Paper 1); RKL signals on high-entropy tokens are unstable (Paper 4).
- **Long-sequence training has systematic risks**: Length inflation → truncation → biased gradients forms a complete causal chain (Paper 2).
- **Excessive distribution gap is harmful**: FKL produces pathological gradients, RKL produces diversity collapse — intermediate targets are needed (Paper 3).
- **OPD can be viewed as dense reward RL**: But teacher reward must be used conservatively (clipping, entropy filtering), otherwise instability / negative transfer occurs (Papers 5, 6).
- **Reward extrapolation has boundaries**: Beyond a threshold, structured-output collapse occurs (Paper 7).
