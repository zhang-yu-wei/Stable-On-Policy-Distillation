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

| Paper | arXiv | Focus |
|-------|-------|-------|
| **Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes** | [2603.25562](https://arxiv.org/abs/2603.25562) | Identifies three failure modes of sampled-token OPD: token-level signal imbalance, unreliable teacher guidance after student prefix drift, and tokenizer/special-token mismatch. Proposes teacher top-K local support matching. |
| **Demystifying OPD: Length Inflation and Stabilization Strategies for LLMs** | [2604.08527](https://arxiv.org/abs/2604.08527) | Discovers length inflation during OPD training, leading to truncation collapse, repetition saturation, biased gradients, and sharp validation degradation. Proposes StableOPD. |

> **[Observation on 2603.25562 — teacher top-K vs student top-K diverges under multi-task]** In single-task math reasoning, teacher top-K, student top-K, and teacher top-K w/ sampled token all perform comparably. However, in multi-task (alternating math + agentic), student top-K and teacher top-K w/ sampled variants collapse on the math benchmarks while teacher top-K remains strong (41.7 vs 28.4/26.9 math avg). Likely cause: multi-task training causes drastic student distribution shifts between tasks, making student top-K an unstable moving target for the support set. Teacher top-K provides a fixed anchor that doesn't co-vary with student drift. This problem **does not arise under full-logit OPD** (full-vocabulary reverse KL), because there is no discrete support-set construction — all tokens participate in the gradient weighted by student probability, so there is no sensitivity to which tokens are "selected" into the support. The signal imbalance failure mode (most sampled tokens receiving negative reward) is also absent: full-logit KL distributes gradients across the entire vocabulary at each position rather than concentrating on a single sampled token.
| **Rethinking On-Policy Distillation of LLMs: Phenomenology, Mechanism, and Recipe** | [2604.13016](https://arxiv.org/abs/2604.13016) | Identifies two necessary conditions for OPD success: student-teacher compatible thinking patterns and genuine novel capability from teacher. Token-level analysis shows successful OPD is "progressive alignment on high-probability tokens at student-visited states" (97-99% probability concentration). Proposes off-policy cold start and teacher-aligned prompt selection as recovery strategies. Questions long-horizon scalability. |

> **[Insight on 2604.13016 — overlap finding is largely a sampled/top-k artifact]** This paper's core finding — that OPD learning happens almost entirely within the student-teacher top-k overlap region (default k=16) — is mechanistically tied to the **support-set bottleneck** of sampled-token and top-k OPD. Non-overlap tokens carry negligible probability mass, so they contribute near-zero gradients regardless of their mismatch. Under **full-logit OPD** (full-vocabulary reverse KL), this issue largely disappears: all tokens participate in the gradient weighted by student probability, and there is no artificial support boundary. Therefore, the "thinking-pattern compatibility" prerequisite identified here is partially a limitation of the sparse OPD formulation, not a fundamental property of distillation. For self-distillation settings where student = teacher (or close checkpoint), overlap is trivially high, making this condition automatically satisfied.

> **[Hypothesis on 2604.08527]** This paper shows that length inflation in standard OPD is irreversible once repetition saturation is triggered. However, in our on-policy self-distillation experiments, inflation appears early but **self-recovers**. Possible explanation: our setup includes explicit truncation feedback (the model receives a signal that its response was truncated), which provides an implicit negative reward that suppresses repetition before it reaches the irreversible tipping point. This is a different stabilization mechanism from Stable-OPD's mixture distillation + divergence constraint. **TODO:** Ablate the truncation feedback to verify causality.
>
> **[Their solution on 2604.08527 — Stable-OPD]** Two complementary mechanisms: (1) **Mixture distillation** — blends on-policy rollouts with off-policy "golden" data (complete, non-truncated, non-repetitive trajectories) via a combined loss `L_mix = L_OPD + λ_gold * L_SFT`. The golden data anchors training on high-quality sequences, preventing on-policy rollouts from being dominated by degenerate samples. (2) **KL regularization** — adds a per-prefix KL divergence constraint against a reference policy (initial student checkpoint): `L_Stable-OPD = L_mix + β_KL * E[KL(π_θ(·|s_t) || π_ref(·|s_t))]`. This limits abrupt policy drift that would otherwise push the model into the repetition-favoring regime. The two are complementary: KL alone gives modest gains (28.0→29.7), mixture distillation alone is stronger (29.7→35.7 on 1.5B), and combining both gives the best result.

**PDF path:** `papers/failure_mode_analysis/`

---

## 2. Stabilization & Target Shaping

| Paper | arXiv | Focus |
|-------|-------|-------|
| **Stable On-Policy Distillation through Adaptive Target Reformulation (Veto)** | [2601.07155](https://arxiv.org/abs/2601.07155) | Shows that when the student-teacher gap is too large, FKL produces pathological gradients while RKL causes diversity collapse. Proposes Veto, constructing intermediate targets in logit space. |
| **Entropy-Aware On-Policy Distillation of Language Models** | [2603.07079](https://arxiv.org/abs/2603.07079) | Shows that RKL's mode-seeking behavior on high-entropy teacher tokens reduces diversity and produces unstable signals. Proposes entropy-aware FKL/RKL mixture. |

**PDF path:** `papers/stabilization_and_target_shaping/`

---

## 3. OPD-RL Connection & Reward Extrapolation

| Paper | arXiv | Focus |
|-------|-------|-------|
| **Scaling Reasoning Efficiently via Relaxed On-Policy Distillation (REOPOLD)** | [2603.11137](https://arxiv.org/abs/2603.11137) | Interprets teacher-student log-likelihood ratio as a token reward. Shows OPD is prone to instability and negative transfer. Relaxes strict imitation via reward clipping and entropy-based dynamic sampling. |
| **Learning beyond Teacher: Generalized OPD with Reward Extrapolation (ExOPD)** | [2602.12125](https://arxiv.org/abs/2602.12125) | Formalizes OPD as dense KL-constrained RL. Through reward scaling / ExOPD, enables student to potentially surpass teacher. |
| **The Extrapolation Cliff in OPD of Near-Deterministic Structured Outputs** | [2605.08737](https://arxiv.org/abs/2605.08737) | Shows that reward extrapolation can improve student performance, but beyond a threshold causes structured-output contract collapse. Provides clip-safety threshold. |

**PDF path:** `papers/opd_rl_connection_and_reward/`

---

## Key Takeaways (Cross-Paper)

- **Dense token-level signals are not inherently reliable**: Teacher guidance after student prefix drift can be misleading (Paper 1); RKL signals on high-entropy tokens are unstable (Paper 4).
- **Long-sequence training has systematic risks**: Length inflation → truncation → biased gradients forms a complete causal chain (Paper 2).
- **OPD success requires preconditions**: Student-teacher must share compatible thinking patterns, and teacher must provide genuinely novel capability beyond student's training data (Paper 3). Without these, distillation fails or produces degenerate behavior.
- **Successful OPD is selective, not uniform**: Token-level mechanism shows 97-99% probability concentration on high-probability tokens at student-visited states — OPD works via progressive alignment, not wholesale distribution matching (Paper 3).
- **Excessive distribution gap is harmful**: FKL produces pathological gradients, RKL produces diversity collapse — intermediate targets are needed (Paper 4).
- **OPD can be viewed as dense reward RL**: But teacher reward must be used conservatively (clipping, entropy filtering), otherwise instability / negative transfer occurs (Papers 6, 7).
- **Reward extrapolation has boundaries**: Beyond a threshold, structured-output collapse occurs (Paper 8).
- **Long-horizon scalability remains an open question**: OPD effectiveness degrades on tasks requiring extended reasoning (Paper 3), and sequence-level objectives introduce high gradient variance (Paper 1).
- **Full-logit OPD sidesteps many sampled/top-K failure modes**: Signal imbalance (most tokens getting negative reward), support-set instability under multi-task training, student-teacher overlap prerequisites, and tokenizer mismatch on individual tokens — these are all artifacts of concentrating supervision on one or a few tokens. Full-vocabulary reverse KL distributes gradients across all tokens weighted by student probability, eliminating discrete support boundaries. The tradeoff is compute cost (materializing full vocabulary logits at each position), but the resulting training signal is denser, more stable, and less sensitive to distribution shift between tasks.
