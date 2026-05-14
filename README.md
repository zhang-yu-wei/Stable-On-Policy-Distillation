# On-Policy Distillation (OPD) Failure Modes — Paper Collection

A curated collection of recent papers investigating failure modes and stabilization mechanisms in On-Policy Distillation for LLMs. Papers are organized into three thematic categories.

---

## Our Position: Design Considerations for Stable On-Policy Distillation

### 1. The teacher is a language model, not a critic

The teacher in OPD is a language model: it assigns probability to the next token that makes the sequence *locally coherent* (grammatical, fluent, contextually natural). But the implicit goal of OPD is to improve *task performance* — i.e., to select tokens that maximize future reward (correctness, helpfulness, task completion). These are fundamentally different objectives. A language model is not a critic: it does not reason about whether a token leads to a correct final answer, it only predicts what token would naturally follow the prefix. This mismatch explains why teacher guidance becomes unreliable on student-drifted prefixes (the teacher happily assigns high probability to fluent continuations of degenerate text), why repetition loops receive positive local reward (repeated tokens are locally probable), and why OPD struggles on long-horizon tasks (local fluency ≠ global correctness). The failure modes documented across these papers are not bugs in specific implementations — they are symptoms of using a next-token predictor as a surrogate for a value function.

### 2. Full-logit KL resolves most support-set failure modes

Many identified failure modes — signal imbalance, support-set instability under multi-task, student-teacher overlap prerequisites, tokenizer mismatch on individual tokens — are artifacts of *sparse* OPD formulations (sampled-token or top-K). Full-vocabulary reverse KL eliminates the discrete support boundary entirely: all tokens participate in the gradient weighted by student probability. There is no "selected" support set to become unstable, no single sampled token to concentrate noisy signal on, and no sensitivity to whether a particular token falls inside or outside an arbitrary top-K cutoff. The tradeoff is compute cost (materializing full-vocabulary logits), but the training signal is fundamentally denser, more stable, and less sensitive to distribution shift.

### 3. JSD (or arithmetic mix-target) is likely more stable than pure reverse KL

Pure reverse KL is mode-seeking: it encourages the student to concentrate on the teacher's dominant modes, sacrificing diversity. Pure forward KL is zero-avoiding: it forces the student to cover all teacher modes, causing gradient explosion on student-ignorant tokens. JSD — or equivalently, targeting an arithmetic mixture `M = α·P_T + (1-α)·P_S` — naturally interpolates between the two extremes. Because the target always has nonzero mass wherever *either* distribution has mass, it avoids both gradient explosion (no division by near-zero student probability) and mode collapse (the target retains student diversity). This is the same stabilization mechanism as Veto (geometric interpolation) but simpler and more principled. For on-policy self-distillation where student ≈ teacher, JSD with α ≈ 0.5 is particularly natural since the mixture is close to both distributions throughout training.

### 4. Reward clipping prevents heavy-tailed negative rewards from dominating

Even under full-logit KL, the token-level reward `R = log(π_T/π_θ)` exhibits a heavy-tailed negative distribution when the student assigns significant mass to tokens the teacher considers unlikely (see REOPOLD Figure 4). Our approach enforces a lower bound on teacher probability relative to student: `π_T(v) ≥ c · π_θ(v)` (e.g., c = 0.2), which directly floors the reward at `log(c) ≈ -1.6`. This is equivalent in effect to REOPOLD's mixture-based clipping `R̂ = max(sg(R), log λ/(1-λ))`, which derives the same type of floor from a convex mixture bound (for λ ≈ 0.17, the floor is also ≈ -1.6). Both prevent extreme negative rewards from dominating gradient updates — the difference is parameterization: ours is a relative ratio bound (more intuitive), REOPOLD's comes from a theoretical mixture stability argument. This is orthogonal to support-set and divergence-direction choices: it addresses *reward magnitude* rather than which tokens or which KL direction is used.

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
| **Rethinking On-Policy Distillation of LLMs: Phenomenology, Mechanism, and Recipe** | [2604.13016](https://arxiv.org/abs/2604.13016) | Identifies two necessary conditions for OPD success: student-teacher compatible thinking patterns and genuine novel capability from teacher. Token-level analysis shows successful OPD is "progressive alignment on high-probability tokens at student-visited states" (97-99% probability concentration). Proposes off-policy cold start and teacher-aligned prompt selection as recovery strategies. Questions long-horizon scalability. |
| **The Many Faces of On-Policy Distillation: Pitfalls, Mechanisms, and Fixes** | [2605.11182](https://arxiv.org/abs/2605.11182) | Examines when OPD/OPSD succeeds or fails. Identifies distribution misalignment, unstable optimization from biased gradients, and inadequate aggregation of teacher outputs. Finds math reasoning highly sensitive to teacher selection and loss design. OPSD works when privileged info represents shared rules. Proposes stop-gradient TopK objectives, RLVR-adapted teachers, and SFT-stabilized students. |
| **Unmasking On-Policy Distillation: Where It Helps, Where It Hurts, and Why** | [2605.10889](https://arxiv.org/abs/2605.10889) | Develops a training-free token-level diagnostic framework using "ideal per-node gradient" and targeted-rollout estimation. Shows distillation guidance has higher alignment with ideal signal on incorrect rollouts than correct ones — teacher signals become noisy when students already perform well. No universally optimal configuration; best approach depends jointly on student capacity and task. |

> **[Note on 2603.25562 — signal imbalance explained]** "Signal imbalance" refers to the skewed distribution of per-token rewards in sampled-token OPD. The update at each step is driven by the log-ratio `log q(y_t|c_t) - log π_θ(y_t|c_t)`. Empirically, the vast majority of sampled tokens have higher student probability than teacher probability, yielding negative rewards. Only a small minority of tokens receive positive rewards. This means optimization is dominated by a few locally-positive tokens (often high-frequency fillers or short continuations that receive favorable local scores but contribute little to trajectory quality), while most positions provide only penalty signals. The proposed fix — teacher top-K local support matching — replaces single-token comparison with a distribution-level comparison over K teacher-supported tokens at each prefix, making the signal denser and more balanced.

> **[Observation on 2603.25562 — teacher top-K vs student top-K diverges under multi-task]** In single-task math reasoning, teacher top-K, student top-K, and teacher top-K w/ sampled token all perform comparably. However, in multi-task (alternating math + agentic), student top-K and teacher top-K w/ sampled variants collapse on the math benchmarks while teacher top-K remains strong (41.7 vs 28.4/26.9 math avg). Likely cause: multi-task training causes drastic student distribution shifts between tasks, making student top-K an unstable moving target for the support set. Teacher top-K provides a fixed anchor that doesn't co-vary with student drift. This problem **does not arise under full-logit OPD** (full-vocabulary reverse KL), because there is no discrete support-set construction — all tokens participate in the gradient weighted by student probability, so there is no sensitivity to which tokens are "selected" into the support. The signal imbalance failure mode (most sampled tokens receiving negative reward) is also absent: full-logit KL distributes gradients across the entire vocabulary at each position rather than concentrating on a single sampled token.

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
| **Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe** | [2605.03677](https://arxiv.org/abs/2605.03677) | Addresses insufficient exploration of informative states and unreliable teacher supervision. Student perspective: data balancing strategies for exploring student-generated states. Teacher perspective: outcome-guided margin calibration ensuring token-level guidance maintains order consistency between correct/incorrect trajectories. Generalizes across LLMs/MLLMs, single/multi-teacher, and cross-modal settings. |

> **[Insight on 2601.07155 — Veto is essentially equivalent to JSD / mix-target]** Veto constructs an intermediate target via geometric interpolation (Product of Experts): `Q ∝ P_T · P_S^β`. JSD / arithmetic mix-target uses: `M = α·P_T + (1-α)·P_S`. Both solve the same core problem: avoid targeting the raw teacher directly, instead constructing a bridge distribution between teacher and student that (1) prevents FKL gradient explosion on student-ignorant tokens and (2) prevents RKL mode collapse by preserving student diversity. The difference is geometric (consensus — token needs both to agree) vs arithmetic (union — either one contributing is enough). JSD is arguably simpler and achieves the same stabilization effect. Veto's additional claims are a sharpening effect (student converges to `P_T^{1/(1-β)}`) and a β-decay curriculum, but these are secondary to the core mechanism of target softening.

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
