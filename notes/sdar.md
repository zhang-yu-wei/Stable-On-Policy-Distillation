# SDAR — Reading Note

Source: [Self-Distilled Agentic Reinforcement Learning](https://arxiv.org/abs/2605.15155)

## One-line Summary

SDAR keeps GRPO/RL as the main optimization objective for multi-turn agents, and adds OPSD only as a **smoothly gated auxiliary loss**. The central idea is to trust PI-conditioned teacher **endorsements** more than teacher **rejections**.

## Problem

Multi-turn agent RL has sparse, trajectory-level rewards. A WebShop or ALFWorld trajectory may contain many reasoning/action tokens, but the reward usually arrives only after the full interaction. OPSD seems attractive because a privileged teacher branch can provide dense token-level guidance.

The paper argues that naive OPSD is brittle for multi-turn agents:

- **Compounding instability:** once the student drifts from the teacher-supported path, later teacher supervision can become unreliable.
- **Teacher is not truly stronger:** the teacher is the same model with extra training-only context, usually retrieved skills. If the skill is bad or poorly used, teacher preferences can be wrong.
- **Negative teacher gaps are ambiguous:** when the teacher assigns lower probability to a student token, that may mean the token is bad, but it may also mean the retrieved skill is irrelevant, incomplete, redundant, or misapplied.

## Setup

At each token position, there are two branches:

- **Student** `π_S(· | s_t)`: no privileged skill/context.
- **Privileged teacher** `π_T(· | s_t, c)`: same model but conditioned on retrieved training-only skill/context `c`.

The student samples token `y_t`. SDAR computes the teacher-student log-probability gap on that same sampled token:

```
Δ_t = log π_T(y_t | s_t, c) - log π_S(y_t | s_t)
```

Interpretation:

- `Δ_t > 0`: the teacher assigns **higher** probability to the token the student already sampled.
- `Δ_t < 0`: the teacher assigns **lower** probability to the token the student sampled.

## Core Mechanism: Asymmetric Trust

SDAR treats `Δ_t > 0` and `Δ_t < 0` differently.

For a **positive gap**, the privileged teacher endorses an on-policy student token. This is relatively trustworthy because:

- the token is already reachable under the student;
- the teacher is not forcing the student onto a foreign trajectory;
- privileged context agrees with what the student is trying to do.

For a **negative gap**, the teacher rejects the student token. SDAR does not trust this rejection as strongly because the rejection might come from noisy privileged context rather than true task knowledge.

So the practical bias is:

```
positive gap  -> stronger distillation / promote this token
negative gap  -> weaker distillation / avoid harsh suppression
```

This matches our broader OPD view: the teacher LM is often more reliable as a **positive recommender** on reachable student behavior than as a **critic** that suppresses sampled tokens.

## Gate

SDAR maps detached token-level signals into a smooth sigmoid gate:

```
g_t = sigmoid(k · score_t)
```

The paper considers several scores:

- **Gap gating:** based on `Δ_t`, promoting positive-gap tokens and attenuating negative-gap tokens.
- **Entropy gating:** based on student entropy, focusing distillation where the student is uncertain.
- **Soft-OR gating:** combines gap and entropy signals.

The gate is detached, so it controls the strength of the OPSD auxiliary loss without changing the semantics of the main RL objective.

## Objective

The total training objective is:

```
L_total = L_GRPO + λ · L_gated_OPSD
```

Important design choice: GRPO remains the backbone. OPSD is not allowed to dominate training; it only adds token-level help where the gate decides the teacher signal is useful.

## Empirical Claim

The paper evaluates on multi-turn agent benchmarks such as ALFWorld, WebShop, and Search-QA. It reports that:

- standalone OPSD can collapse badly, especially on Search-QA;
- naive GRPO+OPSD can be unstable because distillation gradients overwhelm RL;
- SDAR improves over GRPO and hybrid baselines by filtering token-level teacher guidance.

## Relation to Pedagogical RL

SDAR and Pedagogical RL attack different sides of the same problem.

- **SDAR:** assumes the rollout already exists and asks which token-level teacher signals should affect the student.
- **Pedagogical RL:** asks how to train a privileged teacher to generate better trajectories in the first place.

SDAR's positive-gap gating is close in spirit to surprisal-gated assimilation: both prefer updates that stay within the student's current support. The difference is that SDAR checks whether the privileged teacher endorses the sampled token, while Pedagogical RL checks whether teacher-generated tokens are learnable under the student.

## Takeaway for OPD

SDAR is evidence that naive "RL + dense distillation" is not automatically stable. The useful principle is:

> Use RL for task-level direction, and use distillation as a filtered auxiliary signal, mostly trusting teacher endorsements on reachable student behavior.
