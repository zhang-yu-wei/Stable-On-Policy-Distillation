# Losses in `papers/opd_rl_connection_and_reward`

This note compares the objectives used by the papers in
`papers/opd_rl_connection_and_reward`. The common theme is the tension between
dense teacher supervision and reward-grounded policy optimization.

## Quick Taxonomy

Most papers can be understood by asking two questions:

1. **What determines the update direction?**
   - Teacher distribution matching: the teacher controls the direction.
   - Verifiable reward / GRPO: the environment controls the direction.
   - Hybrid: reward controls the coarse direction, while teacher signals reshape
     magnitude, trust region, routing, or auxiliary loss strength.

2. **Where does dense token-level credit enter?**
   - As a KL/JSD loss against teacher logits.
   - As a token-level advantage or reward.
   - As a multiplicative reweighting of GRPO advantages.
   - As a routed/gated auxiliary loss on selected samples or tokens.
   - As an offline distillation target.

## Shared Baselines

### GRPO / RLVR

For a prompt `x`, sample a group of responses, score each response with a
verifier reward `R(x, y)`, normalize within the group, and use the same advantage
for every token in a response:

```text
A_i = (R(x, y_i) - mean_G R) / std_G R
L_GRPO = E sum_t min(r_t A_i, clip(r_t, 1-eps, 1+eps) A_i)
```

The strength is reward grounding. The weakness is coarse credit: every token in
the trajectory inherits the same scalar.

### OPD / OPSD

On-policy distillation uses student rollouts, then matches a teacher on those
rollout prefixes:

```text
L_OPD = E_{y ~ pi_S(.|x)} sum_t D(pi_T(. | x, y_<t) || pi_S(. | x, y_<t))
```

Variants differ in divergence direction: forward KL, reverse KL, JSD, or
mixtures. OPSD uses the same model as teacher and student, but gives the teacher
privileged context such as feedback, a correct sibling rollout, a solution, or a
diagnostic.

Reverse-KL OPD can be read as a policy-gradient update with dense token reward:

```text
r_t = log pi_T(y_t | x, y_<t) - log pi_S(y_t | x, y_<t)
```

That dense signal is the source of OPD's sample efficiency, but also the source
of many failure modes when the teacher is privileged, stale, noisy, or applied
to all tokens.

## Paper-by-Paper Summary

| Paper | Main loss/objective | What is distinctive |
|---|---|---|
| `Survey_OPD_LLM.pdf` | Not a new loss; frames OPD as `E_{y~p_theta} D_f(p_T || p_theta)` over student-sampled trajectories. | Provides the high-level axes: divergence choice, signal source, and stabilization. |
| `dGRPO_Combining_Optimization_Distillation.pdf` | `J_dGRPO = J_GRPO - beta * KL(pi_theta || pi_teacher)` on student rollouts. | The simplest additive hybrid: sparse reward optimization plus dense teacher regularization. |
| `Reinforcement_Aware_KD.pdf` | Replaces the GRPO ratio with a teacher/old-policy mixture ratio: `r_TRRD = r_GRPO^gamma * r_T^(1-gamma)`. | Folds distillation into the PPO/GRPO trust-region ratio instead of adding a separate KL. |
| `RLSD_Self_Distilled_RLVR.pdf` | GRPO with token advantage `A_t = A * clip(exp(sign(A) * Delta_t))`, where `Delta_t = log p_T(y_t) - log p_S(y_t)`. | Reward controls update sign; privileged teacher controls only magnitude. No auxiliary distillation loss. |
| `RLRT_Rebellious_Student_Reversed_Teacher.pdf` | GRPO with reversed teacher weight on correct rollouts: `w_t = exp(sign(A) * log(p_S(y_t)/p_T(y_t)))`, applied only when reward is 1. | Treats student-over-teacher disagreement on successful rollouts as exploration, not imitation. |
| `SRPO_Unifying_Group_Relative_Self_Distillation.pdf` | Routed objective: correct samples use GRPO; incorrect samples with teacher info use dynamically weighted SDPO. | Sample-level routing avoids applying self-distillation to already-correct rollouts. |
| `TRACE_Token_Routed_Self_OPD_Alignment.pdf` | GRPO plus span-routed KL: FKL on annotated key spans, optional RKL on error spans, no KL elsewhere; KL decays to zero. | Routes by token span and divergence direction, not just by sample correctness. |
| `SDAR_Self_Distilled_Agentic_RL.pdf` | `L = L_GRPO + lambda * L_SDAR`, with `L_SDAR` a gated sampled-token OPSD loss. Gate often `sigmoid(alpha * Delta_t)`. | For multi-turn agents, keeps RL primary and gates auxiliary distillation by token-level trust. |
| `Self_Distillation_Zero.pdf` | Two phases: SRT uses two NLL losses, then phase 2 minimizes KL from a frozen reviser teacher to the current generator on generator rollouts. | Converts binary reward into revision-conditioned dense supervision without external teacher demonstrations. |
| `HDPO_Hybrid_Distillation_Policy_Optimization.pdf` | `L_HDPO = L_GRPO + lambda * L_JSD`, but JSD only on cliff prompts where all standard rollouts fail and privileged rollouts pass. | Uses privileged self-distillation only when GRPO has zero gradient. Uses JSD to broaden support. |
| `REOPOLD_Scaling_Reasoning_via_Relaxed_OPD.pdf` | Reverse-KL OPD as policy gradient with reward `R_t = log(p_T(y_t)/p_S(y_t))`, then stop-gradient, reward clipping, and token masks. | Stabilizes OPD by clipping heavy negative teacher rewards and sampling informative high-entropy tokens. |
| `ExOPD_Learning_Beyond_Teacher_Reward_Extrapolation.pdf` | Generalized OPD: `max E[ alpha log(pi_teacher/pi_ref) - KL(pi_theta || pi_ref) ]`. `alpha > 1` is reward extrapolation. | Shows OPD is dense KL-constrained RL; extrapolating teacher-vs-reference reward can push beyond teacher behavior. |
| `Extrapolation_Cliff_OPD_Structured_Outputs.pdf` | ListOPD uses base-relative extrapolated advantage `A = alpha log pi_T - log pi_S - (alpha-1) log pi_B`, with IS clipping. | Studies when extrapolation becomes unstable; derives a clip-safe threshold `alpha*` for structured outputs. |
| `CREDIT_Input_Specific_Credit_OPSD.pdf` | Replaces raw self-distillation reward `log p_T(y|x,z) - log p_S(y|x)` with contrastively debiased reward subtracting teacher logits under mismatched inputs. | Removes input-generic shortcut credit; keeps input-specific feedback credit. |
| `Lightning_OPD_Offline.pdf` | Same OPD advantage `log p_T(y_t) - log p_theta(y_t)`, but rollouts and teacher logprobs are precomputed from the SFT reference policy. | Offline OPD; the key condition is teacher consistency between SFT data generation and OPD teacher. |
| `Power_Distribution_Bridges.pdf` | Power self-distillation: sample from `pi_gamma(y|x) proportional pi(y|x)^gamma`, then train by MLE / forward KL to that power distribution. | Connects power sampling, KL-regularized self-reward RL, and offline self-distillation. |

## Main Differences

### 1. Additive KL vs. Integrated RL-Distillation

`dGRPO` and `HDPO` use the most literal hybrid form:

```text
RL loss + distillation loss
```

This is easy to implement and interpret. The risk is scale mismatch: the KL/JSD
term and the reward term can pull in different directions, and the coefficient
has to balance dense token supervision against sparse reward.

`RLAD/TRRD` makes a more structural move. Instead of adding teacher KL as an
auxiliary term, it changes the GRPO importance ratio so the trust-region anchor
is a geometric mixture of old student policy and teacher policy:

```text
r_TRRD = (pi_theta / pi_old)^gamma * (pi_theta / pi_T)^(1-gamma)
```

This makes teacher guidance advantage-aware: if the advantage is near zero, the
teacher has little effect; if the advantage is positive or negative, the teacher
changes how far the policy can move.

### 1.5 Integrated RL-Distillation vs. Teacher as Credit Magnitude

These two families both combine reward and teacher information, but they put the
teacher in different parts of the update.

In **integrated RL-distillation**, the teacher changes the RL update rule itself.
For example, `RLAD/TRRD` replaces the usual GRPO ratio `pi_theta / pi_old` with a
teacher-aware ratio:

```text
r_TRRD = (pi_theta / pi_old)^gamma * (pi_theta / pi_T)^(1-gamma)
```

The teacher therefore helps define the trust-region anchor or policy-update
geometry. The student is still doing advantage-weighted RL, but the teacher
changes how the optimization step is constrained.

In **teacher-as-credit-magnitude**, the teacher does not define the target
distribution or the trust region. The verifier reward still decides whether a
trajectory is reinforced or penalized. The teacher only redistributes update
strength across tokens:

```text
A_t = A * clip(exp(sign(A) * (log p_T(y_t) - log p_S(y_t))))
```

Because the multiplier is positive, the sign of `A_t` stays the sign of the
GRPO advantage `A`. The teacher cannot flip a bad rollout into a good one, or a
good rollout into a bad one; it only says which tokens inside that rollout should
receive more or less credit.

So the shortest contrast is:

```text
Integrated RL-distillation:
  teacher changes the policy-update geometry / trust-region anchor

Teacher as credit magnitude:
  teacher rescales token advantages; reward keeps control of direction
```

### 2. Teacher as Target vs. Teacher as Credit Magnitude

Naive OPD/OPSD makes the teacher distribution a target:

```text
min D(pi_T || pi_S) or min D(pi_S || pi_T)
```

The teacher controls what the student should become. This is powerful when the
teacher is reliable and information-symmetric, but can be ill-posed when the
teacher sees privileged information unavailable at inference.

`RLSD` changes the role of the teacher. The verifier reward gives the direction:

```text
sign(A_t) = sign(A_GRPO)
```

The privileged teacher only changes the magnitude:

```text
A_t = A * clip(exp(sign(A) * (log p_T(y_t) - log p_S(y_t))))
```

So wrong rollouts are still penalized and correct rollouts are still reinforced;
the teacher only redistributes credit within the trajectory.

`RLRT` uses the same algebraic family but flips the teacher/student ratio on
correct rollouts. It asks: if the student succeeded by choosing tokens the
teacher would not have predicted, should those choices be suppressed or
amplified? RLRT amplifies them:

```text
w_RLRT = exp(sign(A) * log(p_S(y_t) / p_T(y_t)))
```

This makes it an exploration objective rather than an imitation objective.

### 3. Sample Routing vs. Token Routing

`SRPO` routes by sample:

```text
correct rollout -> GRPO
incorrect rollout with teacher info -> SDPO
otherwise -> GRPO
```

This directly addresses the observation that self-distillation on already-correct
samples can create ambiguity: there may be many correct reasoning paths, and a
self-teacher's preference among them is not necessarily reward-relevant.

`TRACE` routes by token span and divergence type:

```text
key spans on correct rollouts -> FKL
error spans on wrong rollouts -> optional RKL
non-spans -> GRPO only
KL weight -> decays to zero
```

This is more surgical. It says the problem is not only which rollout receives
distillation, but which few tokens inside the rollout deserve privileged KL at
all.

### 4. Gating Auxiliary Distillation

`SDAR` keeps the loss additive:

```text
L_total = L_GRPO + lambda * L_SDAR
```

but gates each token's auxiliary distillation by a detached sigmoid gate, usually
based on the teacher-student gap:

```text
Delta_t = log p_T(y_t | privileged context) - log p_S(y_t)
g_t = sigmoid(alpha * Delta_t)
```

The main idea is asymmetric trust. Positive gaps mean the teacher endorses a
token the student already sampled, so they are relatively trustworthy. Negative
gaps are ambiguous in multi-turn agents because retrieved skills or privileged
context may be irrelevant. The gate prevents negative teacher rejections from
overwhelming RL.

### 5. Revision-Based Self-Distillation

`SD-ZERO` is different from the KL-routing papers because it first trains a
reviser. Phase 1, SRT, uses two supervised losses:

```text
L_SRT = L_revision + L_generation
```

`L_revision` trains the model to revise an initial attempt given the reward
prompt. `L_generation` trains it to produce the improved response from the input.

Phase 2 then freezes the reviser and performs on-policy self-distillation:

```text
L_SD = E_{y ~ pi_theta(.|x)} sum_t KL(
  pi_theta(. | x, y_<t) ||
  pi_SRT(. | x, y, reward_prompt, y_<t)
)
```

So the dense teacher is not just a privileged scorer; it is a learned
outcome-conditioned reviser.

### 6. Extrapolation: Beyond Matching the Teacher

`ExOPD` shows that standard OPD can be rewritten as dense KL-constrained RL:

```text
max E[log(pi_T(y|x) / pi_ref(y|x)) - KL(pi_theta || pi_ref)]
```

It then introduces a reward scaling factor:

```text
max E[alpha * log(pi_T / pi_ref) - KL(pi_theta || pi_ref)]
```

`alpha = 1` recovers standard OPD. `0 < alpha < 1` interpolates between the
reference and teacher. `alpha > 1` extrapolates beyond the teacher in the
direction from reference to teacher.

`Extrapolation_Cliff` is the cautionary companion. In structured-output
settings, extrapolation can be beneficial just below a threshold but collapse
when the sharpened fixed point exits the clip-safe region. Its ListOPD advantage
is base-relative:

```text
A(s,a; alpha) =
  alpha log pi_T(a|s) - log pi_S(a|s) - (alpha - 1) log pi_B(a|s)
```

with a GRPO-style IS clip. The paper derives a threshold `alpha*(p,b,c)` from
teacher modal probability `p`, warm-start/base modal probability `b`, and clip
strength `c`. The main message is that reward extrapolation is not a free knob;
it has a geometry-dependent cliff.

### 7. Debiasing the Self-Distillation Reward

`CREDIT` starts from the raw OPSD token reward:

```text
r_t(y_hat) = log p_ref(y_hat | x, y_<t, z) - log p_theta(y_hat | x, y_<t)
```

It interprets this as a Bayesian filtering increment: how much the token makes
feedback `z` more predictable. But this can reward generic tokens that correlate
with feedback across many inputs.

CREDIT decomposes the teacher logprob into:

```text
log p_ref(y_hat | x, y_<t, z) = S_t(y_hat, x) + G_t(y_hat)
```

where `S_t` is input-specific and `G_t` is input-generic. It estimates `G_t` by
running the same response/feedback against contrastive, mismatched inputs, then
subtracts it:

```text
R_CREDIT(y_hat) =
  log p_ref(y_hat | x, y_<t, z)
  - alpha * E_{x'} log p_ref(y_hat | x', y_<t, z)
  - log p_theta(y_hat | x, y_<t)
```

So CREDIT changes *what the dense reward means*, rather than merely gating or
clipping it.

### 8. Offline OPD and Offline Self-Distillation

`Lightning OPD` keeps the usual OPD advantage:

```text
A_t = log p_T(y_t | prefix) - log p_theta(y_t | prefix)
```

but fixes the rollout distribution to the SFT reference policy and precomputes
teacher logprobs:

```text
online OPD:    y ~ pi_theta
Lightning OPD: y ~ pi_ref
```

This makes OPD trainable without a live teacher server. Its crucial condition is
teacher consistency: the teacher used to generate SFT data should be the same
teacher used for OPD logits. Otherwise, the offline objective inherits a
teacher-mismatch bias.

`Power Distribution Bridges` is also offline, but its target is not an external
teacher. It defines a self-reward:

```text
r_self(x,y) = log pi(y|x)
```

The KL-regularized RL optimum under that reward is the power distribution:

```text
pi_gamma(y|x) proportional pi(y|x)^gamma
```

Power self-distillation then samples from `pi_gamma` and trains by MLE / forward
KL to those samples. This amortizes expensive power sampling into the model.

## Practical Reading

The papers disagree less than they first appear to. They mostly explore different
ways to answer one question:

> Dense teacher information is useful, but how do we stop it from overriding the
> reward, copying privileged artifacts, or wasting gradient on unimportant tokens?

The broad answers are:

- **Add a teacher KL**: dGRPO, HDPO. Simple, but coefficient-sensitive.
- **Move teacher information into the RL update**: RLAD/TRRD, RLSD, RLRT.
- **Route the distillation signal**: SRPO by sample, TRACE by token span.
- **Gate the signal**: SDAR, especially for noisy multi-turn agent teachers.
- **Change the reward itself**: CREDIT removes input-generic credit.
- **Extrapolate the implicit reward**: ExOPD, with the cliff paper warning about
  clip-safe boundaries.
- **Make it offline**: Lightning OPD and Power Distribution Bridges.
- **Create a better self-teacher first**: SD-ZERO's revision phase.

My synthesis: the safest losses are the ones that keep a clean separation between
**direction** and **density**. Verifiable reward is best at saying whether a
trajectory should be reinforced or suppressed. Teacher distributions are best at
providing dense local information, but only after they are routed, gated,
debiased, clipped, or restricted to cases where RL has no useful gradient.
