# Pedagogical RL — Notes

Source: [Pedagogical RL: Teaching Models to Teach Themselves from Privileged Information](https://noahziems.com/pedagogical-rl)

## Core Claim

On-policy RL and on-policy self-distillation waste privileged information when they use it only to score sampled rollouts. If the student rarely samples successful trajectories, RL stalls before the update rule matters. Pedagogical RL instead uses privileged information to train a self-teacher that samples trajectories that are both correct and learnable for the current student.

## Mechanism

The setup uses two copies of the same base model:

- **Student** `π_S(τ | x)`: does not see privileged context.
- **Privileged self-teacher** `π_T(τ | x, c)`: sees privileged context such as final answers or execution feedback.

The teacher is trained with GRPO on a pedagogy reward:

```
r_ped(x, c, τ) = R(x, c, τ) · G_spike^{θ_S}(τ | x)
```

where `R` measures task success and `G_spike` measures whether the trajectory stays plausible under the current student. The product form matters: the teacher only receives high reward when the trajectory is both correct and teachable.

The spike-aware learnability term penalizes isolated unsupported jumps more than average NLL would. This addresses the failure case where a privileged teacher finds a correct shortcut that the student cannot realistically imitate.

After teacher RL, the student assimilates sampled teacher trajectories with surprisal-gated imitation:

```
L_assim = E_{τ ~ π_T} [ (1 / Σ_t w_t) · Σ_t w_t · -log π_S(τ_t | x, τ_<t) ]
```

with `w_t` close to 1 for tokens the student already finds plausible and close to 0 for high-surprisal teacher tokens. This prevents a few unreachable teacher moves from dominating the student update.

## Relation to OPD

Pedagogical RL reframes the central OPD bottleneck from "how should we distill a teacher distribution?" to "where do useful teacher trajectories come from?" The blog argues that purely on-policy methods sample blindly: even when the training system has privileged context, the sampler still relies on the unassisted student to stumble into successes.

This is closely aligned with our OPD position:

- **Teacher reliability is prefix-dependent**: a privileged teacher may fail to recover from corrupted student prefixes, so token-level supervision on arbitrary student states can be unreliable.
- **Learnability matters as much as correctness**: correct teacher rollouts can still be harmful if they contain high-surprisal shortcuts outside the student's support.
- **Dense signals need gating**: surprisal-gated imitation is another way to avoid wasting gradient budget on teacher-student disagreements that are not currently useful.
- **PI should shape sampling, not only scoring**: privileged information can be used to construct better mid-training trajectories rather than merely evaluate blind on-policy samples.

## Empirical Notes

The blog reports early experiments on hard MATH subsets and reasoning-intensive regression. Pedagogical RL improves sample efficiency and outperforms GRPO, on-policy self-distillation, direct off-policy distillation, and a teacher-RL ablation. The claimed gains are especially relevant when the base student's pass@1 on the training slice is very low, because standard on-policy RL receives too few positive rollouts.

## Recent OPD / OPSD-RL Works

These papers are the closest recent neighbors, especially around the question "should PI be used as a dense loss, an RL signal, or a sampler?"

| Work | Link | Connection to Pedagogical RL |
|------|------|------------------------------|
| **Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models** | [2601.18734](https://arxiv.org/abs/2601.18734) | Introduces the core OPSD procedure: one model is both student and PI-conditioned teacher, and the student minimizes per-token divergence on student rollouts. This is the main baseline Pedagogical RL argues is still sampling-blind. |
| **Privileged Information Distillation for Language Models** | [2602.04942](https://arxiv.org/abs/2602.04942) | Studies PI transfer in multi-turn agentic RL. Its `π`-Distill jointly trains a PI-conditioned teacher and unconditioned student, while its OPSD variant treats reverse-KL to the PI teacher as an RL regularizer. Very close to Pedagogical RL's teacher-student framing, but less explicitly focused on nearest-success sampling. |
| **Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents** | [2604.10674](https://arxiv.org/abs/2604.10674) | Replaces fixed PI with dynamic skills summarized from completed trajectories. This directly supports the idea that useful PI should shape future sampling, not just score current rollouts. It also uses importance-weighted reverse KL and teacher synchronization to stabilize OPSD. |
| **The Many Faces of On-Policy Distillation: Pitfalls, Mechanisms, and Fixes** | [2605.11182](https://arxiv.org/abs/2605.11182) | Important negative/diagnostic result: OPSD works best when PI encodes a shared rule, but can fail when PI is instance-specific. This sharpens the Pedagogical RL question: if PI is instance-specific, the teacher must turn it into learnable trajectories rather than a diluted consensus policy. |
| **OGLS-SD: On-Policy Self-Distillation with Outcome-Guided Logit Steering for LLM Reasoning** | [2605.12400](https://arxiv.org/abs/2605.12400) | Uses verifiable outcome rewards to contrast successful and failed on-policy trajectories, then steers teacher logits toward outcome-discriminative token guidance. This is an RL-supervised calibration layer on top of OPSD, complementary to Pedagogical RL's sampler-side teacher RL. |
| **Self-Distilled Agentic Reinforcement Learning (SDAR)** | [2605.15155](https://arxiv.org/abs/2605.15155) | Treats GRPO as the primary objective and OPSD as a gated auxiliary loss. The gating strengthens positive teacher-student gaps but attenuates negative teacher rejections, matching our broader view that dense distillation signals need routing rather than blind application. |
| **EDGE-OPD: Internalizing Privileged Context with Evidence Guided On-Policy Distillation** | [2605.23493](https://arxiv.org/abs/2605.23493) | Shows PI can change response length, style, and local token preferences, causing OPSD to learn side effects. Proposes guided rollouts plus evidence masks so updates happen only where the privileged context actually supports the sampled token. Closely related to surprisal-gated assimilation. |
| **A Predictive Law for On-Policy Self-Distillation From World Feedback** | [2605.30070](https://arxiv.org/abs/2605.30070) | Finds that the initial student/self-teacher performance gap linearly predicts final OPSD improvement across context types and model families. This could be a cheap preflight diagnostic for deciding when Pedagogical RL is worth the extra teacher-RL phase. |
| **RLCSD: Reinforcement Learning with Contrastive On-Policy Self-Distillation** | [2606.11709](https://arxiv.org/abs/2606.11709) | Identifies privilege-induced style drift: correct hints make teacher outputs shorter or more direct, so the teacher-student gap can concentrate on style tokens. It subtracts the gap induced by wrong hints from the gap induced by correct hints, isolating task-bearing signal. This is highly relevant to Pedagogical RL's claim that correctness alone is not enough. |

## Synthesis

Recent OPSD-RL work is converging on a common pattern: **dense teacher signals are useful only after they are filtered by outcome, evidence, contrast, or learnability**. Naive OPSD treats every PI-induced teacher-student disagreement as useful; the newer methods instead ask whether the disagreement is reward-relevant, evidence-supported, non-stylistic, and within the student's current support.

Pedagogical RL is distinctive because it moves the stabilization problem one step earlier. Rather than only changing the loss on student-sampled trajectories, it trains the privileged teacher to sample trajectories from an approximate "nearest-success" distribution: correct under `R`, but still plausible under `π_S`. This makes it complementary to methods like OGLS-SD, SDAR, EDGE-OPD, and RLCSD, which mostly calibrate or route the token-level distillation signal after the rollout exists.

## Open Questions

- How sensitive is `G_spike` to the student checkpoint used for scoring learnability?
- Does surprisal gating delay learning genuinely novel reasoning moves that initially look implausible?
- Can the pedagogy teacher and student be updated in a fully alternating online loop without teacher overfitting to stale student support?
- How does this compare against full-logit OPD with reward clipping or JSD targets when both are given the same privileged context?
- Can wrong-hint contrast from RLCSD be combined with the pedagogy reward to subtract PI-induced style drift before teacher RL?
- Can dynamic skill summaries from Skill-SD serve as the privileged context `c`, letting Pedagogical RL learn from reusable strategies rather than instance-specific answers?
- Can the Predictive Law student-teacher gap estimate decide when to use standard OPSD, Pedagogical RL, or plain GRPO?
- Are EDGE-OPD evidence masks and surprisal gates redundant, or do they filter different failure modes: unsupported by PI vs. unsupported by the student?
