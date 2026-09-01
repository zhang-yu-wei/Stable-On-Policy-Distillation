# Why On-Policy Distillation Fails—and How It Is Usually Fixed

## Short answer

On-policy distillation (OPD) samples trajectories from a student and asks a
teacher for next-token probabilities at the prefixes the student visits. It is
attractive because the feedback is dense and is evaluated on the student's own
states. Neither property guarantees that the feedback is useful.

Most reported failures come from one of four layers:

1. **The estimator is fragile.** A sampled token is a noisy summary of a full
   distribution, while a naively truncated top-$K$ loss is biased.
2. **The teacher signal is not the task objective.** A teacher scores locally
   plausible next tokens, not the downstream success of the completed
   trajectory, and may be unreliable on student-drifted prefixes.
3. **The data distribution moves with the model.** Small errors can alter later
   prefixes, and repetition or length pathologies can feed back into subsequent
   training batches.
4. **The transfer problem may be ill-posed.** The teacher may be too different,
   add no new capability, rely on unavailable privileged information, use a
   different tokenizer, or transfer confidence that is invalid at deployment.

The common practical response is therefore not one special loss. Stable systems
combine a sound distribution-matching estimator, a conservative target, clean
rollouts, a compatible teacher, and outcome-based checks.

## 1. What “OPD” means matters

For a student $\pi_\theta$, teacher $q$, and a student-generated prefix $s_t$,
the canonical local objective matches

$$
\pi_\theta(\cdot\mid s_t)
\quad\text{to}\quad
q(\cdot\mid s_t).
$$

In practice, “OPD” can refer to three materially different estimators:

| Form | Signal at each prefix | Main trade-off |
|---|---|---|
| Full-vocabulary KL | All vocabulary logits | Dense and stable, but expensive |
| Sampled-token OPD | The one token drawn by the student | Cheap and unbiased as a local policy-gradient estimator, but high variance |
| Top-$K$ OPD | A selected subset of tokens | Cheaper than full logits, but selection and normalization can introduce bias |

This distinction explains many apparently contradictory results. Full-vocabulary
matching avoids one-token signal imbalance and discrete support selection, but it
does **not** fix a bad teacher, a poor loss direction, or degenerate on-policy
rollouts. Conversely, a variance baseline helps sampled-token OPD but contributes
nothing to an exact full-vocabulary sum at a fixed prefix.

Practical token-local objectives also omit some future-state coupling from the
full sequence objective. That introduces bias, but retaining all future coupling
can make gradient variance grow badly with horizon. The usual solution is to
keep local supervision and improve its quality rather than blindly switching to
a high-variance sequence estimator.

## 2. Failure modes, causes, and common repairs

### 2.1 Sampled-token signal imbalance and variance

**Symptom.** Training is seed-sensitive, a few tokens dominate updates, loss
looks noisy, and performance can collapse on long responses. In sampled-token
OPD, most sampled tokens may receive negative teacher-student log-ratio rewards,
while a small set of positive outliers determines much of the update.

**Why it happens.** One draw discards nearly all of the teacher distribution.
The log-ratio can also be heavy-tailed, so rare student/teacher disagreements
produce disproportionate gradients.

**Usual fixes.** Use full-vocabulary KL when affordable. Otherwise use a
properly constructed teacher top-$K$ objective, more than one local sample, or a
control-variate baseline such as vOPD. Clip extreme log-ratio rewards when the
tail, rather than genuine corrective signal, dominates. These fixes reduce
estimator noise; they do not make an unreliable teacher reliable.

### 2.2 Biased top-$K$ reverse KL

**Symptom.** A seemingly reasonable top-$K$ approximation suppresses mildly
teacher-preferred tokens, drifts toward unstable continuations, or collapses
even when full-vocabulary reverse KL is stable.

**Why it happens.** For full-vocabulary reverse KL, the constant $+1$ in the
gradient cancels because probabilities sum to one. If an unnormalized subset is
used, it no longer cancels. A token is then promoted only when
$q(v)>e\,\pi_\theta(v)$ rather than when $q(v)>\pi_\theta(v)$.

**Usual fixes.** Renormalize both teacher and student within the chosen support,
or use a stop-gradient top-$K$ surrogate that removes the offending term.
Teacher-defined support is often more stable than student-defined support in
multi-task training because it does not move with every student update. Note
that renormalized top-$K$ matches relative probabilities inside the set but
ignores how much total probability mass belongs to it; full-vocabulary KL
remains the clean reference when feasible.

### 2.3 Teacher guidance becomes wrong on student prefixes

**Symptom.** The teacher is strong on its own generations but gives weak or
counterproductive supervision after the student makes an error. Distillation
may improve incorrect rollouts yet damage already-correct ones.

**Why it happens.** OPD deliberately queries the teacher off the teacher's
usual trajectory distribution. A language-model teacher can continue a broken
prefix fluently without repairing it. More fundamentally, next-token likelihood
is only a proxy for task value: teacher-student disagreements mix
reasoning-critical corrections with harmless choices of wording or notation.
Token-level diagnostic work finds the teacher signal much better aligned with
eventual success on incorrect rollouts than on correct ones, where it is often
noise.

**Usual fixes.** Route or weight supervision by outcome, teacher perplexity,
entropy, or another reliability signal. Distill more strongly on failed or
uncertain trajectories and reduce updates on already-mastered ones. Adapt the
teacher with task reward, calibrate its token margins using outcome order, or
combine OPD with verifier-based RL so that global correctness can veto locally
plausible advice. There is no teacher/context choice that is best for every
student and task, so small diagnostic runs are preferable to assuming that the
largest teacher wins.

### 2.4 Distribution gap and the direction of KL

**Symptom.** Forward KL produces extreme gradients on tokens the student barely
supports; reverse KL reduces entropy, diversity, and pass@$k$, particularly at
high-entropy teacher states.

**Why it happens.** Forward KL is mode-covering and tries to recover all teacher
modes, including ones far outside the student's current support. Reverse KL is
mode-seeking and can concentrate on one convenient mode. Both become more
pathological as the teacher-student gap grows.

**Usual fixes.** Construct an intermediate target rather than forcing the raw
teacher distribution immediately. Common choices include geometric bridges
(Veto), arithmetic teacher-student mixtures/Jensen-Shannon objectives, or an
entropy-aware forward/reverse-KL mixture. An off-policy SFT cold start can first
move the student into a compatible region; the bridge can then be strengthened
gradually. Preserve an explicit entropy or diversity diagnostic, because
accuracy alone can hide reverse-KL collapse.

### 2.5 Repetition, length inflation, and truncation collapse

**Symptom.** Responses suddenly become longer, repetitive rollouts saturate the
batch, most examples hit the token limit, and validation accuracy drops sharply.

**Why it happens.** This is an on-policy feedback loop. Repetitive tokens can
receive larger reverse-KL advantages; as they occur more often, they occupy more
of the gradient; the updated student then generates even more of them. Once
truncated trajectories dominate, missing endings and abnormal prefixes bias the
next update and make the loop self-reinforcing.

**Usual fixes.** Combine a divergence constraint to the initial/reference
student with a mixture of clean, complete off-policy trajectories. SFT anchoring
for output format and length serves a similar purpose. Top-$p$ rollout sampling,
special-token masking, log-ratio clipping, and repetition-aware filtering are
useful guardrails, but merely raising the maximum generation length hides the
symptom and permits the feedback loop to continue. Monitor response length,
truncation rate, repeated $n$-grams, and entropy throughout training, not only
task reward.

### 2.6 Teacher-student incompatibility or no transferable novelty

**Symptom.** Loss decreases but capability does not improve, or a stronger
teacher performs worse than a smaller, same-family teacher.

**Why it happens.** Successful OPD mostly adjusts high-probability tokens at
states the student already visits. If teacher and student use incompatible
reasoning patterns, useful teacher modes may be unreachable. At the other
extreme, a same-family teacher can look almost identical from the student's
local perspective and provide no new signal. Raw teacher capability and
*distillability* are therefore different properties.

**Usual fixes.** Measure support overlap and downstream gain before a full run;
choose teacher-aligned prompts; use an off-policy cold start; adapt the teacher
to the target task with RLVR; or calibrate the teacher specifically for
distillability. Balance prompt and rollout data so the student visits informative
states. If these interventions do not create both compatibility and novelty,
change the teacher or training data rather than tuning the OPD coefficient.

### 2.7 Privileged-information failure in self-distillation

**Symptom.** On-policy self-distillation (OPSD) works for internalizing a shared
system prompt or preference but fails when the teacher sees a problem-specific
answer or solution that the student will not see at test time.

**Why it happens.** The deployed student must learn one policy without the
privileged information (PI). Under reverse KL, its optimum is a normalized
geometric mean of the PI-conditioned teacher policies—a consensus. Shared rules
create consistent teacher behavior and survive this aggregation; instance-
specific PI can induce incompatible solution paths whose useful details do not
form a reusable consensus.

**Usual fixes.** Use OPSD to compress shared, generalizable rules, formats, and
preferences. For instance-specific answers, prefer outcome RL, carefully
constructed off-policy demonstrations, or hindsight rewriting that turns the PI
into a learnable trajectory. More PI is not automatically better; its structure
must match what will be available or inferable at deployment.

### 2.8 Vocabulary and special-token mismatch

**Symptom.** Token-level rewards are nonsensical near EOS or chat boundaries,
or cross-family distillation is much worse than expected.

**Why it happens.** Token probabilities are comparable only when they refer to
the same events. Different tokenizers segment text differently, and even shared
vocabularies may assign incompatible semantics to chat, padding, or termination
tokens. A one-token comparison amplifies these mismatches.

**Usual fixes.** Unify or explicitly align vocabularies, compare probabilities
over aligned text spans, and mask incompatible special tokens. Verify chat
templates, BOS/EOS handling, padding masks, and truncation behavior before
attributing the result to the OPD algorithm.

### 2.9 Capability improves while calibration worsens

**Symptom.** Accuracy rises but the student becomes systematically
overconfident.

**Why it happens.** A teacher conditioned on a solution or other PI has more
information than the deployed student. Its conditional confidence is therefore
an optimistic target for the PI-free student, and distilling it can collapse
entropy without producing valid deployment-time uncertainty.

**Usual fixes.** Treat confidence as a separate target. Calibration-aware OPD
estimates empirical success from student rollouts and distills that
student-grounded confidence instead of copying the teacher's self-reported
certainty. Evaluate calibration and selective accuracy separately from task
accuracy.

## 3. A robust default recipe

There is no universal configuration, but the following is a defensible starting
point:

1. **Validate the transfer pair.** Check that the teacher improves outcomes on
   the target prompts, is comprehensible to the student, and contributes more
   than stylistic disagreement.
2. **Prefer full-vocabulary local KL when compute permits.** If using top-$K$,
   use teacher support with correct renormalization or a justified
   stop-gradient loss. If using one-token OPD, add variance reduction.
3. **Use a conservative target.** Begin with a teacher-student bridge or an
   entropy-aware KL mixture; constrain drift from the initial student.
4. **Keep clean data in the loop.** Mix complete off-policy/SFT trajectories
   with student rollouts, especially for long reasoning and agentic tasks.
5. **Ground the teacher in outcomes.** Gate, route, or reweight OPD using
   correctness and uncertainty; combine it with RL or verifier feedback when
   task success is the real objective.
6. **Monitor failure precursors.** Track log-ratio tails, entropy, support mass,
   length, truncation, repetition, correct/incorrect rollout quality, task
   reward, and calibration. Stop or reduce OPD before a rollout pathology
   dominates the next data-collection round.

The key rule is to match the fix to the layer. Full logits fix sparse support,
not bad semantics. A better teacher fixes semantic quality, not a biased top-$K$
gradient. Reference KL controls drift, but does not make instance-specific PI
transferable.

## 4. Source map

- Sampled-token imbalance, prefix drift, and special-token handling:
  [Revisiting OPD](../papers/failure_mode_analysis/Revisiting_OPD_Empirical_Failure_Modes_and_Simple_Fixes.pdf)
- Repetition, length inflation, truncation collapse, and rollout mixtures:
  [Demystifying OPD](../papers/failure_mode_analysis/Demystifying_OPD_Length_Inflation_and_Stabilization.pdf)
- Compatibility, novelty, cold starts, and prompt selection:
  [Rethinking OPD](../papers/failure_mode_analysis/Rethinking_OPD_Phenomenology_Mechanism_and_Recipe.pdf)
- Top-$K$ bias, teacher adaptation, SFT stabilization, and PI consensus:
  [The Many Faces of OPD](../papers/failure_mode_analysis/The_Many_Faces_of_On_Policy_Distillation.pdf)
- Per-token outcome alignment and the absence of a universal teacher/context:
  [Unmasking OPD](../papers/failure_mode_analysis/Unmasking_On_Policy_Distillation.pdf)
- Sampled-token variance reduction:
  [KL for a KL / vOPD](../papers/failure_mode_analysis/KL_for_KL_Control_Variate_Baseline.pdf)
- Teacher-student gaps and distillability calibration:
  [Distillation Traps and Guards](../papers/failure_mode_analysis/Distillation_Traps_and_Guards.pdf)
- Confidence miscalibration under privileged context:
  [The Illusion of Certainty](../papers/failure_mode_analysis/Illusion_of_Certainty_Calibration_OPD.pdf)
- Intermediate targets and KL-direction pathologies:
  [Veto](../papers/stabilization_and_target_shaping/Stable_OPD_Adaptive_Target_Reformulation_Veto.pdf) and
  [Entropy-Aware OPD](../papers/stabilization_and_target_shaping/Entropy_Aware_On_Policy_Distillation.pdf)
- Outcome-based routing and teacher-signal calibration:
  [SCOPE](../papers/stabilization_and_target_shaping/SCOPE_Signal_Calibrated_OPD.pdf) and
  [Uni-OPD](../papers/stabilization_and_target_shaping/Uni_OPD_Dual_Perspective_Recipe.pdf)
- Cross-tokenizer alignment:
  [TokAlign++](../papers/stabilization_and_target_shaping/TokAlign_Plus_Plus_Vocabulary_Adaptation.pdf)
