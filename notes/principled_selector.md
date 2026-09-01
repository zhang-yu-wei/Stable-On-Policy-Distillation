# The Principled Selector: Choosing Teacher Actions by Value of Information

Design note, 2026-08-03. Companion to `signal_routing_blueprint.md` (whose
§6 controller this note derives rather than asserts) and
`principled_router_estimator_view.md` (whose error analysis supplies this
note's inputs). Written to answer two questions: *what makes a selector
principled*, and *should the selector be trained*.

## 0. The two layers, restated

The training system contains two different decisions that the word "router"
conflated:

- **Layer 1 — production (a selector, necessarily).** The teacher decides
  which supervision artifact to produce for a prompt: score logits on a
  student rollout, judge turns / verify outcomes, issue a hint and let the
  student re-roll, demonstrate or take over, or do nothing. These are
  discrete, mutually exclusive uses of teacher compute; fractional actions
  do not exist at this layer.
- **Layer 2 — consumption (an estimator).** Whatever was produced is
  assembled into the student's single gradient with continuous trust
  weights (`principled_router_estimator_view.md`).

This note is about layer 1. The claim: a principled selector is a
**value-of-information policy** — it selects the action with the highest
expected student improvement per unit teacher compute — and layer 2's error
analysis is what makes that value computable.

## 1. The value criterion

Let $J(\theta)$ be the true objective (the KL-regularized expected reward of
the estimator note, or simply held-out eval score). Producing artifact $a$
for prompt $x$ lets the trainer form a gradient contribution
$\hat g_a$; a step of size $\eta$ follows. For $L$-smooth $J$,

$$
J(\theta+\eta\hat g_a)\;\ge\;J(\theta)+\eta\,\langle\nabla J,\hat g_a\rangle-\frac{\eta^2 L}{2}\,\lVert\hat g_a\rVert^2 .
$$

Taking expectations over the sampling in $\hat g_a$ and writing
$\mathbb E[\hat g_a]=\nabla J+b_a$ (bias $b_a$) and
$\mathrm{tr}\,\mathrm{Var}(\hat g_a)=V_a$:

$$
\mathbb E[\Delta J\mid a]\;\ge\;
\eta\Big(\lVert\nabla J\rVert^2+\langle\nabla J,b_a\rangle\Big)
-\frac{\eta^2 L}{2}\Big(\lVert\nabla J+b_a\rVert^2+V_a\Big).
$$

Bias enters through alignment with the true gradient (a misaligned mean can
make the first-order term negative); variance enters through the
second-order term. The principled selector is then

$$
a^*(x)=\arg\max_{a}\;
\frac{\mathbb E\big[\Delta J\mid a,\ \text{measured student state on }x\big]}{c(a)},
$$

with $c(a)$ the action's teacher-compute cost and "do nothing" pinned at
value $0$. Everything on the right-hand side is either measured directly
($c$), estimated by the layer-2 statistics ($b_a$, $V_a$), or common to all
actions ($\lVert\nabla J\rVert$, $\eta$, $L$ — which therefore affect only
the comparison against "nothing", not the ranking among actions).

## 2. The action menu, valued

Each production action populates specific parts of the exact weight
decomposition (term (i) = teacher log-ratio, term (ii) = reward scalar; see
the estimator note §1), and its value terms are the layer-2 columns:

| Action $a$ | Use when (measured trigger) | What it yields | $b_a$ dominated by | $V_a$ dominated by | $c(a)$ |
|---|---|---|---|---|---|
| Score logits on student rollout | failed rollouts exist ($\hat p<1$) **and** the target span passes the drift certificate: teacher PPL of the prefix in the low group-relative range, top-k overlap high, ranking margin $m(x)\ge\delta$ | term (i) exactly, **and** the potential baseline for term (ii) — one forward pass yields both | teacher reward-inconsistency at the prefix (estimated by the drift certificate; ranking margin is the direct test) | ≈ 0 | one teacher pass over $T$ tokens |
| Verify / judge turns | verification: always (it is what produces $\hat p$ in the first place). Teacher turn-judging: $0<\hat p<1$ and the trajectory is long/multi-turn enough that the terminal scalar cannot localize the failure | the scalar for term (ii) | ≈ 0 | group SNR $\propto\hat p(1-\hat p)$; zero at $\hat p\in\{0,1\}$ | verifier call, or teacher judge passes |
| Hint + student re-rollout | $\hat p=0$, and a cheap hint probe (a few hinted rollouts) shows $\hat p_{\text{hint}}>0$ | a sampler under which term (ii) has signal again ($\hat p_{\text{hint}}>0$) | hint-induced distribution shift (bounded by choosing the hint with maximal top-k overlap with the student) | as above, at $\hat p_{\text{hint}}$ | hint generation + $G$ re-rollouts |
| Demonstrate / take over | $\hat p=0$ and $\hat p_{\text{hint}}\approx 0$, but the teacher itself succeeds ($\hat p_T>0$) and student surprisal on the demonstration is moderate (the content is learnable) | approximate samples from the tilted target (forward-KL estimator) | divergence direction + off-policy state term; large where student surprisal on the demo is high (unlearnable content) | acceptance-rate limited | $\propto 1/\hat p_T(x)$ teacher rollouts |
| Nothing | $\hat p=1$ (nothing to fix); or $\hat p_T=0$ even with hints (nobody can solve it — park the prompt and retry at a later checkpoint) | — | — | — | 0 |

The blueprint's §6 controller is the argmax of this table computed by hand:

- $\hat p=1$: every action's numerator is $\approx 0$ (term (ii) has no
  signal, term (i) only risks bias on correct behavior) → "nothing" wins.
- $0<\hat p<1$: verification is mandatory (cheap, unbiased, live SNR);
  logit-scoring wins wherever the certificate holds because near-zero-variance
  signal at one forward pass is the best value in the table; demonstration
  loses on cost and bias.
- $\hat p=0$: every on-policy action has a zero numerator at any cost, so
  the only positive-value actions are the ones that restore signal —
  hint first (cheaper, less bias), takeover, then demonstration.

What makes this *principled* rather than a taxonomy: each branch is now the
solution of an explicit maximization over measurable quantities, so it can
be wrong in a checkable way — if a logged action's measured payoff deviates
from the table's prediction, a specific entry (a bias estimate, a cost
model) is falsified and can be corrected.

## 3. The myopia caveat, and the standard surrogate

The criterion of §1 is greedy: it values immediate descent. Some actions
have near-zero immediate value but change what future actions are worth —
the clearest case is demonstration at $\hat p=0$, whose payoff arrives later
through the RL signal it unlocks once $\hat p$ leaves zero. The exact
problem is therefore a budgeted sequential decision problem over the
student's competence state.

The standard practical surrogate is to replace "expected eventual value"
with **measured learning progress per compute** — the change in held-out
score attributable to the action, used as a bandit reward. The automated-
curriculum literature (Graves et al. 2017; Matiisen et al. 2017; the
learning-progress signals of Oudeyer's intrinsic-motivation line) is built
on exactly this substitution and finds that immediate progress tracks
long-run value well enough to drive curricula. The escalation order at
$\hat p=0$ is the one place where the greedy rule must be manually
overridden — which the blueprint already does by construction.

## 4. Why the selector is learnable at all: stationarity

A schedule indexed by training step is not learnable — its target moves as
the student improves, so data collected early is wrong later. The selector
above has a different structure: it conditions on the **measured student
state** ($\hat p$, drift certificate, ranking margin, surprisal, teacher
pass rate), and its value terms depend on $\theta$ *only through those
statistics*. To the extent the statistics are sufficient for the value
terms, the optimal selector is a **fixed map from statistics to actions**:
"at $\hat p=0$ with a failing certificate, demonstrate" is correct at step
100 and step 10,000, for a 1B or a 30B student.

This is simultaneously (a) the reason state-gated methods beat clock
schedules across the survey, (b) the property that makes selector training
feasible, and (c) the property that makes a trained selector transferable
across runs, tasks, and model scales. The caveat: sufficiency is
approximate — effects like the capacity gap (Unmasking OPD: external-teacher
logits stop helping below a student size) depend on $\theta$ beyond the
statistics, and belong in the context features if the selector is trained.

## 5. Should the selector be trained? Four levels

Each level has a stopping criterion; escalate only when the criterion says
the current level is measurably insufficient.

**Level 1 — derived (no training).** The hand-computed argmax of §2.
*Sufficient unless* logged actions show systematic deviation between
predicted and measured payoff.

**Level 2 — calibrated (offline supervised).** From training logs, regress
measured progress-per-compute on (state features, action); replace hand-set
thresholds with the fitted decision boundary. Off-policy, cheap, no
exploration risk, and it simultaneously calibrates the layer-2 gates (both
consume the same regression). *Sufficient unless* the value landscape
shifts within a run in ways the offline fit cannot track. **This is the
recommended default.**

**Level 3 — contextual bandit (online).** Context = the state statistics;
actions = production-action types; reward = held-out learning progress per
unit compute. The obstacle is credit assignment — many selector decisions
per student update, delayed payoff. The clean instrument is **paired
randomization**: matched prompt sets routed differently (the D2Skill
paired-group construction), giving counterfactual estimates of a decision's
effect. Non-stationarity residuals are handled with discounted or
sliding-window bandits. *Worth it only if* level 2's miscalibration is
demonstrated, because exploration spends student updates on known-worse
actions.

**Level 4 — trained teacher policy (Pedagogical RL, generalized).** The
selector becomes the teacher model itself, trained by RL: action = (type,
span, hint content, demonstration content), reward = subsequent student
improvement. This is the only level that can optimize **content** rather
than type, and content is where training demonstrably pays: Pedagogical
RL's gains come from learning *which trajectories* to demonstrate
(correctness × learnability reward, with the spike-aware term penalizing
unlearnable jumps), and OpenClaw-RL's hint selection by top-k overlap is a
hand-designed policy over content that a learned one should strictly
dominate. The cost is that each action's reward requires student updates to
observe; practical surrogates are one-step proxies (decrease in student
loss on verified-correct held-out data; the spike-gated learnability
score).

**Summary judgment.** Type selection has ~5 actions and strong derived
priors — there is little room for learning to beat the derived rule, so
calibrate it (level 2) and stop. Content selection has an enormous action
space and no derived rule — there, a trained selector (level 4) is not an
optimization but the only option. Level 3 is the bridge to build only on
demonstrated need.

## 6. Reward design for any trained variant

Three requirements, each preventing a known failure:

1. **Held-out reward.** Progress measured on prompts the selector's actions
   never trained on; training-set $\hat p$ as reward reproduces
   over-imitation (lifts pass@1 while collapsing pass@k).
2. **Diversity terms in the reward.** Include pass@k or the entropy-retention
   monitor; otherwise the selector learns to spend everything on imitation
   and logit-matching, which maximize short-horizon progress at the cost of
   the exploration RL needs later.
3. **Compute normalization.** Reward per unit teacher compute, not per
   action; unnormalized, the selector learns that demonstrations and
   takeovers — the most expensive actions — are always best.

## 7. Relation to the other notes

- `signal_routing_blueprint.md` §6 = this note's level 1, computed by hand.
  The two-layer split (production selector / consumption estimator)
  resolves the apparent conflict between that controller and the
  estimator view: the teacher genuinely selects *what to produce*; the
  trainer continuously weights *what was produced*.
- `principled_router_estimator_view.md` supplies $b_a$, $V_a$ estimates and
  the probe-set calibration protocol; the level-2 regression here and the
  gate calibration there are the same fit used twice.
- `pedagogical_rl.md` is the existing instance of level 4 with action space
  restricted to trajectory content; this note's level 4 is that method with
  the action space widened to (type, span, hint, demonstration).

## References

- Graves et al., *Automated Curriculum Learning for Neural Networks* (2017)
  — learning progress as bandit reward.
- Matiisen et al., *Teacher-Student Curriculum Learning* (2017).
- Oudeyer & Kaplan, *What is intrinsic motivation?* (2007) — learning-
  progress signals.
- Pedagogical RL (`pedagogical_rl.md`) — trained content selection with a
  learnability-shaped reward.
- OpenClaw-RL (`papers/agentic_self_distillation`) — hand-designed hint
  selection by student top-k overlap, the natural target for level 4.
- D2Skill (`papers/agentic_self_distillation`) — paired baseline/intervention
  rollout groups, the counterfactual instrument for level 3.
- Unmasking OPD (`papers/failure_mode_analysis`) — capacity-gap effect that
  must enter the context features; probe methodology for calibration.
