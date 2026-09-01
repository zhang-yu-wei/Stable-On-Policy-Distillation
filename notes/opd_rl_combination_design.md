# Combining OPD and RL: The Design Space

Consolidates the combination-space analysis: where a teacher can enter a
policy-gradient method, what each entry point does to the fixed point, the
closed-form optimum and its risks, how to compute the teacher potential,
how the coupling weight is chosen and decayed in practice, and where one
shipped implementation lands in the taxonomy. Companions:
`ppo_opd_value_bridge.md` (the Combination Space section there is the
compressed version of Sections 1–5; eqs. (C1)–(C3) correspond to (1), (3),
(4) here), `soft_rl_textbook_primer.md` (the Gibbs lemma), and
`rl_opd_hybrids_survey_update_2026_07.md` (the papers cited in Section 7).

Throughout: student $\pi_\theta$, frozen teacher $\pi_T$, terminal verifier
reward $R(y)$, prefixes $s_t$, sampled tokens $y_t$. $\beta_S$ is the
*student problem's* temperature — a design choice, distinct from the
teacher's own training temperature $\beta$ (which appears only inside the
teacher-potential identity).

## 1. The three slots

There are exactly three mathematically distinct places a teacher can enter
a policy-gradient method:

| Slot | Teacher enters as | Effect on the fixed point |
|---|---|---|
| 1. Objective | KL-regularization term | changes it — teacher is in the optimum |
| 2. Estimator | potential / critic initialization | provably none — credit assignment only |
| 3. Update rule | regression target (tilt) vs. score-function surrogate | none — selects the path, not the destination |

The additive form $L_{\mathrm{RL}} + \lambda\,L_{\mathrm{OPD}}$, common in
the literature, is a blurred version of slot 1: two gradients coupled at an
arbitrary exchange rate, with no closed-form optimum and (as the survey
note's opening observes) no good equilibrium — which is why every additive
paper ends up scheduling $\lambda$ to zero (Section 7). Section 8 classifies
one shipped implementation, slime's, against these three slots.

## 2. Slot 1: the KL-regularized objective and its optimum

The principled objective-level combination is a single functional:

$$
J(\theta)=\mathbb E_{\pi_\theta}\big[R(y)\big]
-\beta_S\,\mathbb E_{\pi_\theta}\sum_t
\mathrm{KL}\big(\pi_\theta(\cdot\mid s_t)\,\Vert\,\pi_T(\cdot\mid s_t)\big).
\tag{1}
$$

**Derivation of the optimum, two steps.** First, the chain rule for KL:
expanding $\log\frac{\pi(y\mid x)}{\pi_T(y\mid x)}$ into per-token factors
and taking expectations, the per-step KL sum in (1) equals the
sequence-level divergence $\mathrm{KL}(\pi(\cdot\mid x)\Vert\pi_T(\cdot\mid x))$,
so (1) is one variational problem over completion distributions. Second,
complete the KL (Gibbs lemma, soft primer (4.2)): write
$R=\beta_S\log e^{R/\beta_S}$ and fold $\pi_T\,e^{R/\beta_S}$ into a
normalized distribution. For *every* policy,

$$
J(\pi)=\beta_S\log Z(x)-\beta_S\,\mathrm{KL}(\pi\Vert\pi^*),
\qquad
Z(x)=\mathbb E_{y\sim\pi_T}\big[e^{R(y)/\beta_S}\big],
\tag{2}
$$

so the maximum is $\beta_S\log Z(x)$, attained uniquely at

$$
\pi^*(y\mid x)\;\propto\;\pi_T(y\mid x)\,e^{R(y)/\beta_S}.
\tag{3}
$$

**Interpretations of (3).**

- *Bayesian:* $\pi^*$ is a posterior — prior $\pi_T$, likelihood
  $e^{R/\beta_S}$. For binary $R$, $\beta_S\to0$ gives the teacher
  conditioned on success (the rejection-sampling limit); $\beta_S\to\infty$
  returns the teacher unchanged. $\beta_S$ interpolates imitate
  $\leftrightarrow$ maximize.
- *Exchange rate:* $\log\pi^*-\log\pi_T=(R-\beta_S\log Z)/\beta_S$ — reward
  moves log-probability at $1/\beta_S$ nats per reward unit.
- *Projection:* (2) holds for every $\pi$, so maximizing (1) *is*
  minimizing $\mathrm{KL}(\pi\Vert\pi^*)$ — KL-regularized RL is
  distribution matching to the tilted posterior. (This is what slot 3's
  tilted regression descends directly.)
- *Value:* $\beta_S\log Z(x)$ is the prompt's soft value with the teacher
  as reference, bounded by
  $\mathbb E_{\pi_T}[R]\le\beta_S\log Z(x)\le\max_y R(y)$.

**Exact reduction to standard RL.** The gradient of (1) is ordinary policy
gradient on an augmented per-token reward:

$$
\tilde r_t=\beta_S\log\frac{\pi_T(y_t\mid s_t)}{\pi_\theta(y_t\mid s_t)}
\Big|_{\text{stop-grad}},
\qquad
\tilde r_{\text{last}}\mathrel{+}=R(y).
\tag{4}
$$

Exact, not heuristic: differentiating the KL term produces a
score-function part (captured by the sampled log-ratio in the reward) plus
a pass-through part $\mathbb E_{a\sim\pi_\theta}[\nabla\log\pi_\theta]=0$.
Feed (4) into any advantage estimator and update (GAE + clipped surrogate,
or a group baseline) and (1) is optimized exactly. The sampled log-ratio is
the unbiased $k_1$ KL estimator; $k_3$ ($\log r+1/r-1$,
$r=\pi_T/\pi_\theta$) trades small bias for lower variance and
nonnegativity. This is the mathematical core of the dense-KL-reward family
(dGRPO, KDRL).

## 3. The risks of slot 1

Both named risks are one structural fact: **(1) places the teacher in the
fixed point.**

**Bias, quantified.** In (3) the reward can only reweight *within* the
teacher's distribution — it multiplies, never adds support. Since
$e^{R/\beta_S}\le e^{\Delta R/\beta_S}$ ($\Delta R$ = reward range), any
behavior the teacher downweights by more than $\Delta R/\beta_S$ nats is
effectively vetoed: no reward evidence can lift it into the optimum. For
binary reward and $\beta_S=1$ that is a one-nat threshold. The veto is
permanent — it is in the objective, so data never corrects it. Fork
suppression is the concrete instance: the behaviors a biased teacher
downweights are exactly the exploratory ones the verifier would reward.

**Balancing, decomposed.** Three mechanisms make $\beta_S$ hard to set:

1. *Length scaling:* the KL term is a per-token sum, so total
   regularization grows with sequence length while $R$ stays $O(1)$ — one
   coefficient implies different balances for short and long rollouts.
2. *Density mismatch:* after reduction (4), interior tokens' rewards are
   only the log-ratio terms; the verifier appears once, at the end. Teacher
   agreement dominates per-token credit at most positions regardless of
   $\beta_S$; global whitening cannot fix a per-signal imbalance.
3. *Nonstationarity:* the right balance moves over training (guidance
   early, reward late); a fixed coefficient cannot track it.

The constraint form (Section 6) fixes only part of this: a KL budget is
spent globally, so bias-*correcting* deviation and reward-*seeking*
deviation compete for the same $\varepsilon$.

## 4. Slot 2: the teacher as a potential

**Definition.** A potential is a scalar function $\Phi(s)$ of the state,
used *only through its differences* $\Phi(s_{t+1})-\Phi(s_t)$ along
transitions. Because every per-step term is a difference of one function,
the sum along any complete rollout collapses to
$\Phi(\text{end})-\Phi(\text{start})$ regardless of the route — per-step
credit built this way redistributes credit within a rollout but can never
change any completion's total. (An arbitrary per-token bonus lacks this
property and does shift preferences.) The ideal potential is the true value
function; the teacher potential is the teacher's approximation of it, read
from logits by the implicit-critic identity.

**Two equivalent implementations** (Wiewiora 2003):

- *Shaping:* keep the objective, replace the reward by
  $r'_t=r_t+\Phi(s_{t+1})-\Phi(s_t)$ with $\Phi=\widehat V_T$;
- *Critic initialization:* keep the reward, initialize (or
  early-regularize) the value head at $V_\phi\approx\widehat V_T$, then run
  standard TD/GAE on real returns.

**Invariance, with proof.** For the token MDP ($\gamma=1$, $\Phi\equiv0$
at terminals) the added shaping terms telescope along any completion to
$-\Phi(x)$ — a per-prompt constant, identical for every completion of the
prompt. A per-prompt constant is a baseline: no comparison between
completions changes, no advantage changes in expectation, no optimal policy
changes (Ng–Harada–Russell). So slot 2 is teacher influence engineered to
have zero effect on the fixed point, of whatever objective it is attached
to.

**Computing the teacher potential.** Two forward passes over an existing
rollout, no gradients:

1. Roll out the student: tokens $y_0,\dots,y_{T-1}$, prefixes $s_t$.
2. Score the same tokens under teacher and reference (teacher-forced),
   keeping sampled-token log-probabilities.
3. Per-token increments
   $g_j=\beta\big[\log\pi_T(y_j\mid s_j)-\log\pi_{\mathrm{ref}}(y_j\mid s_j)\big]
   \approx V^*(s_{j+1})-V^*(s_j)$ (identity (2) of the bridge note;
   $\beta$ = the *teacher's* temperature, converting nats to reward units).
4. Anchored sums, either direction:
   forward $\widehat V_T(s_t)=c_x+\sum_{j<t}g_j$ (needs a prompt-value
   estimate $c_x$), or terminal-anchored
   $\widehat V_T(s_t)=R(y)-\sum_{j\ge t}g_j$ (needs no constant — the
   group-free form).

Simplification: **the baseline form never needs the anchor.** Under group
or batch whitening, any per-prompt constant in $\Phi$ cancels, so the
unanchored prefix sums suffice; anchored sums are needed only for absolute
values (critic-init regression targets; probability-calibrated targets via
the binary link). See the end-to-end subsection for why the increments
$g_t$ must not be added to the reward stream on top of the terminal
reward.

**Why this resolves the bias risk.** The telescoping identity means a
biased teacher can at worst *mis-time* credit within a rollout — it
structurally cannot change which completions are preferred, which is
exactly what slot-1 bias does. And the mis-timing is transient: the
potential is only the critic's initial condition; TD regression on real
returns overwrites it at the rate data arrives. It also dissolves the
balancing problem: slot 2 has no coefficient trading teacher against
reward — the handoff from teacher-credit to data-credit is carried by
measured quantities (regression weight, GAE $\lambda$ tied to explained
variance).

**Honest limits.** Immunity is asymptotic, not transient: a badly wrong
potential gives wrong early gradient directions, and undoing them costs
samples. The invariance theorem is exact for the objective's optimum;
clipping, entropy bonuses, and finite-sample whitening can leak small
deviations. The failure mode scales gracefully — a worse teacher costs
speed, never the destination.

**End-to-end usage.** For terminal-reward token MDPs, potential shaping
reduces exactly to a per-state baseline: the shaped return-to-go from
position $t$ is $R-\Phi(s_t)$ plus a per-prompt constant. So the two
usable forms are the potential as **baseline** (Monte-Carlo credit, no
critic needed) and the potential as **critic initialization** (bootstrap
credit). The natural pairing: GRPO takes the baseline form, PPO the
critic-init form.

*PPO instantiation (critic-init form).*

```
phase 0 — critic warm start, once:
  collect scored student rollouts (reward R per rollout)
  score the same tokens with π_T, π_ref (no grad):
      g_t = β·[log π_T(y_t|s_t) − log π_ref(y_t|s_t)]
  targets  V̂_T(s_t) = R(y) − Σ_{j≥t} g_j        (terminal-anchored)
  regress V_φ(s_t) → V̂_T(s_t), mixing in a fraction of plain
  Monte-Carlo targets R(y) so real outcomes are present from the start

phase 1 — stock PPO on the raw reward:
  loop:
    rollouts with π_old; terminal reward R only
    δ_t = V_φ(s_{t+1}) − V_φ(s_t)  interior;   δ_last = R − V_φ(s_last)
    GAE(λ); whiten; K minibatch epochs: clipped surrogate + value loss
  λ = 1 − EV₊·(1−λ_min): Monte-Carlo credit while the head is
    uninformative, bootstrap credit as explained variance rises
  optional (Level 2): auxiliary ‖(V_φ(s_{t+1})−V_φ(s_t)) − g_t‖² with
    weight annealed by explained variance
```

Teacher errors enter as a critic prior: they bias early bootstrap credit
and are overwritten by regression on returns — transient, lifetime set by
data.

*GRPO instantiation (baseline form).* Stock GRPO gives every token of a
rollout identical credit; the potential upgrades it to per-token credit
with no critic:

```
loop:
  per prompt x: G rollouts, rewards R_i  (groups are GRPO's own layout;
    the potential itself needs none)
  teacher passes → g_{i,t}; unanchored prefix sums
      Φ̃_{i,t} = Σ_{j<t} g_{i,j}
  per-token advantage   Â_{i,t} = R_i − Φ̃_{i,t}
  standardize Â within the group — this also absorbs the per-prompt
    anchor V*(x), which is constant across the group
  GRPO clipped-ratio update with the per-token Â_{i,t}
```

Because $\widetilde\Phi$ is a function of the prefix alone, this is a
state-dependent baseline: **unbiased for any teacher** by the baseline
lemma — teacher error costs variance only, never bias (a stronger
guarantee than the PPO form's transient prior). The credit reads as: the
outcome, compared with the teacher's forecast at this position.

*A caution common to both.* Do not additionally add $g_t$ to the reward
stream on top of the terminal $R$. For an accurate teacher the final
token's log-ratio already contains $R$ (its value estimate at the end
equals the reward), so reward-plus-increments counts the outcome twice.
Use the potential either as baseline or as critic initialization — not as
extra reward.

## 5. Slot 3: the update rule

With objective and estimator fixed, the actor update can be:

- the **clipped score-function surrogate** (PPO/GRPO style): gradient
  $\propto\hat A_t\nabla\log\pi_\theta$; zero at zero-advantage positions;
  uses sampled tokens only; or
- the **tilted regression** (advantage-tilted OPD): cross-entropy toward
  $\tilde q\propto\pi_T\,e^{\kappa\hat\Delta/\beta_S}$; full $(p-\tilde q)$
  gradient; moves unsampled tokens; bounded influence.

Both descend the same objective — by (2), maximizing (1) is minimizing
$\mathrm{KL}(\pi\Vert\pi^*)$, and the tilt is direct projection onto
$\pi^*$ while the surrogate is policy gradient toward it. The choice
changes the path, not the destination; the bridge note's Step 7 compares
the two gradient by gradient.

## 6. Choosing $\beta_S$

1. **By its meaning.** With binary reward, (3) multiplies successful
   completions' probability by $e^{1/\beta_S}$ and vetoes anything the
   teacher dislikes by more than $1/\beta_S$ nats. Decide the intended
   boost and invert: $\beta_S=1$ → $e\approx2.7\times$; $\beta_S=0.5$ →
   $e^2\approx7.4\times$. Order-one values are the sensible range; far
   outside it the objective is pure imitation or an unregularized tilt.
2. **The constraint form.** Pick a budget $\varepsilon$ (nats of per-token
   KL to the teacher) and treat $\beta_S$ as the Lagrange multiplier of
   $\max\mathbb E[R]$ s.t.
   $\overline{\mathrm{KL}}(\pi_\theta\Vert\pi_T)\le\varepsilon$, updated by
   dual ascent: raise $\beta_S$ while measured KL exceeds $\varepsilon$,
   lower it otherwise (the PPO-penalty adaptive-KL controller, pointed at
   the teacher). $\varepsilon$ is easier to reason about than $\beta_S$
   because it is measured during the run.
3. **Modulate by state, not clock** — Section 7(b); e.g. per-prompt weight
   $\propto4\hat p(1-\hat p)$.

## 7. How the literature decays the OPD weight

All from `rl_opd_hybrids_survey_update_2026_07.md`. Three families:

**(a) Clock-based schedules** (step-indexed):

| Paper | Schedule |
|---|---|
| TGPO (2605.13230) | imitation weight linear to 0 by step 200 |
| CEPO (2605.19436) | $\lambda:0.5\to0$ linearly in 25 steps |
| KDRL (2506.02208) | $\beta$ annealed $5\text{e-}3\to1\text{e-}3$ (floor, not zero) + masked off on correct rollouts |
| SDPG (2606.04036) | trapezoid $\beta(k)$: warm-up, plateau, decay to 0 |
| AMR-SD (2605.18529) | asymmetric coefficients (0.2 pos / 0.1 neg), annealed to zero |

Stated rationale: the additive objective has no good equilibrium, so OPD is
used as a warm-up schedule, not an objective. Weakness: the schedule length
is a free hyperparameter with no connection to what the student has
absorbed.

**(b) State-gated decay** — weight vanishes where a measured signal says
the teacher is no longer needed; decay over time emerges automatically:

| Paper | Gate |
|---|---|
| SSOPD (2605.17497) | per-prompt $\lambda_x=\lambda_0\cdot4\hat p(1-\hat p)$ — zero on all-correct/all-wrong groups |
| KDRL | KD masked on already-correct rollouts |
| SDPG | distill only on $A>0$ rollouts |
| SEED (2607.14777) | per-token sigmoid gate on teacher–student gap |
| EOPD (2603.07079) | hard teacher-entropy gate ($\tau=0.8$) |

**(c) Controller-based:** Direct-OPD (2607.05394) adapts its KL
coefficient by the sign of batch-mean drift — dual ascent in crude form.

**Synthesis.** Decay is what slot 1 does *instead of* invariance: annealing
$\beta_S\to0$ makes teacher influence transient by fiat — a
schedule-imposed imitation of the property slot 2 has by construction.

## 8. A shipped implementation classified: slime's OPD

slime's on-policy distillation (upstream commit `680824dd`; identical math
in the `miles` rename, `miles/backends/training_utils/loss_hub/opd.py`) is
slot 1 in the additive form. Worth recording because the surface
resemblance to Section 4's GRPO instantiation is close — both add per-token
structure to credit that GRPO otherwise leaves flat — while the fixed-point
properties are opposite.

**What it computes.** `apply_opd_kl_to_advantages`
(`slime/backends/megatron_utils/loss.py:620`) runs *after* the base
estimator has produced advantages, and modifies them in place:

$$
\hat A_t \;=\; A_t \;-\; \lambda_{\mathrm{opd}}
\big[\log\pi_\theta(y_t\mid s_t)-\log\pi_T(y_t\mid s_t)\big],
\tag{5}
$$

with $\lambda_{\mathrm{opd}}=$ `--opd-kl-coef`, $\pi_\theta$ the current
student from the no-grad forward pass, and $\pi_T$ the teacher scored on
the same sampled tokens — obtained either from an external SGLang server
during rollout (`--opd-type sglang`) or from a Megatron-resident teacher
forward (`--opd-type megatron`). The bracket is the $k_1$ estimator of the
per-token reverse KL. Under GRPO the incoming $A_t$ is the group-normalized
scalar broadcast flat across the whole response (`get_grpo_returns`,
`slime/utils/ppo_utils.py:368`), so (5) supplies the only per-token
variation in credit — which is what invites the comparison with slot 2.

**The gradient it produces.** The classification runs through the gradient,
so write the loss out. slime never touches the loss function; it modifies
the advantage tensor the loss consumes. That loss
(`slime/utils/ppo_utils.py:132`, reduced per rollout as a token mean at
`loss.py:1044`) is

$$
L(\theta)=\mathbb E\Big[\tfrac1{|y|}\textstyle\sum_t
\max\big(-r_t\hat A_t,\;-\mathrm{clip}(r_t,1-\epsilon,1+\epsilon_{\mathrm{hi}})\hat A_t\big)\Big]
-c_H H+c_{\mathrm{KL}}\mathrm{KL}(\pi_\theta\Vert\pi_{\mathrm{ref}}),
\tag{7}
$$

with $r_t=\pi_\theta(y_t\mid s_t)/\pi_{\mathrm{old}}(y_t\mid s_t)$. Run
on-policy — one gradient step per rollout batch — $r_t=1$ identically at the
point of evaluation and the clip is inactive, so substituting (5),

$$
\nabla_\theta L=-\,\mathbb E\Big[\tfrac1{|y|}\textstyle\sum_t
\big(A_t-\lambda_{\mathrm{opd}}k_t\big)\nabla_\theta\log\pi_\theta(y_t\mid s_t)\Big],
\qquad
k_t=\log\frac{\pi_\theta^{\mathrm{sg}}(y_t\mid s_t)}{\pi_T(y_t\mid s_t)},
\tag{8}
$$

where $\mathrm{sg}$ marks the value from the no-grad forward pass, a
constant inside the loss.

### 8.1 Why this is slot 1

**(a) The teacher term is the gradient of a KL.** Fix a state $s$ and
differentiate the per-state reverse KL, using $\sum_a\nabla\pi_\theta(a\mid s)=\nabla1=0$
to kill the pass-through part:

$$
\nabla_\theta\,\mathrm{KL}\big(\pi_\theta(\cdot\mid s)\Vert\pi_T(\cdot\mid s)\big)
=\sum_a\nabla\pi_\theta(a\mid s)\Big[\log\tfrac{\pi_\theta(a\mid s)}{\pi_T(a\mid s)}+1\Big]
=\mathbb E_{a\sim\pi_\theta}\big[k(s,a)\,\nabla_\theta\log\pi_\theta(a\mid s)\big].
\tag{9}
$$

The right-hand side is exactly the $\lambda_{\mathrm{opd}}k_t$ term of (8),
with the sampled token supplying the expectation. So the update ascends

$$
J_{\mathrm{slime}}(\theta)=\mathbb E_{\pi_\theta}[R]
-\lambda_{\mathrm{opd}}\,\mathbb E_{\pi_\theta}\Big[\tfrac1{|y|}\textstyle\sum_t
\mathrm{KL}\big(\pi_\theta(\cdot\mid s_t)\Vert\pi_T(\cdot\mid s_t)\big)\Big],
\tag{10}
$$

which is (1) with $\beta_S=\lambda_{\mathrm{opd}}$, up to the two deviations
below. A teacher inside the objective is the definition of slot 1. Note the
converse also holds and is worth stating: because the pass-through term
vanishes, the sampled-token value $k_t$ is a *sufficient* statistic — the
teacher's full distribution is never needed, which is what allows the
teacher to be a remote server returning only sampled-token log-probs.

*Deviation 1 — the visitation term.* Identity (9) holds the state
distribution fixed. The exact gradient of (10) also carries
$\sum_s\nabla_\theta d_\theta(s)\,\mathrm{KL}(s)$, the policy's effect on
which states are visited. slime drops it: in (8) each $k_t$ pairs only with
$\nabla\log\pi_\theta(y_t\mid s_t)$, never with the prefix, so token $t$'s
teacher disagreement never reaches token $t-1$'s credit. This is the
$\gamma=0$ choice, and it is what separates slime from eq. (4), which puts
the log-ratio into the *reward stream* ahead of the estimator so that it
enters every earlier position's return-to-go. The consequence for the fixed
point is that the per-state optimum is the tilt

$$
\pi^\star(a\mid s)\;\propto\;\pi_T(a\mid s)\,e^{A(s,a)/\lambda_{\mathrm{opd}}},
\tag{11}
$$

the local form of (3); the visitation-weighted global optimum differs from
(3) by the dropped term. What does not change is the structural fact: the
teacher sits in the fixed point, so Section 3's veto bound applies — behavior
the teacher downweights by more than $\Delta R/\lambda_{\mathrm{opd}}$ nats
cannot be lifted into the optimum by any amount of reward evidence, which at
slime's default $\lambda_{\mathrm{opd}}=1.0$ with binary reward is one nat.

*Deviation 2 — length normalization.* (7) reduces by a per-rollout token
**mean**, not a sum, so the KL contribution stays $O(1)$ against $R$ whatever
the length. Section 3's first balancing mechanism (a per-token KL sum growing
against an $O(1)$ reward) therefore does not apply to slime as written. The
density-mismatch and nonstationarity mechanisms still do.

**(b) The baseline test, which slot 2 passes and (5) fails.** Slot 2's
per-token term $\tilde\Phi_t=\sum_{j<t}g_j$ is $\sigma(s_t)$-measurable — it
is built from tokens strictly before $t$ — so it factors out of the
inner expectation and the baseline lemma applies:

$$
\mathbb E_{y_t\sim\pi_\theta(\cdot\mid s_t)}\big[\tilde\Phi_t\,\nabla_\theta\log\pi_\theta(y_t\mid s_t)\big]
=\tilde\Phi_t\,\mathbb E\big[\nabla_\theta\log\pi_\theta(y_t\mid s_t)\big]=0 .
\tag{12}
$$

Variance changes, the expected gradient does not — unbiased for *any*
teacher. slime's $k_t$ is a function of the sampled $y_t$, so it does not
factor out, and by (9) its contribution is not zero but $\nabla\mathrm{KL}$.
This is the entire difference, and it turns on *which arguments the term
takes*, not on which models produce it.

**(c) The potential test.** Potential-based shaping requires a single
function $\Phi$ of state, with $\Phi\equiv0$ at terminals, contributing
$\Phi(s_{t+1})-\Phi(s_t)$ per step (Section 4). Ask whether (5)'s term has
that form. The token MDP is a tree, so every state has a unique history and
integration along it always produces a candidate:

$$
\Phi(s_t)=\Phi(x)+\lambda_{\mathrm{opd}}\sum_{j<t}\log\frac{\pi_T(y_j\mid s_j)}{\pi_\theta(y_j\mid s_j)} .
$$

Path-independence is therefore vacuous here and cannot be the discriminator.
The terminal condition is. Imposing $\Phi(s_T)=0$ forces

$$
\Phi(x)=-\lambda_{\mathrm{opd}}\sum_{j<T}\log\frac{\pi_T(y_j\mid s_j)}{\pi_\theta(y_j\mid s_j)},
\tag{13}
$$

whose right side depends on the completion $y$ while the left must be one
number per prompt. No such $\Phi$ exists unless the total log-ratio is
identical across all completions of $x$. So (5) is not potential-based
shaping, and Section 4's invariance theorem does not reach it.

The failure is substantive, not technical. The total credit (5) adds along a
completion is $-\lambda_{\mathrm{opd}}$ times that completion's
sequence-level KL estimate, which *varies across completions* and therefore
re-ranks them: completions the teacher finds probable gain. A potential adds
the same total to every completion of a prompt, which is exactly why it
cannot re-rank. That is (b) again, stated over whole sequences instead of
single tokens.

**(d) Where the frozen-reference question does and does not enter.** Using
the bridge note's Step-2 decomposition against any frozen
$\pi_{\mathrm{ref}}$,

$$
\log\frac{\pi_T(y_t\mid s_t)}{\pi_\theta(y_t\mid s_t)}
=\underbrace{\log\frac{\pi_T(y_t\mid s_t)}{\pi_{\mathrm{ref}}(y_t\mid s_t)}}_{g_t/\beta}
+\underbrace{\log\frac{\pi_{\mathrm{ref}}(y_t\mid s_t)}{\pi_\theta(y_t\mid s_t)}}_{\text{drift}} .
\tag{14}
$$

Within one iteration $k_t$ is a frozen constant, and in on-policy
single-step operation $\pi_\theta$ *is* the sampling policy — so (8) is a
faithful implementation of (10), not a broken implementation of slot 2. At
initialization with $\pi_{\mathrm{ref}}=\pi_{\theta_0}$ the drift term
vanishes and slime's per-token term coincides numerically with the potential
increment $g_t/\beta$. The coincidence ends immediately: slot 2 freezes the
reference *across* iterations, slime re-reads it every iteration, and the
accumulated drift is what Step 2 identifies as the non-conservative
component.

Two things are independent of all this. First, tests (b) and (c) already
fail at $\theta=\theta_0$, where the drift is exactly zero — they concern
aggregation and the terminal condition, not the denominator. Second, the
sign: substituting $\pi_\theta=\pi_{\mathrm{ref}}$ into (5) gives
$\hat A_t=A_t+(\lambda_{\mathrm{opd}}/\beta)\,g_t$, the increment *added* to
an advantage that already contains the terminal reward, where slot 2
*subtracts* the accumulated $\tilde\Phi_t$ from the return. Adding $g_t$ on
top of $R$ is the one configuration Section 4's closing caution rules out:
for an accurate teacher the final increment already contains $R$, so the
outcome is counted twice.

### 8.2 Summary of the differences

| | slot 2 (Section 4) | slime, eq. (5) |
|---|---|---|
| argument of the term | the prefix $s_t$ alone | the sampled token $y_t$ |
| aggregation | prefix sum $\tilde\Phi_t=\sum_{j<t}g_j$ | current token only |
| combination | subtracted from the return $R$ | added to the finished advantage |
| model pair | $\pi_T$ vs $\pi_{\mathrm{ref}}$, frozen across iterations | $\pi_T$ vs $\pi_\theta$, re-read each iteration |
| expected gradient contribution | zero, by (12) | $\nabla\mathrm{KL}$, by (9) |
| fixed point | unchanged | tilted, eq. (11) |

Rows one and two are decisive and hold unconditionally; row four only
compounds them over training.

### 8.3 Two implementation details with theoretical weight

1. *Whitening order.* (5) is applied before `normalize_advantages`
   (`loss.py:778`). With whitening on, the RL part and the KL part are
   divided by the same batch standard deviation, so $\lambda_{\mathrm{opd}}$
   is not a stable exchange rate between them — it drifts with batch
   statistics. This is the concrete form of Section 3's observation that
   global whitening cannot repair a per-signal imbalance.
2. *No schedule.* `opd_kl_coef` is a plain float
   (`slime/utils/arguments.py:1130`), read at one site and never written;
   the repository contains no annealing, gating, or dual-ascent controller
   for it, and the argument validation does not bound it. Measured against
   Section 7, slime implements the additive slot-1 objective without the
   decay that every additive method in the survey found necessary — which
   is the configuration Section 9's diagnostic principle argues against.

A separate slot-1 channel exists alongside it: `--kl-coef` against the
reference model (`loss.py:700-715`), which *is* folded into the reward
stream before the estimator on the PPO and REINFORCE++ paths. So the
machinery for the eq. (4) form is present in the codebase; OPD does not
use it.

### 8.4 What migration to slot 2 would require

The target, from Section 4's GRPO instantiation:

$$
g_{i,t}=\beta\big[\log\pi_T(y_{i,t}\mid s_{i,t})-\log\pi_{\mathrm{ref}}(y_{i,t}\mid s_{i,t})\big],
\qquad
\hat A_{i,t}=R_i-\sum_{j<t}g_{i,j},
\tag{15}
$$

then standardize within the group. Three substantive changes: swap
$\pi_\theta$ for a frozen $\pi_{\mathrm{ref}}$, accumulate prefix sums, and
subtract from the return rather than add to the advantage. Also required:
the real verifier reward must be restored, since the shipped sglang path
returns $0.0$ for every sample (`on_policy_distillation.py:65`) on the
premise that the KL carries the whole signal — slot 2 keeps $R$ and changes
only credit assignment. And `--use-opd` must be off, by (d)'s
double-counting argument.

Two placement constraints decide where the computation goes, and both point
away from the training step:

- *The prefix sum against context parallelism.* Under CP each rank holds two
  non-contiguous chunks of the sequence (`cp_utils.py:9-44`), so an
  exclusive scan along the response needs a two-phase scan across the CP
  group. Computing $\tilde\Phi$ on the whole sequence at rollout side avoids
  this: per-token tensors arriving from rollout are already sharded
  correctly by `slice_log_prob_with_cp` (`actor.py:261-276`).
- *Group standardization.* DP partitioning can split a group across ranks,
  which is why slime normalizes rewards at rollout side
  (`ray/rollout.py:664-687`) rather than in the loss. At rollout
  post-process time the group is intact.

So the natural implementation computes $g$, $\tilde\Phi$ and the
standardized $\hat A$ in a `--custom-reward-post-process-path` function,
ships the finished per-token advantage through a new `Sample` field
(five one-line plumbing edits mirroring `teacher_log_probs`), and reads it
in a `--custom-advantage-function-path` function that replaces the estimator
branch at `loss.py:717-720`. Nothing in the loss changes. The second
scoring pass against $\pi_{\mathrm{ref}}$ can reuse the multi-model serving
already present (`get_model_url(args, "ref", "/generate")`,
`sglang_rollout.py:65-81`). The Megatron-resident alternative needs no extra
served model but must implement the CP scan, and additionally patch
`placement_group.py:178`, where ref residency is currently gated on
`kl_coef != 0 or use_kl_loss`.

Two choices the code cannot make: $\beta$, the teacher's temperature,
setting the scale of $\tilde\Phi$ against $R$; and which reference — the
teacher's own pre-RL checkpoint makes the log-ratio the teacher's RL delta,
while the student init is the documented fallback that records
teacher-vs-init capability gaps as value.

## 9. Recommendation

- Prefer slots 2 (+3) as the primary channels: invariant, transient under
  data, nothing to balance. This is the bridge note's design.
- If slot 1 is used (a stock PPO/GRPO actor with KL-to-teacher reward):
  hold per-token KL at a budget $\varepsilon$ by dual ascent rather than
  fixing $\beta_S$; modulate per prompt by $4\hat p(1-\hat p)$; let the
  floor be zero only if the teacher must eventually be outgrown.
- Diagnostic principle: if you find yourself engineering elaborate decay
  schedules to neutralize slot-1 bias, that is evidence for moving the
  teacher to slot 2, where no schedule is needed.

## References

- Ng, Harada & Russell, *Policy Invariance Under Reward Transformations*
  (ICML 1999) — potential-based shaping.
- Wiewiora, *Potential-Based Shaping and Q-Value Initialization Are
  Equivalent* (JAIR 2003).
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017) — the
  adaptive-KL controller reused in Section 6.
- Survey entries: KDRL 2506.02208; TGPO 2605.13230; SDPG 2606.04036; SEED
  2607.14777; SSOPD 2605.17497; EOPD 2603.07079; CEPO 2605.19436; AMR-SD
  2605.18529; Direct-OPD 2607.05394 — details in
  `rl_opd_hybrids_survey_update_2026_07.md`.
