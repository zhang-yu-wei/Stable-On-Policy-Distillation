# SFT, RL, and On-Policy Distillation: A Unified View

This note treats supervised fine-tuning (SFT), reinforcement learning (RL), and
on-policy distillation (OPD) in their most general forms, ignoring specific
algorithms. GRPO, PPO, and REINFORCE are estimators of one RL gradient; forward
KL, reverse KL, and JSD variants of distillation are instances of one
divergence-minimization problem. The claims of the note:

1. All three methods minimize a divergence between the policy $\pi_\theta$ and
   a **target distribution** $\pi^*$. They differ in what $\pi^*$ is and in how
   the learner can access it.
2. All three gradients have the same algebraic form: a weighted sum of
   score functions $\nabla_\theta\log\pi_\theta(y_t\mid s_t)$. They differ in
   which distribution the trajectories come from and in the weight.
3. The differences that remain after this unification are exactly two:
   **whose distribution generates the training states**, and **how the target
   is specified** (samples, token-level density, or reward evaluations). Every
   practical property—distribution shift, sample efficiency, the teacher
   ceiling, credit assignment—follows from these two axes.

## 0. Setup and notation

A prompt $x\sim p(x)$; a response $y=(y_1,\ldots,y_T)$ with tokens from a
vocabulary $\mathcal V$, terminated by an EOS token so that sequence length is
part of $y$. Write the prefix (state) as

$$
s_t=(x,y_{<t}),\qquad
\pi_\theta(y\mid x)=\prod_{t=1}^{T}\pi_\theta(y_t\mid s_t).
$$

This is a token-level MDP with deterministic transitions (append the sampled
token to the prefix), which is why RL formalism applies to all three methods.

Divergences, for distributions $p,q$ over any space:

$$
\operatorname{KL}(p\Vert q)=\sum_z p(z)\log\frac{p(z)}{q(z)}.
$$

"Forward KL to a target $\pi^*$" means $\operatorname{KL}(\pi^*\Vert\pi_\theta)$;
"reverse KL" means $\operatorname{KL}(\pi_\theta\Vert\pi^*)$.

Throughout, $\mathbb E_x$ averages over prompts and is often left implicit.

## 1. The three objectives in their most general form

### 1.1 SFT: forward KL to a distribution you can only sample

Given a data distribution $p_{\mathrm{data}}(y\mid x)$—human demonstrations,
or a teacher's samples (off-policy sequence-level distillation is SFT with
$p_{\mathrm{data}}=\pi_T$)—SFT minimizes the negative log-likelihood:

$$
L_{\mathrm{SFT}}(\theta)
=-\,\mathbb E_{x,\,y\sim p_{\mathrm{data}}(\cdot\mid x)}
  \big[\log\pi_\theta(y\mid x)\big].
$$

**Claim.** $L_{\mathrm{SFT}}$ equals the forward KL up to a constant:

$$
\mathbb E_x\operatorname{KL}\!\big(p_{\mathrm{data}}(\cdot\mid x)\,\Vert\,\pi_\theta(\cdot\mid x)\big)
=L_{\mathrm{SFT}}(\theta)
-\mathbb E_x\,H\big(p_{\mathrm{data}}(\cdot\mid x)\big).
$$

*Derivation.* Expand the KL:

$$
\operatorname{KL}(p_{\mathrm{data}}\Vert\pi_\theta)
=\sum_y p_{\mathrm{data}}(y\mid x)\log p_{\mathrm{data}}(y\mid x)
-\sum_y p_{\mathrm{data}}(y\mid x)\log\pi_\theta(y\mid x).
$$

The first term is $-H(p_{\mathrm{data}}(\cdot\mid x))$ and does not depend on
$\theta$; the second is the per-prompt SFT loss. $\square$

So the fundamental form of SFT is: **minimize forward KL to a target
distribution, given only samples from that target.** This access constraint is
essential. The forward KL is an expectation under $\pi^*=p_{\mathrm{data}}$, so
a sample average over the dataset estimates it without ever evaluating
$p_{\mathrm{data}}$'s density—which for human data does not exist in evaluable
form.

### 1.2 RL: expected reward, usually with a KL penalty

Given a reward function $R(x,y)$ defined on complete responses (per-token
rewards $r_t$ are a special case with $R=\sum_t r_t$), the general RL objective
with reference policy $\pi_{\mathrm{ref}}$ and coefficient $\beta\ge 0$ is

$$
J_\beta(\theta)
=\mathbb E_x\,\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\big[R(x,y)\big]
-\beta\,\mathbb E_x\operatorname{KL}\!\big(\pi_\theta(\cdot\mid x)\,\Vert\,\pi_{\mathrm{ref}}(\cdot\mid x)\big).
$$

$\beta=0$ is pure reward maximization. REINFORCE, PPO, and GRPO are gradient
estimators for $J_\beta$ with different variance-reduction and trust-region
machinery; the objective itself is the fundamental object. (See
`ppo_trpo_gae_textbook_primer.md` for the estimators and
`soft_rl_textbook_primer.md` §6 for the KL-regularized problem.)

### 1.3 OPD: per-token divergence to a teacher, on student states

Given a teacher $\pi_T$ whose token-level log-probabilities can be evaluated at
arbitrary prefixes, OPD samples from the **student** and matches the teacher at
the student's own prefixes:

$$
L_{\mathrm{OPD}}(\theta)
=\mathbb E_x\,\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}
\left[\sum_{t=1}^{T}
\operatorname{KL}\!\big(\pi_\theta(\cdot\mid s_t)\,\Vert\,\pi_T(\cdot\mid s_t)\big)\right].
$$

The general form allows any divergence $D$ in place of the reverse KL; the
reverse-KL case is canonical and is the one whose structure this note analyzes.

**Claim (chain rule).** The token-level loss equals the sequence-level reverse
KL between the full response distributions:

$$
\operatorname{KL}\!\big(\pi_\theta(\cdot\mid x)\,\Vert\,\pi_T(\cdot\mid x)\big)
=\mathbb E_{y\sim\pi_\theta}
\left[\sum_{t=1}^{T}
\operatorname{KL}\!\big(\pi_\theta(\cdot\mid s_t)\,\Vert\,\pi_T(\cdot\mid s_t)\big)\right].
$$

*Derivation.* By the autoregressive factorization,

$$
\operatorname{KL}(\pi_\theta\Vert\pi_T)
=\mathbb E_{y\sim\pi_\theta}\!\left[\log\frac{\pi_\theta(y\mid x)}{\pi_T(y\mid x)}\right]
=\mathbb E_{y\sim\pi_\theta}\sum_{t}
\log\frac{\pi_\theta(y_t\mid s_t)}{\pi_T(y_t\mid s_t)}.
$$

Take the $t$-th term and condition on the prefix $s_t$ (whose distribution is
induced by $\pi_\theta$):

$$
\mathbb E_{y\sim\pi_\theta}\!\left[\log\frac{\pi_\theta(y_t\mid s_t)}{\pi_T(y_t\mid s_t)}\right]
=\mathbb E_{s_t}\,\mathbb E_{y_t\sim\pi_\theta(\cdot\mid s_t)}
\!\left[\log\frac{\pi_\theta(y_t\mid s_t)}{\pi_T(y_t\mid s_t)}\right]
=\mathbb E_{s_t}
\operatorname{KL}\!\big(\pi_\theta(\cdot\mid s_t)\Vert\pi_T(\cdot\mid s_t)\big).
$$

Sum over $t$. $\square$

So sequence-level and token-level reverse-KL OPD are the same objective. The
token-level form additionally reveals that the loss supervises the **full
next-token distribution at every student-visited prefix**, which is where the
density of the signal comes from.

## 2. First unification: one problem, three targets

### 2.1 KL-regularized RL is reverse KL to a tilted distribution

Define the per-prompt **Gibbs (reward-tilted) distribution**

$$
\pi^*_\beta(y\mid x)
=\frac{\pi_{\mathrm{ref}}(y\mid x)\,e^{R(x,y)/\beta}}{Z_\beta(x)},
\qquad
Z_\beta(x)=\sum_y \pi_{\mathrm{ref}}(y\mid x)\,e^{R(x,y)/\beta}.
$$

**Theorem.**

$$
J_\beta(\theta)
=-\beta\,\mathbb E_x\operatorname{KL}\!\big(\pi_\theta(\cdot\mid x)\,\Vert\,\pi^*_\beta(\cdot\mid x)\big)
+\beta\,\mathbb E_x\log Z_\beta(x).
$$

Since $\log Z_\beta(x)$ does not depend on $\theta$, maximizing $J_\beta$ is
exactly minimizing the reverse KL to $\pi^*_\beta$, and the unconstrained
optimum is $\pi_\theta=\pi^*_\beta$.

*Derivation.* Fix $x$ and expand:

$$
\operatorname{KL}(\pi_\theta\Vert\pi^*_\beta)
=\mathbb E_{y\sim\pi_\theta}\!\left[
\log\pi_\theta-\log\pi_{\mathrm{ref}}-\frac{R}{\beta}+\log Z_\beta
\right]
=\operatorname{KL}(\pi_\theta\Vert\pi_{\mathrm{ref}})
-\frac{1}{\beta}\,\mathbb E_{\pi_\theta}[R]
+\log Z_\beta .
$$

Multiply by $-\beta$ and recognize the first two terms as the per-prompt
$J_\beta$. $\square$

As $\beta\to 0$, $\pi^*_\beta$ concentrates on
$\arg\max_y R(x,y)$ within the support of $\pi_{\mathrm{ref}}$: pure RL is the
sharp limit of the same divergence-minimization problem.

### 2.2 OPD is KL-regularized RL with a log-ratio reward

Set $R(x,y)=\beta\log\dfrac{\pi_T(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}$ in the
Gibbs formula:

$$
\pi^*_\beta(y\mid x)
\propto \pi_{\mathrm{ref}}\cdot\frac{\pi_T}{\pi_{\mathrm{ref}}}
=\pi_T(y\mid x),
$$

with $Z_\beta(x)=1$. So OPD is exactly the KL-regularized RL problem whose
tilted target happens to equal an existing model. (This identity is the
starting point of ExOPD; see `opd_rl_connection_and_reward_losses.md`, Step 8.)
Conversely, KL-regularized RL is on-policy distillation from a teacher that
does not exist as a network—$\pi^*_\beta$—but whose **unnormalized**
log-density $\log\pi_{\mathrm{ref}}+R/\beta$ can be evaluated at any sampled
response.

### 2.3 The access mode forces the divergence direction

All three methods now read: minimize a KL between $\pi_\theta$ and a target
$\pi^*$. The direction of that KL is not a free design choice; it is determined
by how $\pi^*$ can be accessed:

- **Forward KL** $\operatorname{KL}(\pi^*\Vert\pi_\theta)$ is an expectation
  under $\pi^*$. Estimating it requires **samples from $\pi^*$** and never
  requires $\pi^*$'s density. This is the only option for human data.
- **Reverse KL** $\operatorname{KL}(\pi_\theta\Vert\pi^*)$ is an expectation
  under $\pi_\theta$. Estimating its gradient requires the **log-density of
  $\pi^*$ (up to a per-prompt constant) at student samples**, and never
  requires sampling from $\pi^*$. This is the only option when $\pi^*$ is the
  Gibbs distribution, which is intractable to sample but easy to score.

This gives a complete two-by-two classification:

|  | Access: samples from $\pi^*$ (forward KL) | Access: $\log\pi^*$ at student samples (reverse KL) |
|---|---|---|
| **Target = data / teacher** | SFT; off-policy distillation | OPD |
| **Target = tilted reference $\pi_{\mathrm{ref}}e^{R/\beta}/Z$** | rejection-sampling SFT (best-of-$n$ / RAFT / ReST: approximate target samples, then MLE) | KL-regularized RL; RLHF/RLVR |

Every cell is occupied by a known method family, and each family is the unique
combination of a target and an access mode.

## 3. Second unification: one gradient, three weightings

### 3.1 The score-function lemma and baselines

**Lemma.** For any $x$,

$$
\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\big[\nabla_\theta\log\pi_\theta(y\mid x)\big]=0 .
$$

*Derivation.*
$\mathbb E[\nabla\log\pi_\theta]=\sum_y\pi_\theta\frac{\nabla\pi_\theta}{\pi_\theta}
=\nabla\sum_y\pi_\theta=\nabla 1=0$. $\square$

The same holds per token conditioned on the prefix:
$\mathbb E_{y_t\sim\pi_\theta(\cdot\mid s_t)}[\nabla\log\pi_\theta(y_t\mid s_t)]=0$.

**Corollary (baselines).** Any weight term that is constant given the state at
which the score is taken—a per-prompt constant $b(x)$ for the sequence score,
or any function of $s_t$ for the token-$t$ score—contributes zero to the
expected gradient. This corollary is used three times below.

### 3.2 The master identity

All three gradients can be written as

$$
\boxed{\;
\nabla_\theta L(\theta)
=-\,\mathbb E_x\,\mathbb E_{y\sim\mu(\cdot\mid x)}
\left[\sum_{t=1}^{T} w_t(x,y)\,
\nabla_\theta\log\pi_\theta(y_t\mid s_t)\right]
\;}
$$

for a sampling distribution $\mu$ and weights $w_t$. Every method is weighted
maximum likelihood: make the sampled tokens more likely in proportion to
$w_t$, or less likely when $w_t<0$.

**(a) SFT.** Differentiating $L_{\mathrm{SFT}}$ directly gives
$\mu=p_{\mathrm{data}}$, $w_t=1$: every token of every data sequence is pushed
up with equal weight. An importance-sampling rewrite puts SFT in on-policy
form:

$$
\nabla L_{\mathrm{SFT}}
=-\,\mathbb E_{y\sim p_{\mathrm{data}}}\big[\nabla\log\pi_\theta(y\mid x)\big]
=-\,\mathbb E_{y\sim\pi_\theta}\!\left[
\frac{p_{\mathrm{data}}(y\mid x)}{\pi_\theta(y\mid x)}\,
\nabla\log\pi_\theta(y\mid x)\right],
$$

i.e. a policy gradient whose sequence weight is the importance ratio
$p_{\mathrm{data}}/\pi_\theta$—large exactly on data sequences the student
currently underweights. (The rewrite requires
$\operatorname{supp}\,p_{\mathrm{data}}\subseteq\operatorname{supp}\,\pi_\theta$
and is an identity in expectation, not the estimator SFT actually uses; SFT
samples from $p_{\mathrm{data}}$ and needs no such ratio.)

**(b) RL.** The score-function (REINFORCE) identity: for $\beta=0$,

$$
\nabla_\theta\,\mathbb E_{y\sim\pi_\theta}[R]
=\sum_y R\,\nabla\pi_\theta
=\sum_y R\,\pi_\theta\,\nabla\log\pi_\theta
=\mathbb E_{y\sim\pi_\theta}\!\left[R(x,y)\sum_t\nabla\log\pi_\theta(y_t\mid s_t)\right].
$$

So $\mu=\pi_\theta$ and $w_t=R(x,y)$: **one scalar copied to every token**. By
the baseline corollary, $w_t=R-b(x)$ has the same expectation; advantage
estimation, group normalization (GRPO), and reward-to-go are all
variance-reduction refinements of this single weight (see
`ppo_trpo_gae_textbook_primer.md` §2).

**(c) Reverse KL to any target with evaluable unnormalized density.** Let
$\tilde\pi^*(y\mid x)$ be an unnormalized target density,
$\pi^*=\tilde\pi^*/Z(x)$. Then

$$
\nabla_\theta\operatorname{KL}(\pi_\theta\Vert\pi^*)
=\mathbb E_{y\sim\pi_\theta}\!\left[
\Big(\log\pi_\theta(y\mid x)-\log\tilde\pi^*(y\mid x)\Big)\,
\nabla\log\pi_\theta(y\mid x)\right].
$$

*Derivation.* Write
$\operatorname{KL}=\mathbb E_{\pi_\theta}[g_\theta]$ with
$g_\theta(y)=\log\pi_\theta-\log\pi^*$. Differentiating through both the
sampling distribution and the integrand,

$$
\nabla\,\mathbb E_{\pi_\theta}[g_\theta]
=\mathbb E_{\pi_\theta}\big[g_\theta\,\nabla\log\pi_\theta\big]
+\mathbb E_{\pi_\theta}\big[\nabla g_\theta\big],
$$

and $\nabla g_\theta=\nabla\log\pi_\theta$, whose expectation vanishes by the
lemma. Finally $\log\pi^*=\log\tilde\pi^*-\log Z(x)$, and the per-prompt
constant $\log Z(x)$ drops by the baseline corollary. $\square$

The two instantiations:

- $\tilde\pi^*=\pi_T$ (**OPD**): the sequence weight is
  $\log\pi_T(y\mid x)-\log\pi_\theta(y\mid x)=\sum_t r_t$ with **per-token
  reward**

  $$
  r_t=\log\pi_T(y_t\mid s_t)-\log\pi_\theta(y_t\mid s_t).
  $$

  Because $r_j$ for $j<t$ is a function of the prefix $s_t$, it is a valid
  baseline for the token-$t$ score and drops from the expected gradient. The
  minimal-variance form of the weight is therefore the reward-to-go
  $w_t=\sum_{j\ge t} r_j$. Each $r_t$ is **computed exactly** from teacher and
  student log-probabilities, not estimated from outcomes.

- $\log\tilde\pi^*=\log\pi_{\mathrm{ref}}+R/\beta$ (**KL-regularized RL**):
  the weight becomes
  $-\frac1\beta\big(R(x,y)-\beta\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}\big)$
  (as a loss; sign flips for the maximization direction). This is precisely the
  KL-shaped reward $R-\beta\log(\pi_\theta/\pi_{\mathrm{ref}})$ used in
  practical RLHF, derived here rather than postulated.

### 3.3 Summary table

| | $\mu$ (rollout distribution) | $w_t$ | signal density | where the noise is |
|---|---|---|---|---|
| SFT | $p_{\mathrm{data}}$ | $1$ | every token | none in $w_t$; the cost is train/deploy state mismatch |
| RL | $\pi_\theta$ | $R(x,y)-b(x)$, one scalar per response | one number per response | Monte-Carlo estimation of the weight; credit assignment across tokens |
| OPD (reverse KL) | $\pi_\theta$ | $\sum_{j\ge t}\big(\log\pi_T(y_j\mid s_j)-\log\pi_\theta(y_j\mid s_j)\big)$ | every token | almost none: the weight is computed, not estimated |

RL and OPD run the identical algorithm—sample from the student, weight the
scores, ascend—and differ only in whether the weight comes from an external
scalar evaluation or from an evaluable target log-density. SFT differs from
both only in whose samples are weighted.

## 4. The two fundamental axes

The unifications above compress every distinction into two independent axes.

### Axis 1: whose distribution generates the training states

SFT optimizes the policy on prefixes drawn from $p_{\mathrm{data}}$; at
deployment the policy conditions on prefixes drawn from itself. A policy with
per-token error rate $\varepsilon$ on the data distribution can accumulate
$O(T^2\varepsilon)$ total error over a length-$T$ rollout, because each mistake
moves the state distribution further from the training distribution and the
policy has never been supervised on the resulting states (Ross & Bagnell,
2010; DAgger, 2011). Training on the policy's own state distribution—which
both RL and OPD do—restores the $O(T\varepsilon)$ rate. This is the
technique-independent argument for on-policy training.

The same axis explains the qualitative difference between the KL directions
when the student cannot represent the target exactly:

- Forward KL is **mass-covering**: the integrand
  $\pi^*\log(\pi^*/\pi_\theta)\to\infty$ wherever $\pi_\theta\to 0$ while
  $\pi^*>0$, so the student must spread mass over every mode of the target,
  including modes it cannot model well.
- Reverse KL is **zero-forcing**: the integrand
  $\pi_\theta\log(\pi_\theta/\pi^*)$ blows up wherever $\pi_\theta>0$ while
  $\pi^*\approx 0$, so the student withdraws mass from regions the target
  rejects and may concentrate on a subset of the target's modes.

When $\pi^*$ is realizable and optimization is exact, both directions have the
same unique optimum $\pi_\theta=\pi^*$; the directions differ only under model
misspecification or limited optimization—which is the practical regime.

### Axis 2: how the target is specified

1. **Samples of $\pi^*$** (SFT). Cheapest supervision to collect for human
   targets; density never needed. Forces forward KL and off-policy states.
2. **Token-level log-density of $\pi^*$** (OPD). The weight in the master
   identity is exact at every token, so gradient variance is minimal and
   sample efficiency is high. The price: the target must exist as an evaluable
   model, so the student's endpoint is the teacher—an in-principle ceiling
   (extrapolation schemes such as ExOPD relax it by modifying the target, not
   the mechanism).
3. **Reward evaluations $R(x,y)$** (RL). The target
   $\pi_{\mathrm{ref}}e^{R/\beta}/Z$ is defined implicitly by the reward and
   need not equal any existing model, so no ceiling exists: RL can specify
   behavior better than every current policy. The price: the weight is a
   single Monte-Carlo scalar per response, so the gradient is sparse and
   high-variance, and assigning that scalar to individual tokens is the credit
   assignment problem that PPO/GAE/GRPO machinery exists to mitigate.

A shared limitation of both on-policy columns: the expectation is under
$\pi_\theta$, so the gradient carries no information about states the student
never visits. If the student assigns probability $\approx 0$ to every
rewarded response (or to the teacher's preferred continuations), both the RL
weight and the OPD weight are computed only on unrewarded, student-typical
trajectories, and learning stalls. GRPO's zero gradient on all-wrong groups is
one instance; the general fact is a property of any
$\mathbb E_{y\sim\pi_\theta}[\cdot]$ objective.

## 5. Reductions of named methods

- **REINFORCE / PPO / GRPO**: estimators of $\nabla J_\beta$ from §3.2(b) with
  baselines, learned advantages, importance ratios, and clipping added for
  variance and trust-region control. None change the objective.
- **Off-policy word-level distillation** (teacher-generated text, forward KL
  on teacher prefixes): SFT-side cell of the table—target samples, forward
  divergence, target states.
- **GKD / on-policy distillation**: §1.3; the reverse-KL member is the one
  equal to RL with reward $\beta\log(\pi_T/\pi_{\mathrm{ref}})$.
- **Best-of-$n$ / rejection-sampling fine-tuning (RAFT, ReST)**: approximate
  sampling from the tilted target $\pi^*_\beta$, then forward-KL fitting—the
  fourth cell of the table.
- **RLHF with the reward $R-\beta\log(\pi_\theta/\pi_{\mathrm{ref}})$**: the
  reverse-KL gradient of §3.2(c) with
  $\log\tilde\pi^*=\log\pi_{\mathrm{ref}}+R/\beta$.
- **Hybrid teacher-plus-reward methods** (dGRPO, TRRD, RLSD, SDAR, …): all
  operate inside the master identity, altering $w_t$ or its clipping using
  both a reward term and a teacher log-ratio term; see
  `opd_rl_connection_and_reward_losses.md`.

## 6. What resists unification

The unified view should not be overstated. Three differences are fundamental,
not presentational:

1. **Objective identity under misspecification.** Forward and reverse KL to
   the same target are different objectives with different optima whenever the
   target is not realizable. SFT and OPD from the same teacher are therefore
   not the same method with different estimators; they are different targets
   for a limited-capacity student.
2. **The ceiling.** OPD's target is an existing model; RL's target is defined
   by a specification. Whatever the estimator, distillation cannot specify
   behavior no teacher exhibits, and RL can.
3. **Where the statistical cost is paid.** SFT pays in train/deploy state
   mismatch, OPD pays in teacher-evaluation compute and teacher trust, RL pays
   in weight variance and credit assignment. These costs live in different
   parts of the master identity ($\mu$ versus $w_t$ versus the estimator of
   $w_t$) and cannot be traded away by algorithmic rearrangement alone—only
   by changing the access mode, which changes the method's cell in the table.

## Cross-references

- `soft_rl_textbook_primer.md` §6: the KL-regularized RL problem and the Gibbs
  optimum, derived via soft Bellman equations.
- `ppo_trpo_gae_textbook_primer.md` §2: the policy gradient theorem, baselines,
  and reward-to-go in full detail.
- `opd_rl_connection_and_reward_losses.md`: the specific hybrid losses that
  live between the RL and OPD cells.
- `ppo_opd_value_bridge.md`: value-function view of the OPD reward.

## References

- Ross & Bagnell (2010), *Efficient Reductions for Imitation Learning*; Ross,
  Gordon & Bagnell (2011), *DAgger* — compounding-error bounds for off-policy
  imitation.
- Williams (1992), *REINFORCE* — the score-function gradient.
- Peters & Schaal (2007); Rawlik et al. (2012); Levine (2018), *RL as
  inference* — the Gibbs form of KL-regularized control.
- Agarwal et al. (2024), *GKD* — on-policy distillation and divergence choice.
- Korbak, Perez & Buckley (2022), *RL with KL penalties is better viewed as
  Bayesian inference* — the distribution-matching reading of RLHF.
