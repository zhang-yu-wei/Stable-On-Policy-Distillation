# PPO, TRPO, and GAE: A Textbook Primer

Companion to `soft_rl_textbook_primer.md` and `ppo_opd_value_bridge.md`. The
soft-RL primer covered the *objective* side: what problem KL-regularized RL
poses and what its optimal solution looks like. This note covers the
*optimization* side: how PPO — the algorithm the bridge note compares against,
gradient by gradient — actually computes its update. Written in the same
style: starting from the policy gradient theorem, changing one thing at a
time, with derivations, worked readings of each formula, and exercises. A
table mapping this note's notation to the bridge note's is at the end.

The chapter answers three questions in order:

1. What is the gradient of expected return with respect to the policy
   parameters? (Section 2 — the policy gradient theorem.)
2. How large a step may be taken on data collected from the *previous*
   policy? (Sections 3–7 — the performance difference lemma, TRPO, PPO.)
3. How should per-step advantages be estimated from a learned value
   function? (Section 8 — GAE.)

PPO as used in practice is the combination of the answers to 2 and 3.

---

## 1. Setting: two separate problems

Notation as in the soft primer: states $s$, actions $a$, dynamics
$p(s',r\mid s,a)$, discount $\gamma$, and a policy $\pi_\theta(a\mid s)$ with
parameters $\theta$. The objective is undiscounted-start expected return,

$$
J(\theta) = \mathbb{E}_{\pi_\theta}\big[G_0\big],
\qquad
G_t = \sum_{k\ge0}\gamma^k R_{t+k+1}.
\tag{1.1}
$$

The value functions of a policy are the conditional expectations of the
return — the expected return from a state (or a state–action pair) when
$\pi$ is followed thereafter:

$$
v_\pi(s) = \mathbb{E}_\pi\big[G_t \mid S_t{=}s\big],
\qquad
q_\pi(s,a) = \mathbb{E}_\pi\big[G_t \mid S_t{=}s, A_t{=}a\big].
\tag{1.2}
$$

We want to do gradient ascent on $J$. Two distinct difficulties stand
between us and that, and it is worth separating them at the outset because
different components of the algorithm address them:

- **Estimation.** $\nabla J$ is an expectation over trajectories; we only
  have samples. The variance of naive estimators is large, and reducing it
  is the entire subject of baselines and GAE (Sections 2 and 8).
- **Staleness.** Every estimate is an expectation *under the current
  policy's own state distribution*. The moment we update $\theta$, the
  distribution changes and the estimate refers to a policy we no longer
  have. How far one update may push $\theta$ before the estimate stops being
  informative is the subject of trust regions (Sections 3–7).

## 2. The policy gradient theorem

Section 1 set the goal: maximize $J(\theta)$ by gradient ascent,
$\theta \leftarrow \theta + \eta\,\nabla J(\theta)$. This section computes
$\nabla J$. Its endpoint is one theorem — the **policy gradient theorem**,
stated at (2.6). Equations (2.1)–(2.5) are the steps of its proof, and
(2.7)–(2.8) refine the theorem into the form algorithms actually implement.

The proof starts from the **score function identity**. For any family of
distributions $p_\theta$ and any function $f$ that does not depend on
$\theta$,

$$
\nabla_\theta\, \mathbb{E}_{x\sim p_\theta}[f(x)]
= \sum_x \nabla_\theta\, p_\theta(x)\, f(x)
= \sum_x p_\theta(x)\,\nabla_\theta \log p_\theta(x)\, f(x)
= \mathbb{E}_{x\sim p_\theta}\big[f(x)\,\nabla_\theta \log p_\theta(x)\big],
\tag{2.1}
$$

where the middle step uses
$\nabla_\theta\, p_\theta = p_\theta\, \nabla_\theta \log p_\theta$. The
identity converts the derivative of an expectation — not directly estimable
by sampling, because $\theta$ sits inside the distribution being sampled —
into an expectation of a derivative, which a sample average can estimate.
Its reading: to increase $\mathbb{E}[f]$, increase the log-probability of
every outcome $x$ in proportion to its value $f(x)$.

Now apply (2.1) with $x$ an entire trajectory
$\tau = (s_0, a_0, s_1, a_1, \dots)$ and $f$ the return. A trajectory's
probability is a product of factors of two alternating kinds — the agent
chooses actions, the environment chooses successor states:

$$
p_\theta(\tau) = p(s_0)\prod_t \pi_\theta(a_t\mid s_t)\,
p(s_{t+1}\mid s_t, a_t).
\tag{2.2}
$$

Only the $\pi_\theta(a_t\mid s_t)$ factors contain $\theta$; the
initial-state factor $p(s_0)$ and the transition factors
$p(s_{t+1}\mid s_t, a_t)$ are fixed properties of the environment. Taking
the logarithm of (2.2) turns the product into a sum of terms, and
differentiating with respect to $\theta$ deletes every term that does not
contain $\theta$:

$$
\nabla_\theta \log p_\theta(\tau)
= \sum_t \nabla_\theta \log \pi_\theta(a_t\mid s_t).
\tag{2.3}
$$

It is worth pausing on what (2.3) does and does not require. Before
differentiation, the log-probability of a trajectory contained the
transition probabilities $p(s_{t+1}\mid s_t, a_t)$ — quantities we usually
cannot evaluate, because the environment's rules are not available to us in
numerical form. After differentiation they are gone: to evaluate (2.3) on a
sampled trajectory, we only need our own policy — run it at each visited
state and differentiate the log-probability it assigned to the action that
was actually taken there. The environment is still needed to *produce* the
trajectory (acting in the world is how one samples from
$p(s_{t+1}\mid s_t, a_t)$), but its probabilities never have to be known,
evaluated, or differentiated. Methods with this property are called
**model-free**: they learn a policy without ever constructing, or being
given, a model of the dynamics. Equation (2.3) is the reason
policy-gradient methods are model-free.

Substituting (2.3) into (2.1), with $f(\tau) = G_0$ the return of the whole
trajectory:

$$
\nabla J(\theta)
= \mathbb{E}_{\pi_\theta}\Big[\,G_0 \sum_t
\nabla_\theta \log\pi_\theta(a_t\mid s_t)\Big].
\tag{2.4}
$$

One refinement remains. Distribute $G_0$ across the sum in (2.4): the
time-$t$ term is $G_0\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)$, the score
of action $a_t$ multiplied by *all* the rewards. Relative to time $t$, the
return splits into two parts:

$$
G_0
= \underbrace{r_1 + \gamma r_2 + \cdots + \gamma^{t-1} r_t}_{
\text{already received when } a_t \text{ is chosen}}
\;+\; \gamma^t\, G_t.
$$

(Timing convention: $r_{t+1}$ is the reward that follows action $a_t$, so
$r_1, \dots, r_t$ are consequences of $a_0, \dots, a_{t-1}$, already fixed
numbers by the time the agent stands in $s_t$.) So (2.4) multiplies the
score of $a_t$ partly by return the action can still affect — the
$\gamma^t G_t$ part — and partly by rewards determined before $a_t$ was
drawn, crediting the action for them as if it had caused them. The
already-determined part contributes nothing in expectation: condition on
the trajectory up to $s_t$; the early rewards become constants, and the
conditional expectation of $\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ over
$a_t \sim \pi_\theta$ is zero — the same computation as the baseline lemma
(2.7) below (Exercise 11.1 asks for the details). Discarding that part
leaves each action paired with the return from its own time onward:

$$
\nabla J(\theta)
= \mathbb{E}_{\pi_\theta}\Big[\sum_t \gamma^t\, G_t\,
\nabla_\theta \log\pi_\theta(a_t\mid s_t)\Big].
\tag{2.5}
$$

The dropped terms are the products
$\big(r_1 + \gamma r_2 + \cdots + \gamma^{t-1} r_t\big)\,
\nabla_\theta\log\pi_\theta(a_t\mid s_t)$, one per $t$. They are zero in
expectation but nonzero on each sample, so removing them estimates the same
gradient with less noise.

The results so far assemble into the section's central statement.

**Theorem (policy gradient; Sutton et al. 2000).** For any differentiable
policy $\pi_\theta$, with
$d^\pi(s) = (1-\gamma)\sum_t \gamma^t \Pr(S_t{=}s)$ the discounted
state-visitation distribution,

$$
\nabla J(\theta)
= \tfrac{1}{1-\gamma}\,
\mathbb{E}_{s\sim d^\pi}\,\mathbb{E}_{a\sim\pi_\theta}
\big[q_\pi(s,a)\,\nabla_\theta\log\pi_\theta(a\mid s)\big].
\tag{2.6}
$$

*Proof.* Two steps remain from (2.5). First, condition each term on
$(s_t, a_t)$: the factor $\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ is a
fixed number once $(s_t, a_t)$ is fixed, so it passes through the
conditional expectation, and
$\mathbb{E}[G_t \mid s_t, a_t] = q_\pi(s_t, a_t)$ by definition (1.2).
This replaces $G_t$ by $q_\pi(s_t, a_t)$. Second, the remaining
outer sum $\sum_t \gamma^t\,\mathbb{E}[\cdot]$ weights each state by how
often, discounted, the policy visits it; collecting those weights into the
distribution $d^\pi$ (the factor $1-\gamma$ normalizes the total weight
$\sum_t \gamma^t$ to one) gives (2.6). $\blacksquare$

Sampling $s \sim d^\pi$ is not a new procedure: the states of $\pi$'s own
trajectories, taken with weight $\gamma^t$, are samples from $d^\pi$ — the
data already being collected. The substance of the theorem is what is
*absent* from (2.6): $d^\pi$ depends on $\theta$, since changing the policy
changes which states are visited, yet no $\nabla_\theta\, d^\pi$ term
appears. The score terms alone give the exact gradient.

Why prefer this form to (2.5), which estimates the same gradient? Because
of what it says the gradient depends on: only which $(s,a)$ pairs are
visited, and the value of the *function* $q_\pi$ there — the sampled return
no longer appears. Consequently anything that estimates $q_\pi$ may stand
in for the sampled return; substituting a learned estimate (a critic) is
the actor–critic idea, and choosing that estimate is the subject of
Section 8.

The distribution $d^\pi$ is defined here for a reason that arrives in
Sections 4–5: the improvement of a new policy over an old one is an
expectation under the *new* policy's visitation distribution (4.2), the
surrogate objective (5.1) is built by substituting the old policy's
$d^\pi$ for it, and that substitution is the single approximation the
trust-region methods exist to control. The analysis needs a name for "the
distribution of states a policy visits."

In the token MDP, $d^\pi$ has a concrete reading: a state is a prefix, one
rollout visits every prefix of the completion it generates, and $d^\pi$ is
the distribution of contexts the model creates for itself while writing —
each token position of each rollout in a batch is one sample from it. A
policy change therefore changes not only which tokens are chosen but which
prefixes are faced at all; that shift is the $d^{\pi'} \ne d^{\pi}$
mismatch of Sections 4–5, and it is the same phenomenon the OPD notes call
prefix drift.

**The baseline.** For any function $b(s)$ of the state alone,

$$
\mathbb{E}_{a\sim\pi_\theta}\big[b(s)\,\nabla_\theta\log\pi_\theta(a\mid s)\big]
= b(s)\,\nabla_\theta \sum_a \pi_\theta(a\mid s) = b(s)\,\nabla_\theta 1 = 0,
\tag{2.7}
$$

so subtracting $b(s)$ from $q_\pi(s,a)$ in (2.6) changes nothing in
expectation, while it can change the variance greatly. The standard choice
$b = v_\pi$ turns the coefficient into the **advantage**:

$$
\nabla J(\theta)
\propto \mathbb{E}\big[A_\pi(s,a)\,\nabla_\theta\log\pi_\theta(a\mid s)\big],
\qquad
A_\pi(s,a) = q_\pi(s,a) - v_\pi(s).
\tag{2.8}
$$

The reading of (2.8): *raise the log-probability of each action in
proportion to how much better it is than the average action at its state.*
The advantage is the natural coefficient because it is centered at every
state — the part of the return that the action did not control (the state's
baseline value) contributes no gradient, only variance, and (2.7) licenses
removing it.

REINFORCE (Williams 1992) is the one-sample estimate of (2.5). An
**actor–critic** method replaces the sampled return $G_t$ with an estimate
built from a learned state-value function $V_\phi \approx v_\pi$; the policy
$\pi_\theta$ is the actor, $V_\phi$ the critic. How exactly to build that
estimate is deferred to Section 8; Sections 3–7 take an advantage estimate
$\hat A_t$ as given and ask how far to step.

## 3. Why the step size is a separate problem

Suppose we have an accurate gradient. Why not take ordinary gradient steps
with a small learning rate? Two reasons, one about geometry and one about
feedback.

**Geometry.** The learning rate bounds the step in *parameter* space, but
the quantity that must stay small is the change in the *policy* — the
conditional distributions $\pi_\theta(\cdot\mid s)$. The map from parameters
to distributions is not uniformly scaled: the same parameter displacement
can leave the policy nearly unchanged in one region of parameter space and
change action probabilities drastically in another (probabilities respond
exponentially to logits, and the sensitivity depends on where the current
probabilities sit). A learning rate that is safe at one point of training is
destructive at another. The correct control variable is a bound on the
change of the policy's distributions directly — for instance a bound on
$\mathrm{KL}\big(\pi_{\theta_{\text{old}}}(\cdot\mid s)\,\Vert\,
\pi_\theta(\cdot\mid s)\big)$ averaged over states.

**Feedback.** In on-policy RL the next batch of data is collected by the
policy the last update produced. A destructive update therefore does not
merely lose progress; it degrades the data used for every subsequent update.
An update that makes the policy much worse can leave it collecting
trajectories from which no informative gradient can be estimated. This is
why on-policy methods insist on controlled steps rather than relying on the
averaging-out of occasional bad ones.

The next three sections build the tool that makes "controlled step" precise:
an *exact* expression for the improvement of one policy over another
(Section 4), an approximation to it computable from old data together with
the region where the approximation is valid (Section 5), and two practical
algorithms that maximize the approximation inside that region (Sections 6
and 7).

## 4. The performance difference lemma

**Lemma (Kakade & Langford 2002).** For any two policies $\pi'$ and $\pi$,

$$
J(\pi') - J(\pi)
= \mathbb{E}_{\tau\sim\pi'}\Big[\sum_t \gamma^t A_\pi(s_t,a_t)\Big].
\tag{4.1}
$$

*Proof.* Add and subtract the old policy's values along the new policy's
trajectories:

$$
\mathbb{E}_{\tau\sim\pi'}\Big[\sum_t \gamma^t
\big(r_{t+1} + \gamma\, v_\pi(s_{t+1}) - v_\pi(s_t)\big)\Big].
$$

Summed over $t$, the value terms telescope: every $v_\pi(s_{t+1})$ appears
once with coefficient $+\gamma^{t+1}$ and once with $-\gamma^{t+1}$, leaving
only $-v_\pi(s_0)$, whose expectation is $-J(\pi)$. The reward terms sum to
$J(\pi')$. So the whole expression equals $J(\pi') - J(\pi)$. On the other
hand, conditioned on $(s_t, a_t)$, the environment behaves identically under
both policies, so
$\mathbb{E}[r_{t+1} + \gamma v_\pi(s_{t+1})\mid s_t,a_t] = q_\pi(s_t,a_t)$,
and each summand has conditional expectation
$q_\pi(s_t,a_t) - v_\pi(s_t) = A_\pi(s_t,a_t)$. $\blacksquare$

Restated with the visitation distribution:

$$
J(\pi') - J(\pi)
= \tfrac{1}{1-\gamma}\,
\mathbb{E}_{s\sim d^{\pi'}}\,\mathbb{E}_{a\sim\pi'}\big[A_\pi(s,a)\big].
\tag{4.2}
$$

The reading: *the improvement from switching to $\pi'$ equals the old
policy's advantage function, evaluated on the new policy's action choices,
accumulated over the new policy's states.* This is exact — no approximation
yet. Every factor is available from data collected under $\pi$ except one:
the outer expectation is over $d^{\pi'}$, the state distribution of the
policy we are trying to choose. That single inaccessible factor is the
entire difficulty, and everything in TRPO and PPO is a response to it.

## 5. The surrogate objective and its region of validity

Make the one available approximation: replace the new policy's state
distribution by the old one's. Define the **surrogate objective**

$$
L_\pi(\pi')
= J(\pi) + \tfrac{1}{1-\gamma}\,
\mathbb{E}_{s\sim d^{\pi}}\,\mathbb{E}_{a\sim\pi'}\big[A_\pi(s,a)\big]
= J(\pi) + \tfrac{1}{1-\gamma}\,
\mathbb{E}_{s\sim d^{\pi},\,a\sim\pi}
\Big[\frac{\pi'(a\mid s)}{\pi(a\mid s)}\,A_\pi(s,a)\Big].
\tag{5.1}
$$

The second equality is importance sampling over the *action only*, and it
is nothing more than multiplying and dividing by $\pi(a\mid s)$ inside the
sum:

$$
\mathbb{E}_{a\sim\pi'}\big[A_\pi(s,a)\big]
= \sum_a \pi'(a\mid s)\,A_\pi(s,a)
= \sum_a \pi(a\mid s)\,\frac{\pi'(a\mid s)}{\pi(a\mid s)}\,A_\pi(s,a)
= \mathbb{E}_{a\sim\pi}\Big[\frac{\pi'(a\mid s)}{\pi(a\mid s)}\,
A_\pi(s,a)\Big].
$$

This is exact, and cheap: for the sampled action, both policies'
probabilities are known numbers. Everything in (5.1) is now an expectation
under the old policy — computable from the batch we already have. The ratio
$\rho(s,a) = \pi'(a\mid s)/\pi(a\mid s)$ will be the central object of
Sections 6 and 7.

Two properties determine how (5.1) may be used.

**First-order agreement.** At $\pi'=\pi$ the correction term vanishes
(advantages average to zero at every state), so $L_\pi(\pi)=J(\pi)$; and the
gradients of $L$ and $J$ with respect to $\pi'$ coincide there (Exercise
11.3). For infinitesimal steps, maximizing the surrogate *is* following the
policy gradient. The two objectives separate only at finite step sizes.

**The error bound.** (Schulman et al. 2015, sharpening Kakade & Langford.)
With $\varepsilon = \max_{s,a}|A_\pi(s,a)|$,

$$
\big|\,J(\pi') - L_\pi(\pi')\,\big|
\;\le\; \frac{4\,\varepsilon\,\gamma}{(1-\gamma)^2}\,
\max_s \mathrm{KL}\big(\pi(\cdot\mid s)\,\Vert\,\pi'(\cdot\mid s)\big).
\tag{5.2}
$$

The surrogate is trustworthy exactly to the extent that the policy change is
small in KL. Note the factor $(1-\gamma)^{-2}$: one factor of the horizon
because errors are summed over time, a second because the state-distribution
mismatch itself grows over time — a policy change at early steps shifts
which states are visited at all later steps. Long-horizon problems make the
surrogate less trustworthy at the same per-state KL.

**Monotonic improvement in principle.** Combining: the penalized surrogate
$L_\pi(\pi') - C\cdot\max_s\mathrm{KL}$ (with $C$ the constant of (5.2)) is
a lower bound on $J(\pi')$ that touches $J$ at $\pi'=\pi$. Maximizing it
over $\pi'$ therefore can only produce a policy at least as good as $\pi$ —
each iteration maximizes a lower bound that is tight at the current point,
so $J$ never decreases. This is an idealized algorithm, not a practical one:
the theoretical $C$ is so large that the maximizing step is minuscule. TRPO
and PPO are two ways to keep the structure of this argument while replacing
$C$ with something usable.

## 6. TRPO: the constraint form

TRPO (Schulman et al. 2015) replaces the penalty with a **constraint**:

$$
\max_\theta\;
\hat{\mathbb{E}}\Big[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\text{old}}}(a\mid s)}
\,\hat A(s,a)\Big]
\quad\text{subject to}\quad
\hat{\mathbb{E}}_s\,\mathrm{KL}\big(\pi_{\theta_{\text{old}}}(\cdot\mid s)
\,\Vert\,\pi_\theta(\cdot\mid s)\big) \le \delta.
\tag{6.1}
$$

Two substitutions relative to the theory: the maximum-over-states KL becomes
an average (the maximum is not estimable from samples), and the penalty
coefficient — which theory sets too conservatively and which would in any
case need re-tuning per problem — becomes a direct bound $\delta$ on the
policy change (a small constant such as $10^{-2}$, meaningful in the same
units across problems).

**Solving (6.1).** Expand the objective to first order and the constraint
to second order around $\theta_{\text{old}}$. The constraint's expansion
has no zeroth- or first-order terms: the KL of a distribution to itself is
zero, and zero is its minimum, so the gradient there vanishes too — the
quadratic term is the first one that survives. This leaves

$$
\max_d\; g^\top d
\quad\text{s.t.}\quad
\tfrac12\, d^\top F\, d \le \delta,
\qquad
F = \mathbb{E}\big[\nabla\log\pi_\theta\,\nabla\log\pi_\theta^\top\big],
\tag{6.2}
$$

where $g$ is the surrogate gradient and $F$ — the Hessian of the KL at zero
displacement — is the **Fisher information matrix**. The solution is

$$
d^* = \sqrt{\frac{2\delta}{g^\top F^{-1} g}}\; F^{-1} g.
\tag{6.3}
$$

The direction $F^{-1}g$ is the **natural gradient** (Kakade 2001). Its
literal meaning: among all steps of a fixed small size *measured in KL*,
it is the one that maximally increases the linearized objective. Because the
size is measured on the policy's distributions rather than on the
parameters, the step is invariant to how the policy is parameterized — the
geometry problem of Section 3, answered exactly.

In implementation, $F^{-1}g$ is computed by conjugate gradient using
Fisher-vector products (never forming $F$), and because (6.2) is only an
approximation of (6.1), the step is finished with a backtracking line
search that checks the *actual* surrogate improvement and the *actual* KL.

**Cost.** Each update needs the conjugate-gradient inner loop, a
line search, and per-architecture care (parameters shared between policy and
value heads complicate the Fisher metric). TRPO is also awkward to combine
with the standard workflow of many minibatch epochs under Adam. PPO exists
to keep TRPO's behavior — improve the surrogate while keeping the policy
change bounded — with first-order optimization only.

## 7. PPO: the clipping form

PPO (Schulman et al. 2017) drops both the constraint and the penalty and
instead modifies the surrogate so that, per sample, moving the ratio beyond
a band around 1 offers no further objective gain. With
$\rho_t(\theta) = \pi_\theta(a_t\mid s_t)/\pi_{\theta_{\text{old}}}(a_t\mid s_t)$:

$$
L^{\mathrm{CLIP}}(\theta)
= \hat{\mathbb{E}}_t\Big[
\min\Big(\rho_t(\theta)\,\hat A_t,\;
\mathrm{clip}\big(\rho_t(\theta),\,1{-}\epsilon,\,1{+}\epsilon\big)\,\hat A_t
\Big)\Big].
\tag{7.1}
$$

The formula rewards a case analysis. Write it out per sample:

| sign of $\hat A_t$ | position of $\rho_t$ | active branch | gradient |
|---|---|---|---|
| $+$ | $\rho_t \le 1+\epsilon$ | unclipped $\rho_t\hat A_t$ | present; increases $\rho_t$ |
| $+$ | $\rho_t > 1+\epsilon$ | clipped $(1{+}\epsilon)\hat A_t$, constant | zero |
| $-$ | $\rho_t \ge 1-\epsilon$ | unclipped $\rho_t\hat A_t$ | present; decreases $\rho_t$ |
| $-$ | $\rho_t < 1-\epsilon$ | clipped $(1{-}\epsilon)\hat A_t$, constant | zero |

The pattern: **the gradient is zeroed exactly when the ratio has already
moved past the band in the direction the advantage asks for, and is retained
everywhere else** — including far outside the band on the opposite side,
where the retained gradient points back toward the band. So the clip removes
the *incentive to keep going* past $1\pm\epsilon$, but never removes the
gradient that returns an over-shot ratio. The $\min$ also makes (7.1) a
pointwise lower bound on the unclipped surrogate, equal to it at
$\rho_t = 1$: a pessimistic surrogate, in keeping with the lower-bound
argument of Section 5.

**Why multiple epochs.** After a single gradient step the ratios barely
leave 1 and the clip is never active — (7.1) would then be indistinguishable
from the plain surrogate. PPO's sample efficiency comes from running
*several epochs* of minibatch updates on each batch; the clip is what stops
the ratios from moving far during those epochs. Clipping and data reuse are
one design, not two.

**What clipping does not do.** It is not a projection: a ratio can pass the
band (the gradient from *that sample* becomes zero, but gradients from other
samples in the minibatch still move the shared parameters). It enforces no
bound on the measured KL between $\pi_{\theta_{\text{old}}}$ and
$\pi_\theta$, and in practice the KL can grow beyond intention; common
safeguards are stopping the inner epochs early when a measured-KL threshold
is passed, or the PPO-penalty variant, which puts
$-\beta_{\mathrm{KL}}\,\mathrm{KL}$ back into the objective with an
adaptively adjusted coefficient.

**The full practical loss** adds a critic regression and an entropy bonus:

$$
L(\theta,\phi) = L^{\mathrm{CLIP}}(\theta)
- c_v\,\hat{\mathbb{E}}_t\big(V_\phi(s_t) - \hat V^{\mathrm{targ}}_t\big)^2
+ c_H\,\hat{\mathbb{E}}_t\,\mathcal H\big(\pi_\theta(\cdot\mid s_t)\big).
\tag{7.2}
$$

The entropy term is the small-$\alpha$ version of the soft-RL objective
(2.1) of the soft primer, used here only as a regularizer against premature
determinism rather than as part of the problem definition.

**A caution: two different KLs.** In LLM fine-tuning, PPO appears together
with a KL penalty against a reference model, and the two uses of KL are
easily confused because both are written $\mathrm{KL}$ and both are called
regularization:

- $\mathrm{KL}(\pi_{\theta_{\text{old}}}\Vert\pi_\theta)$ — old policy vs.
  new policy. Step-size control. Algorithmic, transient: it constrains each
  update and leaves no trace in what the final policy is.
- $\mathrm{KL}(\pi_\theta\Vert\pi_{\mathrm{ref}})$ — policy vs. reference
  model. Part of the *objective* (equation (6.1) of the soft primer, entering
  PPO through the per-token reward). Permanent: it changes what the optimal
  policy is, and is what makes the LLM problem a soft-RL problem.

TRPO/PPO clipping concerns only the first. The bridge note's Step 7
(point 5, "the trust regions are in different places") depends on keeping
these two apart.

## 8. GAE: estimating the advantage

One object in (7.1) is still unspecified: $\hat A_t$. The exact advantage is
unknown; it must be estimated from sampled rewards and a learned value
function $V \approx v_\pi$, and the choice of estimator is a bias–variance
decision with one parameter. That parameter is $\lambda$, and GAE (Schulman
et al. 2016) is the family it indexes.

**The TD residual.** The elementary unit is

$$
\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t).
\tag{8.1}
$$

This compares two estimates of the same return: $V(s_t)$, made before the
step, and $r_{t+1} + \gamma V(s_{t+1})$, made after observing one real
reward and the actual next state. $\delta_t$ is the correction one
observed step forces on the critic. It is the elementary unit because
every estimator in this section — each $\hat A^{(k)}$ and every
GAE($\lambda$) — will turn out to be a weighted sum of these residuals.

If $V = v_\pi$ exactly, then
$\mathbb{E}[\delta_t \mid s_t, a_t]
= q_\pi(s_t,a_t) - v_\pi(s_t) = A_\pi(s_t,a_t)$:
the one-step residual is an unbiased advantage estimate. If $V$ has error
$e = V - v_\pi$, the bias is
$\gamma\,\mathbb{E}[e(s_{t+1})] - e(s_t)$ (Exercise 11.4) — the estimator
inherits the critic's error.

**$k$-step estimators.** Use $k$ sampled rewards before falling back on the
critic:

$$
\hat A^{(k)}_t
= \sum_{l=0}^{k-1}\gamma^l r_{t+l+1} + \gamma^k V(s_{t+k}) - V(s_t)
= \sum_{l=0}^{k-1}\gamma^l\,\delta_{t+l},
\tag{8.2}
$$

the second equality by telescoping. To see it at $k=2$, write out the sum
of two residuals:

$$
\delta_t + \gamma\,\delta_{t+1}
= \big(r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\big)
+ \gamma\big(r_{t+2} + \gamma V(s_{t+2}) - V(s_{t+1})\big);
$$

the two $V(s_{t+1})$ terms cancel, leaving
$r_{t+1} + \gamma r_{t+2} + \gamma^2 V(s_{t+2}) - V(s_t) = \hat A^{(2)}_t$.
In general every intermediate $V(s_{t+l})$ appears once with each sign and
only the endpoints survive. As $k$ grows, the learned $V$ enters only
through the final term $\gamma^k V(s_{t+k})$, whose weight shrinks — bias
from the critic decreases — while more sampled rewards enter — variance
increases. $k$ trades bias against variance, but only in integer steps.

The reading of (8.2): define the forecast of $G_t$ after $l$ observed
steps,

$$
F_l = \sum_{j=0}^{l-1}\gamma^j r_{t+j+1} + \gamma^l V(s_{t+l}),
\qquad F_0 = V(s_t).
$$

Observing one more step revises the forecast by
$F_{l+1} - F_l = \gamma^l\,\delta_{t+l}$ — the discounted residual is
exactly the revision that step causes. Summing, $\hat A^{(k)}_t = F_k -
F_0$: the advantage estimate is the *total forecast revision* — how much
better or worse the return looks after $k$ real steps than it looked
before the action was taken. That is why the advantage is a sum of
residuals: the return is a sum over time, so information about it arrives
additively, one step's revision at a time. And when $V = v_\pi$, given
$(s_t, a_t)$ only the first revision has nonzero mean
($\mathbb{E}[\delta_t] = A_\pi$); every later $\delta_{t+l}$ has zero mean
conditioned on its own state. The later residuals are included not for
their mean but for what they cancel: when $V$ is wrong, each additional
real step replaces the critic's error at a nearby state with a sampled
outcome.

**GAE.** Instead of choosing one $k$, average them all with exponentially
decaying weights $(1-\lambda)\lambda^{k-1}$:

$$
\hat A^{\mathrm{GAE}(\gamma,\lambda)}_t
= (1-\lambda)\sum_{k\ge1}\lambda^{k-1}\hat A^{(k)}_t
= \sum_{l\ge0}(\gamma\lambda)^l\,\delta_{t+l}.
\tag{8.3}
$$

The collapse to the single sum is Exercise 11.5 (swap the order of the two
sums; the weights on $\delta_{t+l}$ sum to $\lambda^l$). In the
forecast-revision reading, GAE keeps the revision at lag $l$ with weight
$\lambda^l$: full weight on the immediate revision, geometrically
attenuated weight on later ones. The two endpoints
recover the two extremes:

- $\lambda = 0$: $\hat A_t = \delta_t$. Lowest variance; bias is whatever
  the critic's error induces.
- $\lambda = 1$: $\hat A_t = \sum_l \gamma^l r_{t+l+1} - V(s_t)
  = G_t - V(s_t)$, the Monte Carlo return minus a baseline. Highest
  variance; and here a subtlety worth stating precisely. Pointwise this is
  *not* an unbiased estimate of $A_\pi(s_t,a_t)$ unless $V = v_\pi$ — but as
  a *policy-gradient* estimator it is unbiased for **any** $V$, because at
  $\lambda=1$ the critic appears only as a state-dependent baseline, and
  (2.7) says baselines never bias the gradient. For $\lambda < 1$,
  unbiasedness of the gradient does require $V = v_\pi$; the critic is then
  inside the estimate, not just subtracted at the end.

**The reading of $\lambda$**: it is the weight given to the critic's
estimates of the future relative to sampled rewards. The right setting
depends on how accurate the critic currently is — a statement about the
*state of training*, not about the problem. This is why the bridge note
treats $\lambda$ as a measured quantity rather than a fixed constant: it
sets $\lambda = 1 - \mathrm{EV}_+(1-\lambda_{\min})$, so that a critic
explaining none of the return variance gives $\lambda = 1$ (pure Monte
Carlo credit) and $\lambda$ falls toward $\lambda_{\min}$ as explained
variance rises.

**$\gamma$ versus $\lambda$.** Superficially symmetric in (8.3), the two
parameters are different in kind. $\gamma$ is part of the *problem
statement*: changing it changes $J$, hence the optimal policy. $\lambda$
only changes the *estimator* of a fixed target: with an exact critic every
$\lambda$ estimates the same advantage, and no value of $\lambda$ changes
what the algorithm converges to. (In practice $\gamma$ is also often set
below the problem's true discount as an additional variance reduction — that
use of $\gamma$ *does* introduce bias in the original problem.)

**Computation.** Expanding (8.3) one step gives the backward recursion

$$
\hat A_t = \delta_t + \gamma\lambda\,\hat A_{t+1},
\tag{8.4}
$$

one sweep from the end of the trajectory. The critic's regression target in
(7.2) is then $\hat V^{\mathrm{targ}}_t = \hat A_t + V(s_t)$ (the
$\lambda$-return). Common practice also standardizes the advantages across
the batch to mean zero and unit variance before the actor update — the
`whiten` step of the bridge note's pseudocode.

**The token MDP.** For LLM fine-tuning: deterministic dynamics, $\gamma=1$,
reward only at the terminal token. With $\lambda=1$, (8.3) telescopes to

$$
\hat A_t = R - V(s_t),
\tag{8.5}
$$

the sequence reward minus the critic at each prefix. If additionally the
critic is a per-prompt constant equal to the mean reward of a group of
rollouts, (8.5) is exactly GRPO's advantage — every token in a rollout gets
the same credit. This chain of specializations, ($\gamma{=}1$,
$\lambda{=}1$, constant critic) $\Rightarrow$ GRPO, is what the bridge
note's procedure uses in reverse: it starts at the GRPO corner and moves
toward PPO-grade credit as the measured explained variance justifies
lowering $\lambda$ and trusting a real critic.

**The OPD reading.** In the token MDP the residual (8.1) itself
specializes: interior steps have zero reward, so

$$
\delta_t = V(s_{t+1}) - V(s_t) \quad (\text{interior}),
\qquad
\delta_T = R - V(s_T) \quad (\text{terminal}).
\tag{8.6}
$$

Per-token credit is the revision that writing the token causes in the
forecast of the final reward, and the sampled outcome enters only through
the terminal residual. Now recall the identity the companion notes are
built on (soft primer (6.3); bridge note eq. (2)): a soft-optimal teacher
carries its value function in its logits,

$$
\beta\log\frac{\pi_T(y_t\mid s_t)}{\pi_{\mathrm{ref}}(y_t\mid s_t)}
= V_T(s_t y_t) - V_T(s_t).
\tag{8.7}
$$

Here $\beta$ is the temperature of the KL-regularized objective the
teacher was trained on, $\mathbb{E}[R] - \beta\,\mathrm{KL}(\pi\Vert
\pi_{\mathrm{ref}})$, and (8.7) is the closed-form optimal policy of that
objective (soft primer (6.2)) solved backwards for the values. Note the
teacher alone determines only the ratio $R/\beta$ (the policy depends on
the pair only through $e^{R/\beta}$); fixing the reward's units — e.g.
binary $R \in \{0,1\}$ — makes $\beta$ measurable by calibrating
teacher-implied values against observed outcomes (the bridge note's link
function (6)).

The right side is the interior residual of (8.6) with $V = V_T$. So the
teacher's per-token log-ratio on the sampled token *is* a TD residual —
this section's elementary unit, evaluated with the critic implicit in the
teacher's logits instead of a learned one. Running GAE with that critic:

- $\lambda = 0$:
  $\hat A_t = \beta\log(\pi_T/\pi_{\mathrm{ref}})(y_t\mid s_t)$ — OPD's
  sampled-token credit. Dense, available by forward pass, no sampling
  variance; and the observed reward never reaches interior tokens.
- $\lambda = 1$: $\hat A_t = R - V_T(s_t)$ — the Monte Carlo estimate
  with the teacher's forecast as baseline, unbiased for any $V_T$
  by (2.7).
- $0 < \lambda < 1$: the outcome enters token $t$'s credit with weight
  $\lambda^{T-t}$; near-terminal tokens are checked against the actual
  reward, early tokens rely on the teacher's judgment.

The bias condition of this section now reads as a statement about OPD:
$\lambda < 1$ is unbiased only when the critic equals the *current
policy's* value, and $V_T$ is the teacher's value evaluated on the
student's visitation distribution — the gap between the two is prefix
drift, and at $\lambda = 0$ the objective cannot detect it, because the
outcome never reaches the credit. The bridge note's procedure is the
correction stated in this section's own terms: initialize the critic at
$V_T$, let regression on real returns overwrite it, and lower $\lambda$
from $1$ only as explained variance rises. (This identifies OPD's credit
coefficient on the sampled token; the full OPD gradient also moves
unsampled tokens through the softmax difference $p - q$, which no
score-function method expresses — the bridge note's Step 7.)

## 9. The assembled algorithm

```
initialize policy θ, critic φ
loop:
  π_old ← π_θ ; collect a batch of trajectories with π_old
  compute δ_t from V_φ                       (8.1)
  compute Â_t by backward recursion           (8.4)
  targets V̂ ← Â + V_φ ; optionally whiten Â across the batch
  for K epochs of minibatches:
    ascend  L_CLIP(θ)  using ratios π_θ/π_old (7.1)
    descend (V_φ − V̂)²
    optionally stop early if mean KL(π_old‖π_θ) exceeds a threshold
```

Where each section of this note appears in the loop: the gradient being
followed near $\rho=1$ is the policy gradient (Section 2); the ratios exist
because the batch is reused across epochs and the surrogate must be
evaluated at policies other than the one that collected the data (Section
5); the clip and the epoch count are one mechanism bounding how far the
reuse is pushed (Section 7); and the advantages entering the surrogate are
GAE (Section 8), with the critic fitted to the $\lambda$-return targets it
itself helped construct — the consistency condition of ordinary policy
evaluation, run concurrently with improvement.

## 10. Notation correspondence with `ppo_opd_value_bridge.md`

| This note | Bridge note |
|---|---|
| critic $V_\phi$, fitted to $\lambda$-returns | the critic head on real returns |
| $\hat A^{\mathrm{GAE}}$ with $\gamma=1$ (8.3) | $\mathrm{GAE}_\lambda(t)=\sum_l \lambda^l \delta_{t+l}$ |
| whitened advantages | $\widehat\Delta_t = \mathrm{whiten}(\mathrm{GAE}_\lambda(t))$ |
| $\lambda$ set by critic accuracy (Section 8) | $\lambda = 1-\mathrm{EV}_+(1-\lambda_{\min})$, the head-vs-MC trust |
| $\gamma{=}1$, $\lambda{=}1$, group-mean constant critic (8.5) | GRPO's $\hat A_i$; round 1 of the procedure |
| the OPD reading (8.6)–(8.7): teacher log-ratio $=$ TD residual with $V{=}V_T$ | the teacher-as-implicit-critic identity, eq. (2); Step 7's credit-channel comparison |
| clipped-surrogate gradient at $\rho_t=1$ | $\nabla_z L^{\mathrm{PPO}}_t = -\widehat\Delta_t(\mathbf 1_{y_t}-p)$, eq. (12) — the bridge takes PPO at $r_t=1$, i.e. at the start of the inner epochs, which is why no clip appears there |
| step-size KL: $\mathrm{KL}(\pi_{\text{old}}\Vert\pi_\theta)$, Sections 5–7 | "PPO: a hard clip on the ratio" — the trust region in update space |
| objective KL: $\mathrm{KL}(\pi\Vert\pi_{\mathrm{ref}})$, soft primer (6.1) | the $\beta$-weighted KL to the teacher — the trust region in solution space |

The one fact from this chapter that the bridge depends on most: PPO's
actor update is, at $\rho=1$, exactly "advantage times score function"
(Sections 2 and 7), so comparing PPO with an OPD-style update reduces to
comparing the *coefficient* each places on the score function — which is
precisely the gradient-by-gradient comparison the bridge note's Step 7
carries out.

## 11. Exercises

**11.1** (Causality.) Show that the terms coupling
$\nabla\log\pi(a_t\mid s_t)$ with rewards $r_{k}$ for $k \le t$ vanish in
expectation, turning (2.4) into (2.5). *Hint: iterate the baseline argument
(2.7) with $b$ equal to a past reward, which is a function of the
trajectory up to time $t$.*

**11.2** (Performance difference.) Reprove (4.1) for the finite-horizon
undiscounted case ($\gamma=1$, episodes of length $T$), and verify that the
token-MDP instance (terminal reward only) reads: the improvement of $\pi'$
over $\pi$ equals the expectation under $\pi'$-rollouts of the sum of
$\pi$'s per-token advantages.

**11.3** (First-order agreement.) Differentiate (5.1) with respect to the
new policy's parameters at $\pi'=\pi$ and check the result equals (2.6).
Conclude that clipping in (7.1) has no effect on the first gradient step of
each PPO batch, and that $\nabla L^{\mathrm{CLIP}}$ at $\rho=1$ is the
advantage-weighted score function.

**11.4** (Critic bias.) With $e = V - v_\pi$, show
$\mathbb{E}[\delta_t\mid s_t,a_t] = A_\pi(s_t,a_t)
+ \gamma\,\mathbb{E}[e(s_{t+1})\mid s_t,a_t] - e(s_t)$, and deduce the bias
of $\hat A^{(k)}$ and of GAE($\lambda$). Verify that at $\lambda=1$ the
critic-dependent terms reduce to $-e(s_t)$, a pure baseline.

**11.5** (The collapse.) Prove the second equality of (8.3) by exchanging
the order of summation, and verify the two endpoint cases $\lambda\in\{0,1\}$
directly.

**11.6** (The GRPO corner.) In the token MDP with terminal reward $R$ and
$\gamma=\lambda=1$, show $\hat A_t = R - V(s_t)$ for every $t$; then take
$V \equiv \bar R_x$ (the mean reward of $G$ rollouts of the same prompt) and
show the whitened credit equals GRPO's normalized advantage, constant across
the tokens of each rollout.

## References

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018),
  Chapter 13 — policy gradient fundamentals.
- Williams, *Simple Statistical Gradient-Following Algorithms for
  Connectionist Reinforcement Learning* (1992) — REINFORCE.
- Sutton, McAllester, Singh & Mansour, *Policy Gradient Methods for RL with
  Function Approximation* (NeurIPS 2000) — the policy gradient theorem.
- Kakade, *A Natural Policy Gradient* (NeurIPS 2001).
- Kakade & Langford, *Approximately Optimal Approximate Reinforcement
  Learning* (ICML 2002) — the performance difference lemma and conservative
  policy iteration.
- Schulman, Levine, Moritz, Jordan & Abbeel, *Trust Region Policy
  Optimization* (ICML 2015).
- Schulman, Moritz, Levine, Jordan & Abbeel, *High-Dimensional Continuous
  Control Using Generalized Advantage Estimation* (ICLR 2016).
- Schulman, Wolski, Dhariwal, Radford & Klimov, *Proximal Policy
  Optimization Algorithms* (arXiv 1707.06347, 2017).
