# Soft RL and the Soft Bellman Equations: A Textbook Primer

Companion to `ppo_opd_value_bridge.md`. That note's Appendix A derives the
soft Bellman equations rigorously for the token MDP; this note is the
*textbook chapter that would precede it*, written in the style of Sutton &
Barto — starting from the ordinary Bellman equations, changing one thing at
a time, with backup diagrams, worked numerical examples, and exercises. A
table mapping this note's notation to the bridge note's is at the end.

---

## 1. Review: the ordinary ("hard") Bellman equations

Recall the finite MDP: states $s$, actions $a$, dynamics $p(s', r \mid s, a)$,
discount $\gamma$. A policy $\pi(a\mid s)$ gives rise to the return
$G_t = \sum_{k\ge0}\gamma^k R_{t+k+1}$, and the value functions are its
conditional expectations:

$$
v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t{=}s], \qquad
q_\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t{=}s, A_t{=}a].
$$

The defining property of value functions is that they satisfy recursive
consistency conditions — the Bellman equations. For a fixed policy (the
*prediction* problem):

$$
v_\pi(s) = \sum_a \pi(a\mid s)\, q_\pi(s,a),
\qquad
q_\pi(s,a) = \sum_{s',r} p(s',r\mid s,a)\big[r + \gamma\, v_\pi(s')\big].
\tag{1.1}
$$

For the best achievable values (the *control* problem), the average over
actions is replaced by a maximum:

$$
v_*(s) = \max_a q_*(s,a),
\qquad
q_*(s,a) = \sum_{s',r} p(s',r\mid s,a)\big[r + \gamma\, v_*(s')\big].
\tag{1.2}
$$

Two structural facts about (1.2) are our points of departure. First, the
$\max$ is a **hard** operator: it commits entirely to the best action, and
the values of all other actions are irrelevant to $v_*$ no matter how close
they come. Second, the optimal policy it induces — greedy with respect to
$q_*$ — can always be taken **deterministic**. In the backup diagram for
$v_*$, the agent at the root looks down its action arcs and keeps exactly
one:

```
        (s)                 max: keep the best arc,
       / | \                discard the others entirely
     a₁  a₂  a₃
      1  0.9 0.9      →     v*(s) = 1
```

Soft RL arises from asking: what problem are we solving if, instead, the
agent must keep *all* the arcs, weighted by how good they are?

## 2. A new problem statement: being paid for randomness

Soft RL is not a new algorithm for the old problem; it is a **new problem**.
We change the objective the agent maximizes. In the *entropy-regularized*
(maximum-entropy) formulation, the agent is paid, at every step, a bonus
proportional to the entropy of its action distribution at the current state:

$$
J(\pi) \;=\; \mathbb{E}_\pi\!\Big[\sum_{t}\gamma^t\big(R_{t+1}
+ \alpha\, \mathcal H(\pi(\cdot\mid S_t))\big)\Big],
\qquad \mathcal H(\mu) = -\sum_a \mu(a)\log\mu(a).
\tag{2.1}
$$

The **temperature** $\alpha > 0$ sets the exchange rate between reward and
randomness: one nat of entropy is worth $\alpha$ units of reward.

Why would anyone want to be paid for randomness? Several reasons, of
increasing depth:

1. **Exploration that survives optimality.** In hard RL, exploration is an
   add-on ($\varepsilon$-greedy, optimistic initialization) that must be
   scheduled away. In soft RL the optimal policy is itself stochastic;
   hedging across good actions is part of what "optimal" means.
2. **All good behaviors, not one.** If two action sequences achieve nearly
   the same return, the soft-optimal policy keeps both. This makes the
   learned policy robust to perturbations of the environment and a better
   initialization for later fine-tuning.
3. **Smoothness.** The hard $\max$ makes $v_*$ a non-differentiable function
   of the rewards, and greedy policies change discontinuously. Softening the
   max makes everything smooth in a way that interacts well with function
   approximation.
4. **The inference view.** Soft RL is exactly the problem whose solution is
   a posterior distribution — "RL as probabilistic inference" (Levine 2018).
   The posterior emerges directly from the equations in Section 8.

A remark on names before continuing: everything below generalizes from
"entropy bonus" to "KL penalty against a reference policy," and it is the KL
form that matters for language models. We develop the entropy case first
because it is the textbook case, and do the generalization in Section 6 — it
takes a single line.

## 3. Soft value functions and the soft Bellman expectation equation

Since we changed the objective, we must redefine the value functions to
match. The **soft state value** of $s$ under $\pi$ is the expected
discounted sum of rewards *plus all future entropy bonuses*:

$$
v_\pi(s) = \mathbb{E}_\pi\!\Big[\sum_{k\ge0}\gamma^k\big(R_{t+k+1}
+ \alpha\,\mathcal H(\pi(\cdot\mid S_{t+k}))\big)\,\Big|\, S_t{=}s\Big].
\tag{3.1}
$$

For the **soft action value** we must fix a convention, and it matters:
$q_\pi(s,a)$ is defined *without* the entropy bonus of the current state,

$$
q_\pi(s,a) = \sum_{s',r} p(s',r\mid s,a)\big[r + \gamma\, v_\pi(s')\big].
\tag{3.2}
$$

The reason is bookkeeping: the entropy at state $s$ is a property of the
*distribution* the agent uses there, not of any particular action sampled
from it. So the bonus is assigned to the state — it appears in $v$, not in
$q$. With this convention, peeling one step off (3.1) gives the **soft
Bellman expectation equation**:

$$
v_\pi(s) \;=\; \sum_a \pi(a\mid s)\, q_\pi(s,a)
\;+\; \alpha\,\mathcal H(\pi(\cdot\mid s))
\;=\; \sum_a \pi(a\mid s)\big[q_\pi(s,a) - \alpha\log\pi(a\mid s)\big].
\tag{3.3}
$$

Compare with (1.1): the backup through the transition, equation (3.2), is
**completely unchanged**. Only the backup through the action node has
changed, by the additive entropy term. Note also that for a *fixed* policy
the entropy term is just a known constant at each state — so the system
(3.2)–(3.3) is still linear in the values, and soft policy evaluation works
exactly like ordinary policy evaluation. Nothing interesting has happened
yet. Everything interesting happens when we maximize.

## 4. The key lemma: maximizing over a distribution

The central result of the entire subject is a one-state problem. In
hard RL, the improvement step at a state is: given the action values
$q(s,\cdot)$, pick the best action. In soft RL it must be: given
$q(s,\cdot)$, pick the best *distribution* $\mu$ over actions, where "best"
now means

$$
\max_{\mu}\; \sum_a \mu(a)\, q(s,a) + \alpha\,\mathcal H(\mu).
\tag{4.1}
$$

Without the entropy term the answer is trivial — put all mass on the argmax.
With it, spreading mass earns bonus, and concentrating mass earns reward;
the optimum balances the two. The balance has a closed form.

**Lemma (Gibbs variational principle).** For any function $f$ over actions
and any distribution $\mu$,

$$
\sum_a \mu(a) f(a) + \alpha \mathcal H(\mu)
\;=\;
\alpha\log\sum_a e^{f(a)/\alpha}
\;-\;
\alpha\,\mathrm{KL}\big(\mu \,\big\Vert\, \mathrm{softmax}(f/\alpha)\big).
\tag{4.2}
$$

The proof is a substitution (`ppo_opd_value_bridge.md` A.2 gives it in the
KL form); the reading is what matters. The first term on the right does not
depend on $\mu$ at all, and the second is a non-negative penalty that
vanishes exactly when $\mu$ *is* the softmax distribution. Therefore:

$$
\max_\mu \Big\{\textstyle\sum_a \mu(a) f(a) + \alpha\mathcal H(\mu)\Big\}
= \alpha\log\sum_a e^{f(a)/\alpha},
\qquad
\mu^*(a) = \frac{e^{f(a)/\alpha}}{\sum_b e^{f(b)/\alpha}}.
\tag{4.3}
$$

The solution produces two objects at once. Each is a smooth substitute for
one of the two hard operations ordinary RL is built on, and the rest of the
theory is obtained by writing the standard equations with the soft version
substituted wherever the hard version appears:

| hard object | soft object that replaces it |
| --- | --- |
| $\max_a f(a)$ — a number | the **soft maximum** $\alpha\log\sum_a e^{f(a)/\alpha}$ (log-sum-exp) |
| $\arg\max_a f(a)$ — one action | the soft argmax: the **Boltzmann distribution** $\mathrm{softmax}(f/\alpha)$ |

A caution about the word "softmax," which the field uses for two different
things. The function machine-learning practice calls softmax — the vector of
normalized exponentials — is not a soft *maximum*; it is a soft *argmax*.
The genuine soft maximum is the log-sum-exp, of which the familiar softmax
is the gradient. The soft Bellman equation involves both, each in the role
the hard equation gave to $\max$ and $\arg\max$ respectively.

How soft is the soft max? The reader should verify the two-sided bound

$$
\max_a f(a) \;\le\; \alpha\log\sum_a e^{f(a)/\alpha}
\;\le\; \max_a f(a) + \alpha\log|\mathcal A|.
\tag{4.4}
$$

The soft max is an *optimistic smoothing* of the max: never below it (every
extra action can only add to the sum), and above it by at most the entropy
bonus of full randomness. As $\alpha\to0$ the two bounds coincide and we recover
hard RL exactly; as $\alpha\to\infty$ the bonus dominates and the optimal
distribution flattens toward uniform.

## 5. The soft Bellman optimality equation

We can now do to the soft problem what Chapter 3 of the textbook does to the
hard one. Define $v_*(s)$ as the best achievable soft value from $s$. At
each state the agent faces exactly the one-state problem (4.1) with
$f(a) = q_*(s,a)$, so applying the Lemma:

$$
\boxed{\;
v_*(s) = \alpha\log\sum_a e^{\,q_*(s,a)/\alpha},
\qquad
q_*(s,a) = \sum_{s',r} p(s',r\mid s,a)\big[r+\gamma\,v_*(s')\big].
\;}
\tag{5.1}
$$

These are the **soft Bellman optimality equations**. Set them beside (1.2):
the $q$-backup through the environment is *identical*; only the state backup
changed, $\max \to$ log-sum-exp. In the backup diagram, instead of keeping
the best arc, the root blends all arcs — exponentially weighted, so the best
arc still dominates, but every decent arc contributes:

```
        (s)                 soft max: blend all arcs,
       / | \                best weighted most
     a₁  a₂  a₃
      1  0.9 0.9      →     v*(s) = log(e¹ + 2e⁰·⁹) ≈ 1.67   (α = 1)
```

And where the hard theory says "any greedy policy is optimal, and a
deterministic one exists," the soft theory says something stronger and
simpler — the optimal policy is *unique*, stochastic, and available in
closed form:

$$
\pi_*(a\mid s) = e^{\big(q_*(s,a) - v_*(s)\big)/\alpha}.
\tag{5.2}
$$

No normalization constant appears because, by (5.1), $e^{v_*(s)/\alpha}$
*is* the normalizer $\sum_a e^{q_*(s,a)/\alpha}$. This innocuous observation
— **the optimal value is the log-partition-function of the optimal action
values** — is the single most consequential identity in soft RL. Take
logarithms of (5.2):

$$
\alpha\log\pi_*(a\mid s) = q_*(s,a) - v_*(s) = \text{(soft advantage)}.
\tag{5.3}
$$

*The log-probabilities of a soft-optimal policy are its own advantage
function.* A soft-optimal policy contains its own critic; values can be
read from it with a forward pass. (This is the "language models are
secretly Q-functions" identity, and it is the entire reason the
teacher-as-implicit-critic bridge exists.)

Existence and uniqueness follow directly: log-sum-exp is a non-expansion in the
sup-norm (its gradient is a probability vector), so the soft Bellman
operator is a $\gamma$-contraction just as the hard one is, and soft value
iteration converges to the unique fixed point. (A caution for the
literature: this is true of log-sum-exp, but *not* of the
"Boltzmann-averaged" operator $\sum_a \mathrm{softmax}(q/\alpha)(a)\,q(a)$,
which is not a non-expansion and can have multiple fixed points — one reason to keep the two
soft operators of Section 4 distinct. See Asadi & Littman 2017 on
mellowmax.)

**Example 5.1 (one state, two actions, temperature sweep).** Let
$q_* = (1, 0)$. For two actions (5.2) reduces to a sigmoid in the gap:
$\pi_*(a_1) = \sigma\big((q_1-q_2)/\alpha\big)$.

| $\alpha$ | $v_*(s) = \alpha\log(e^{1/\alpha}+1)$ | $\pi_*(a_1)$ |
|---|---|---|
| $\to 0$ | $1$ | $1$ (hard RL) |
| $0.5$ | $1.06$ | $0.88$ |
| $1$ | $1.31$ | $0.73$ |
| $2$ | $1.95$ | $0.62$ |
| $\to\infty$ | $\approx \alpha\log 2 + \tfrac12$ | $\tfrac12$ (uniform) |

**Example 5.2 (breadth beats depth).** This example shows the qualitative
way soft values *differ* from hard values, not merely smooth them. Two
states, three actions each, $\alpha=1$:

- State $L$: action values $(1, -10, -10)$ — one excellent way forward, the
  rest terrible.
- State $R$: action values $(0.8,\, 0.8,\, 0.8)$ — three good ways forward.

Hard RL: $v_*(L)=1 > v_*(R)=0.8$; prefer $L$. Soft RL:
$v_*(L) = \log(e^1 + 2e^{-10}) \approx 1.00$, but
$v_*(R) = \log(3e^{0.8}) = 0.8 + \log 3 \approx 1.90$; prefer $R$,
decisively. The soft value of a state measures **not how good the best
continuation is, but how much good continuation there is** — a log-count of
good paths, each weighted by its quality. In statistical mechanics language,
$v_*$ is a free energy and $e^{v_*/\alpha}$ a partition function. (For
sequence models this is exactly why *fork* states — prefixes with many
viable continuations — carry extra soft value, and why a teacher that
suppresses forks is mis-scoring precisely the quantity soft RL says to
measure.)

## 6. From entropy to a reference: KL-regularized RL

Now the one-line generalization. Entropy is, up to a constant, negative KL
divergence to the uniform distribution:
$\mathcal H(\mu) = \log|\mathcal A| - \mathrm{KL}(\mu\,\Vert\,U)$. So the
entropy bonus was already a penalty for straying from a fixed reference
policy — the uniform one. Replace it with an arbitrary reference
$\pi_{\mathrm{ref}}$:

$$
J(\pi) = \mathbb{E}_\pi\Big[\sum_t \gamma^t\big(R_{t+1}
- \alpha\,\mathrm{KL}\big(\pi(\cdot\mid S_t)\,\Vert\,
\pi_{\mathrm{ref}}(\cdot\mid S_t)\big)\big)\Big].
\tag{6.1}
$$

Every derivation above goes through with $e^{f/\alpha}$ replaced by
$\pi_{\mathrm{ref}}\,e^{f/\alpha}$ — the Lemma in Section 4 was already
proved in this generality in the bridge note's A.2. The soft Bellman
optimality equation and optimal policy become

$$
v_*(s) = \alpha\log\sum_a \pi_{\mathrm{ref}}(a\mid s)\,e^{\,q_*(s,a)/\alpha},
\qquad
\pi_*(a\mid s) = \pi_{\mathrm{ref}}(a\mid s)\,
e^{\big(q_*(s,a)-v_*(s)\big)/\alpha},
\tag{6.2}
$$

and the advantage identity (5.3) becomes a statement about the
**log-ratio**:

$$
\alpha\log\frac{\pi_*(a\mid s)}{\pi_{\mathrm{ref}}(a\mid s)}
= q_*(s,a) - v_*(s).
\tag{6.3}
$$

One genuine difference from the entropy case deserves a caution, because it
is easy to miss and it changes the character of the value function. The
two-sided bound (4.4) fails: since
$\sum_a \pi_{\mathrm{ref}}(a)e^{q(a)/\alpha} \le e^{\max_a q(a)/\alpha}$,
and by Jensen's inequality on the other side,

$$
\mathbb{E}_{a\sim\pi_{\mathrm{ref}}}\big[q_*(s,a)\big]
\;\le\; v_*(s) \;\le\; \max_a q_*(s,a).
\tag{6.4}
$$

The entropy-regularized value *exceeds* the hard value (the agent is paid
extra for randomness); the KL-regularized value *never* does (KL only
penalizes). It interpolates between the reference's expected performance
($\alpha\to\infty$: pure imitation, $\pi_*\to\pi_{\mathrm{ref}}$) and hard
optimality ($\alpha\to0$, restricted to the reference's support). The
temperature is a continuous parameter interpolating between *imitate* and
*maximize* — exactly the trade-off a distillation-plus-RL method needs to
control.

## 7. Solving soft MDPs: soft policy iteration and its descendants

The algorithmic chapter of the textbook transfers essentially unchanged; we record the
correspondences.

**Soft policy iteration.** Evaluation: solve (3.2)–(3.3) for the current
policy (still a linear system). Improvement: at every state replace $\pi$ by
the Boltzmann/tilted distribution of (4.3) applied to $q_\pi$. The **soft
policy improvement theorem** holds: the new policy's soft value is no less
than the old one's at every state, by exactly the argument of the hard case
with the Lemma supplying the one-state step. Iterating converges to $\pi_*$.

**Soft Q-learning.** Write value iteration on (5.1)/(6.2) and replace the
expectation over transitions with samples. The one-sample target for
$q(s,a)$ after observing $(r, s')$ is

$$
r + \gamma\,\alpha\log\sum_{a'}\pi_{\mathrm{ref}}(a'\mid s')\,
e^{\,q(s',a')/\alpha}
$$

— ordinary Q-learning with the hard $\max_{a'}$ replaced by the soft one.
This is soft Q-learning (Fox, Pakman & Tishby 2016; Haarnoja et al. 2017);
with function approximation and an explicit policy network trained against
the tilt, it becomes Soft Actor-Critic (Haarnoja et al. 2018). This line of
work goes back through maximum-entropy inverse RL (Ziebart et al. 2008–2010) and
linearly-solvable control (Todorov 2007; Kappen 2005).

## 8. Remark: the free-energy form and RL-as-inference

One more consequence, special but illuminating. Suppose the dynamics are
**deterministic** (as in token generation), $\gamma = 1$, horizon finite,
reward terminal. Exponentiate the state equation in (6.2): since
$q_*(s,a) = v_*(s')$ for the successor $s'$,

$$
e^{v_*(s)/\alpha} = \sum_a \pi_{\mathrm{ref}}(a\mid s)\, e^{v_*(s')/\alpha}
= \mathbb{E}_{a\sim\pi_{\mathrm{ref}}}\big[e^{v_*(s')/\alpha}\big].
$$

The *exponentiated* value is a martingale under **reference** rollouts: the
nonlinear Bellman recursion became linear in $e^{v/\alpha}$. Unrolling to
the horizon,

$$
e^{v_*(s)/\alpha} = \mathbb{E}_{\pi_{\mathrm{ref}}}\big[e^{R/\alpha} \mid s\big]
\quad\Longleftrightarrow\quad
v_*(s) = \alpha\log \mathbb{E}_{\pi_{\mathrm{ref}}}\big[e^{R/\alpha}\mid s\big],
\tag{8.1}
$$

the log-moment-generating function of the return under the reference — in
physics terms a free energy, in decision-theory terms a *risk-seeking*
(optimistic) certainty equivalent. And multiplying (6.2) along a full
trajectory telescopes the advantages into $R(y) - v_*(s_0)$, giving the
sequence-level statement

$$
\pi_*(y) \propto \pi_{\mathrm{ref}}(y)\, e^{R(y)/\alpha}:
$$

the soft-optimal policy is the reference **posterior-tilted by exponentiated
reward**. Local Boltzmann optimality at every state and global exponential
tilting of whole trajectories are the same fact — this is the precise
content of "RL as inference." (Caution: the linearization step used
determinism; with stochastic dynamics the expectation over $s'$ sits inside
the exponent in $q_* = r + \gamma\,\mathbb E[v_*]$ and the Jensen gap breaks
(8.1). Todorov's linearly-solvable MDPs recover it by letting the agent
choose next-state distributions directly.)

---

## Notation correspondence with `ppo_opd_value_bridge.md`

The bridge note's Appendix A is this chapter specialized to the token MDP,
with notation $\beta \leftrightarrow \alpha$:

| This note | Bridge note |
|---|---|
| KL-regularized objective (6.1), $\gamma=1$, terminal $R$ | the soft-RL problem of Step 1 |
| Soft Bellman expectation eq. (3.3) | (A.1) |
| Gibbs variational lemma (4.2)/(4.3) | (A.2), "completing the KL" |
| Soft optimality equations (6.2) | (A.3)/(A.4), i.e. eq. (1) |
| $q_*(s,a) = r + \gamma\mathbb E[v_*(s')]$ collapsing to $v_*(sa)$ | "$Q^*(s,a)=V^*(sa)$ by determinism" |
| Advantage identity (6.3) | eq. (2), the teacher-as-implicit-critic identity |
| Free-energy form (8.1) | (A.5), and the binary link (A.6)/eq. (6) |
| Breadth-beats-depth (Example 5.2) | why fork prefixes carry soft value — the quantity fork suppression corrupts |

The two facts from this textbook layer that the whole PPO–OPD bridge
depends on:
from (6.3), *a soft-optimal policy's log-ratios against its reference are
advantages* — so a (soft-optimal) teacher is a critic you read by forward
pass; and from (6.4), the KL-regularized value's meaning — an optimistic
aggregate of the *reference's* continuations, not the student's — which is
exactly why the bridge note's link function (6) and the soft-vs-hard
calibration caveat are not optional.

Two items here *add* to the bridge note rather than restate it:

- **The bound (6.4)** sharpens the "soft vs hard calibration" caveat into a
  directional statement — the teacher potential's optimism is bounded above
  by the best continuation, below by the reference's average.
- **Example 5.2** is the cleanest statement of why fork suppression is an
  error *in the very quantity soft RL defines value to be* — candidate
  sentence for the bridge note's Step 2.

## Exercises

**8.1** Show
$\mathcal H(\mu) = \log|\mathcal A| - \mathrm{KL}(\mu\Vert U)$, and conclude
that the entropy-regularized equations are the $\pi_{\mathrm{ref}} = U$ case
of (6.2) up to a per-step constant $\alpha\log|\mathcal A|$ that shifts $v$
and $q$ equally and cancels from every advantage.

**8.2** Prove the two-sided bound (4.4), and show both bounds are tight (which $f$
achieves each?).

**8.3** For two actions with gap $\Delta = q_1 - q_2$, show
$\pi_*(a_1) = \sigma(\Delta/\alpha)$ and
$v_* = q_2 + \alpha\log(1 + e^{\Delta/\alpha})
= q_1 + \alpha\log(1 + e^{-\Delta/\alpha})$; the second form makes the
$\alpha\to0$ limit ($v_*\to q_1$) immediate. Check both limits in $\alpha$.

**8.4** (Non-expansion.) Show
$\big|\alpha\log\sum_a e^{f(a)/\alpha} - \alpha\log\sum_a e^{g(a)/\alpha}\big|
\le \max_a |f(a)-g(a)|$, and conclude soft value iteration converges.
*Hint: the gradient of log-sum-exp is a probability vector.*

**8.5** (The telescoping.) From (6.3) with deterministic dynamics, show the
log-ratios of $\pi_*$ along any trajectory sum to $R(y) - v_*(s_0)$ — and
explain in one sentence why a policy that is *not* soft-optimal for any
$(R, \pi_{\mathrm{ref}})$ pair has log-ratios that fail to telescope (the
bridge note's Step 2, the "non-conservative field").

## References

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018)
  — the hard-RL baseline this note softens.
- Ziebart et al., *Maximum Entropy Inverse Reinforcement Learning* (AAAI
  2008); Ziebart, PhD thesis (2010) — origin of max-ent control.
- Todorov, *Linearly-solvable Markov decision problems* (NeurIPS 2007);
  Kappen, *Path integrals and symmetry breaking for optimal control* (2005)
  — the linear/free-energy structure of Section 8.
- Fox, Pakman & Tishby, *Taming the Noise in Reinforcement Learning via Soft
  Updates* (UAI 2016) — G-learning / soft Q-learning.
- Haarnoja et al., *Reinforcement Learning with Deep Energy-Based Policies*
  (ICML 2017); *Soft Actor-Critic* (ICML 2018).
- Asadi & Littman, *An Alternative Softmax Operator for Reinforcement
  Learning* (ICML 2017) — mellowmax; the non-expansion caution in Section 5.
- Levine, *Reinforcement Learning and Control as Probabilistic Inference*
  (arXiv 1805.00909, 2018) — the survey behind Section 8.
