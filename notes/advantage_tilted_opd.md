# Advantage-Tilted OPD: Fusing Verifier and Teacher in Distribution Space

This note develops a proposal for combining on-policy distillation with
verifiable reward. It is a continuation of
`opd_rl_connection_and_reward_losses.md`, motivated by three objections to the
basic combinations analyzed there:

1. **Additive KL (dGRPO/HDPO):** empirically, the KL gradient can be much larger
   than the GRPO gradient, so the net update direction is dominated by the
   teacher term.
2. **TRRD:** the teacher only reshapes a scalar reward update's geometry; the
   dense next-token distribution — OPD's actual asset — is discarded.
3. **RLSD:** the teacher's signed, distributional judgment is collapsed to a
   positive scalar weight on the advantage; OPD's role becomes vestigial.

Hard token selection (routing/masking) is excluded by design.

## Step 0: The Idea in Plain Language

**One sentence:** ask what the teacher *would have said, had it known how the
rollout turned out*, and run ordinary OPD toward that answer.

### Why adding losses fails: a tug-of-war in mismatched units

Picture the student as a junior writer with two mentors. The copyeditor
(teacher) marks every line in red ink — dense, always has an opinion. The
publisher (verifier) attaches one sticky note to the finished piece: "sold" or
"didn't sell" — sparse, but grounded in reality.

The additive method lets both mentors grab the pen at once and push. The
copyeditor's push is measured in log-ratios, which have no ceiling: wherever
teacher and student disagree strongly, the red-ink force is enormous. The
publisher's push is a whitened advantage, standardized to unit scale and
therefore usually only a few units in magnitude. (Whitening does not strictly
clip it.) Summing forces measured in different units with one constant $\beta$
is a fight the red ink structurally wins — which is exactly the empirical
failure: the net gradient direction follows the KL term. It is a units
problem, not a tuning problem.

The other combinations avoid the fight by silencing someone: TRRD and RLSD let
the publisher decide everything and demote the copyeditor to adjusting font
sizes (one positive scalar per token); naive OPD fires the publisher.

### The fix: argue on the page, not on the pen

Do not let two opinions push the same pen. Merge them into one edited
manuscript first — a single target distribution per prefix that already
contains both views — and let the student plainly imitate that document.

Why this kills the imbalance is almost silly: probabilities are capped at
100%. The loudest thing the verifier can say inside a distribution is "this
token should be certain." The loudest thing the teacher can say is "match me
exactly." Both opinions are converted into the same bounded currency —
probability mass — *before* anything touches a gradient, so there is no
volume knob left to win with. The gradient itself becomes a thermostat:
"current probability minus target probability," a step proportional to a gap
that can never exceed one, instead of forces sized by unbounded log-ratios.

### The merged document: the teacher who saw the ending

The teacher's next-token distribution is a prior. The rollout's outcome is
evidence — but only about the one token actually sampled. Bayes' update is:
multiply that token's prior probability by $e^{\hat A/\beta}$ and renormalize.
The result is literally the teacher's opinion, updated by the news of how the
response ended.

This is the same object SD-ZERO spends a whole training phase building — an
outcome-conditioned reviser — except here it exists in closed form, one
softmax away.

### A worked example

At some prefix the teacher gives a token 30%. The rollout containing it
succeeded with $\hat A/\beta=1.5$, so multiply by $e^{1.5}\approx4.5$:

$$
30\%\ \to\ \frac{0.30\times4.5}{0.30\times4.5+0.70}\approx 66\%.
$$

Had the rollout failed with $\hat A/\beta=-1.5$: $30\%\to\approx9\%$. Two
details carry the elegance. First, the freed 21 points of probability are not
scattered uniformly — they are redistributed among the alternatives in the
teacher's own proportions. When the verifier says "not this token," the
teacher answers the question GRPO cannot: where should the mass go instead?
Second, the update saturates: no advantage, however huge, can push a target
past 100% or below 0%.

### Who wins arguments: an auction with prices in nats

When teacher and verifier disagree about the sampled token, the rule (derived
as (5) below) is an auction: the token is reinforced iff the verifier's bid
beats the price of disagreement,

$$
\hat A/\beta\;>\;\operatorname{logit}(p_a)-\operatorname{logit}(q_a).
$$

Example: the student currently gives a token 40%, the teacher only 5%. The
price is $\operatorname{logit}(0.40)-\operatorname{logit}(0.05)\approx2.5$
nats. A routine success ($\hat A\approx1$, $\beta=1$) does not meet the
price — the token is trimmed back toward the teacher despite the success. An
exceptional success ($\hat A\approx3$) outbids the teacher and the token
grows anyway. Nobody holds a veto; evidence is priced, and $\beta$ is the
exchange rate.

### The caveat to keep in mind: credit in a decided game

Think of a chess engine's evaluation bar. The exact math (Step 2) says each
token's tilt should be how much that move *shifted the bar* — its marginal
contribution. GRPO's $\hat A$ instead hands every token the full final
result, so mopping-up moves in an already-won game collect full credit for
the win. Policy gradients forgive this (baselines average out, by linearity);
a tilted *target* does not, because the credit sits inside an exponential.
This is the main bias of the cheap surrogate, and a value head that reads the
bar is the principled fix (Step 6).

The rest of the note makes each of these statements exact.

## The Diagnosis: All Three Combine in the Wrong Space

- Additive combines in **gradient space**: two unbounded vector fields summed
  with a scalar $\beta$. Log-ratios over a vocabulary are unbounded; whitened
  advantages are $O(1)$. The two summands do not share a scale, so domination
  is structural, not a tuning accident.
- TRRD combines in **ratio space**: only the sampled token's probability enters,
  through the denominator of an importance ratio.
- RLSD combines in **scalar-weight space**: the teacher distribution is reduced
  to one positive number per token.

The proposal is to combine in **distribution space**: build a single target
distribution per prefix into which both signals are fused *before* any gradient
is taken, then take one divergence toward it. On the probability simplex
everything is bounded and normalized, so scale domination cannot arise.

## Notation

Fix a prompt $x$. Write $s_t=(x,y_{<t})$ for a prefix, and at a given prefix:

$$
\begin{aligned}
q(\cdot)&=\pi_T(\cdot\mid s) && \text{teacher distribution},\\
p(\cdot)&=\pi_\theta(\cdot\mid s) && \text{current student distribution},\\
p_0(\cdot)&=\pi_{\mathrm{old}}(\cdot\mid s) && \text{rollout policy},\\
\rho_t&=\frac{p(y_t)}{p_0(y_t)} && \text{importance ratio for the sampled token},\\
R(x,y)&& & \text{terminal verifier reward},\\
\hat A_i && & \text{whitened GRPO group advantage of response } i.
\end{aligned}
$$

## Step 1: One Objective, Two Estimators

Start from the KL-regularized objective that the additive methods implicitly
optimize:

$$
J(\pi)=\mathbb E_{y\sim\pi}\!\left[R(x,y)\right]-\beta\,\mathrm{KL}\!\left(\pi\,\Vert\,\pi_T\right).
$$

### Closed-form optimum

Adding a Lagrange multiplier for normalization and setting the functional
derivative to zero gives

$$
\pi^*(y\mid x)=\frac{\pi_T(y\mid x)\,e^{R(x,y)/\beta}}{Z(x)},
\qquad
Z(x)=\mathbb E_{y\sim\pi_T}\!\left[e^{R(x,y)/\beta}\right].
$$

### Why this has the form of Bayes' rule

Bayes' rule updates a prior belief over hypotheses after observing evidence:

$$
P(h\mid E)
=\frac{P(h)P(E\mid h)}{\sum_{h'}P(h')P(E\mid h')}.
$$

Here the “hypothesis” $h$ is a complete response $y$. Before consulting the
verifier, use the teacher as the prior belief over responses:

$$
P(y\mid x)=\pi_T(y\mid x).
$$

Now introduce abstract evidence $E$ meaning “this response has a desirable
outcome,” and assign responses the likelihood factor

$$
P(E\mid x,y)\ \propto\ e^{R(x,y)/\beta}.
$$

Substituting these two choices into Bayes' rule gives

$$
\begin{aligned}
P(y\mid x,E)
&=\frac{P(y\mid x)P(E\mid x,y)}
        {\sum_{y'}P(y'\mid x)P(E\mid x,y')}\\
&=\frac{\pi_T(y\mid x)e^{R(x,y)/\beta}}
        {\sum_{y'}\pi_T(y'\mid x)e^{R(x,y')/\beta}}\\
&=\frac{\pi_T(y\mid x)e^{R(x,y)/\beta}}{Z(x)}
=\pi^*(y\mid x).
\end{aligned}
$$

The correspondence is therefore:

| Bayes-rule object | Here |
|---|---|
| Hypothesis | A complete response $y$ |
| Prior $P(y\mid x)$ | Teacher probability $\pi_T(y\mid x)$ |
| Evidence likelihood | Reward factor $e^{R(x,y)/\beta}$ |
| Evidence normalizer (marginal likelihood, up to scale) | $Z(x)$ |
| Posterior $P(y\mid x,E)$ | Outcome-aware target $\pi^*(y\mid x)$ |

The word **factor** is important. $e^{R/\beta}$ need not itself be a calibrated
probability and can be greater than one. Bayes' rule only depends on likelihood
ratios, so any positive factor proportional to a likelihood gives the same
normalized posterior. This construction is often called **generalized Bayes**,
a **Gibbs posterior**, or **control as inference**. If a literal probabilistic
story is desired and rewards are bounded, multiply every factor by a constant
small enough to put it in $[0,1]$; that constant cancels during normalization.

The cleanest way to see the update is through posterior odds. For two complete
responses $y_1$ and $y_2$,

$$
\frac{\pi^*(y_1\mid x)}{\pi^*(y_2\mid x)}
=\underbrace{\frac{\pi_T(y_1\mid x)}{\pi_T(y_2\mid x)}}_{\text{teacher's prior odds}}
 \underbrace{e^{(R(x,y_1)-R(x,y_2))/\beta}}_{\text{reward evidence}}.
$$

For example, suppose only two responses are possible and their teacher
probabilities are $0.8$ and $0.2$, so the teacher initially favors the first
response by odds $4{:}1$. If the rewards are respectively $0$ and $1$, with
$\beta=1$, the second response receives an evidence multiplier
$e\approx2.72$. Its posterior probability rises from $0.20$ to

$$
\frac{0.2e}{0.8+0.2e}\approx0.40.
$$

The reward evidence helps it, but does not erase the teacher's prior. A larger
reward gap or a smaller $\beta$ can overcome that prior. More generally:

- large $\beta$ makes $e^{R/\beta}$ closer to one, so $\pi^*$ stays near the
  teacher;
- small $\beta$ makes reward differences more decisive;
- adding the same constant to every reward changes no posterior, because the
  common multiplier cancels in $Z(x)$;
- a response with zero teacher probability retains zero posterior probability,
  just as Bayes' rule cannot recover a hypothesis excluded by the prior.

This explains “both signals live inside one distribution.” Before taking a
student gradient, the teacher probability and verifier reward are multiplied
and normalized into the single target $\pi^*$. Training can then use one
divergence toward that target. There is no independently scaled “reward loss”
gradient fighting a separate “teacher loss” gradient, even though identity
(1) shows that this posterior represents the same underlying regularized
objective.

For any policy $\pi$, expanding $\mathrm{KL}(\pi\Vert\pi^*)$ term by term gives
the exact identity

$$
J(\pi)=\beta\log Z(x)-\beta\,\mathrm{KL}\!\left(\pi\,\Vert\,\pi^*\right).
\tag{1}
$$

Maximizing $J$ is therefore *identical* to minimizing the reverse KL to the
posterior $\pi^*$. Everything hinges on **how the approach to $\pi^*$ is
executed** — which descent of which divergence.

### Estimator A (score function) = the additive method

Differentiating $J(\pi_\theta)$ directly and using
$\mathbb E[\nabla\log\pi_\theta]=0$:

$$
\nabla_\theta J
=\underbrace{\mathbb E_{y\sim\pi_\theta}\!\left[R\,\nabla_\theta\log\pi_\theta(y)\right]}_{\text{GRPO gradient}}
\;-\;\beta\,\underbrace{\nabla_\theta\mathrm{KL}(\pi_\theta\Vert\pi_T)}_{\text{OPD gradient}}.
$$

The displayed expectation is the ideal on-policy gradient. In actual GRPO, a
group of responses is sampled once from a frozen rollout policy
$\pi_{\mathrm{old}}$, and the optimizer may take several steps on that same
batch. At a visited prefix $s_t$, let $a=y_t$ be the sampled token. Then

$$
\rho_t
=\frac{\pi_\theta(a\mid s_t)}{\pi_{\mathrm{old}}(a\mid s_t)}
=\frac{p(a)}{p_0(a)}.
$$

This $\rho_t$ (Greek *rho*) is the **importance ratio**. It measures how the
current student probability of the sampled token compares with the probability
that generated the stored rollout:

- $\rho_t=1$: the two policies give that token the same probability;
- $\rho_t>1$: the current student now makes it more likely;
- $\rho_t<1$: the current student now makes it less likely.

For example, if the rollout policy assigned the sampled token probability
$0.20$ and the current student assigns it $0.24$, then
$\rho_t=0.24/0.20=1.2$.

At the first update on a freshly generated batch, normally
$\pi_\theta=\pi_{\mathrm{old}}$ and hence $\rho_t=1$. It departs from one as the
same batch is reused. The unclipped GRPO surrogate replaces an on-policy token
term by $\rho_t\hat A_i$; PPO-style clipping additionally prevents a batch from
rewarding an excessively large move in the intended direction. Thus $\rho$
does not describe reward quality or teacher agreement—it only accounts for the
current-policy/rollout-policy mismatch. If fresh rollouts were generated after
every parameter update, it would always be one when the batch is consumed.

The word **whitened** refers to how GRPO constructs $\hat A_i$. For the $G$
responses sampled for the same prompt, with rewards $R_1,\ldots,R_G$, a common
definition is

$$
\hat A_i
=\frac{R_i-\bar R}
       {\sqrt{G^{-1}\sum_{j=1}^G(R_j-\bar R)^2+\epsilon}},
\qquad
\bar R=G^{-1}\sum_{j=1}^G R_j.
$$

In other words, subtract the group's mean reward and divide by its standard
deviation. The resulting advantages have approximately mean zero and variance
one within the group: positive means “better than this prompt's sampled
siblings,” and negative means “worse.” For example, rewards $[1,1,0,0]$ become
advantages $[1,1,-1,-1]$ under the population-standard-deviation convention.
The same response-level $\hat A_i$ is copied to every token of response $i$.
If all rewards are equal, the group contains no relative preference, so
implementations normally produce zero advantages or skip it.

Whitening is also called **standardization** here. It does not make the rewards
white noise, and it does not impose a hard bound such as $[-2,2]$; it merely
puts typical advantages on an order-one scale. Exact values vary with group
size, the standard-deviation convention, and $\epsilon$.

With clipping inactive, differentiating the sampled token term with respect to
the current logits $z$ gives

$$
\nabla_z(\rho_t\hat A_i)
=\hat A_i\rho_t\,\nabla_z\log p(a)
=\hat A_i\rho_t(\mathbf 1_a-p).
$$

This explains every factor in the expression below: $\hat A_i$ gives the sign
and relative strength, $\rho_t$ corrects for batch reuse, and
$\mathbf 1_a-p$ is the direction that raises the sampled token and lowers its
alternatives. On a clipped plateau the corresponding reward gradient is zero.

The additive method (dGRPO) places this GRPO estimator beside a separate OPD
gradient. Compare their per-token logit-gradient magnitudes. The reward term
contributes

$$
\hat A_i\,\rho_t\,(\mathbf 1_a-p),\qquad
\text{typically order one near }\rho_t=1,
$$

while the analytic reverse-KL term contributes

$$
\beta\,p_v\!\left[\log\tfrac{p_v}{q_v}-\mathrm{KL}(p\Vert q)\right],
$$

whose scale is set by **log-ratios** — unbounded, and routinely tens of nats
early in training. The empirically observed KL domination is thus a property of
the *estimator*, not of the objective: identity (1) says the objective itself
is coherent.

### Estimator B (forward projection) = the proposal

By identity (1), maximizing $J$ *is* minimizing the reverse projection
$\mathrm{KL}(\pi_\theta\Vert\pi^*)$ — so the additive method is not a different
objective, it is the score-function descent of the reverse projection. The
proposal keeps the target $\pi^*$ but switches the projection geometry: match
$\pi^*$'s per-token conditionals with **forward** KL on visited prefixes.

Three facts make this precise rather than merely suggestive:

1. **Same fixed point.** If $\pi^*$ is realizable, both projections have the
   unique optimum $\pi_\theta=\pi^*$ (on prefixes with visitation mass). If it
   is not realizable, they differ in the usual way — reverse is mode-seeking,
   forward is mode-covering — which is a modeling choice, not an error.
2. **The forward direction is essential.** Here “direction” means the order
   of the two arguments to KL. To discuss the geometry without getting ahead
   of the construction, let $t$ denote any fixed (stop-gradient) target
   distribution at the current prefix. In this section the desired exact
   target is $t=\pi^*(\cdot\mid s)$; Step 2 derives its token-level form, and
   Step 3 later introduces the computable approximation called $\tilde q$.
   The two possible losses toward $t$ are

   $$
   L_{\mathrm{F}}=\mathrm{KL}(t\Vert p)
   \quad\text{and}\quad
   L_{\mathrm{R}}=\mathrm{KL}(p\Vert t).
   $$

   Both are nonnegative and both are minimized at $p=t$. Therefore the
   choice does not look important if one considers only the final optimum. It
   is important during training, however, because their gradients away from
   the optimum are different.

   For the **forward** direction, the part of the loss that depends on the
   student is just cross-entropy:

   $$
   L_{\mathrm{F}}
   =\underbrace{\sum_u t_u\log t_u}_{\text{constant in }\theta}
    -\sum_u t_u\log p_u.
   $$

   If $z_v$ is the student's logit for token $v$, the softmax identity
   $\partial(-\log p_u)/\partial z_v=p_v-\mathbf 1[u=v]$ gives

   $$
   \frac{\partial L_{\mathrm{F}}}{\partial z_v}=p_v-t_v.
   $$

   This gradient reads literally as “current probability minus desired
   probability.” Since both numbers lie in $[0,1]$, every coordinate lies in
   $[-1,1]$. A target probability of $10^{-6}$ rather than $10^{-9}$ cannot
   create an enormous force; in both cases the gradient is essentially just
   $p_v$ if the student assigns the token appreciable mass.

   For the **reverse** direction, the student distribution appears both as the
   weighting distribution and inside the logarithm:

   $$
   L_{\mathrm{R}}=\sum_u p_u\log\frac{p_u}{t_u},
   \qquad
   \frac{\partial L_{\mathrm{R}}}{\partial z_v}
   =p_v\!\left[
      \log\frac{p_v}{t_v}
      -\mathrm{KL}(p\Vert t)
    \right].
   $$

   Now a token that the student considers plausible but the target considers
   nearly impossible produces a large $\log(p_v/t_v)$. The gradient is
   therefore controlled not only by the probability gap, but also by how many
   orders of magnitude separate the probabilities.

   For example, consider a two-token vocabulary with

   $$
   p=(0.5,0.5),\qquad t=(10^{-6},1-10^{-6}).
   $$

   The forward-KL logit gradient is approximately $(0.5,-0.5)$. The reverse-KL
   logit gradient is approximately $(3.45,-3.45)$, and its magnitude keeps
   growing as the target's first probability approaches zero. Thus merely
   replacing $\mathrm{KL}(t\Vert p)$ by
   $\mathrm{KL}(p\Vert t)$ would turn the bounded probability target back into
   an unbounded log-ratio force—the same scale problem seen
   in Estimator A.

   That is, the stability argument specifically requires the forward
   ordering. Building a target in distribution space
   is not sufficient by itself; the training loss must also expose that target
   to the student through the bounded difference $p-t$. In Step 3, after
   $\tilde q$ is actually defined, this generic $p-t$ becomes
   $p-\tilde q$.
3. **Forward KL's usual risk is tempered here.** Mass-covering toward teacher
   modes the student cannot reach is the classic forward-KL objection; it is
   softened because the targets are evaluated only at prefixes the student
   itself visits (OPD's on-policy design), and the target is the teacher
   *conditioned at the student's own states*, not the teacher's marginal
   behavior.

This requires $\pi^*$'s per-token conditionals, derived next.

## Step 2: Exact Token-Wise Factorization of the Posterior

### What problem is this step solving?

Step 1 defines the ideal target

$$
\pi^*(y\mid x)\propto \pi_T(y\mid x)e^{R(x,y)/\beta},
$$

but this is a distribution over **complete responses** $y$. An autoregressive
language model does not emit a complete response in one decision. At each
prefix $s=(x,y_{<t})$, it must produce a distribution over the **next token**.
To train with token-level cross-entropy, we therefore need to answer:

> If complete responses were distributed according to $\pi^*$, what would the
> conditional distribution of their next token be at prefix $s$?

That desired conditional is $\pi^*(a\mid s)$. One could define it by summing
the sequence-level probabilities of every possible completion beginning with
token $a$, but the number of completions is exponential. Step 2 rewrites that
sum recursively and shows that the result has a simple form: start from the
teacher's next-token distribution and tilt each token according to the quality
of all complete responses reachable after it.

This step is still an **exact, conceptual construction**. Computing it would
require knowing the verifier reward for all relevant future completions. Step
3 will replace that unavailable quantity with evidence from sampled rollouts.

### First collect the reward-weighted mass below each prefix

Let $c$ denote a complete continuation from prefix $s$, sampled according to
the teacher; $sc$ is the completed response obtained by appending it. Define

$$
Z(s)
=\sum_{c}\pi_T(c\mid s)e^{R(sc)/\beta}
=\mathbb E_{c\sim\pi_T(\cdot\mid s)}\!\left[e^{R(sc)/\beta}\right].
$$

$Z(s)$ is the total **reward-weighted teacher mass** of all completions below
$s$. A subtree has a large $Z(s)$ when the teacher assigns substantial
probability to continuations with high verifier reward.

For a candidate next token $a$, define the corresponding mass after committing
to that token:

$$
Z(s,a)=
\begin{cases}
e^{R(sa)/\beta}, & a\ \text{terminates the response},\\[2pt]
Z(sa), & \text{otherwise}.
\end{cases}
$$

The masses obey

$$
Z(s)=\sum_a q(a\mid s)Z(s,a),
$$

because the teacher first chooses $a$ with probability $q(a\mid s)$ and then
chooses a continuation below $sa$.

The note expresses these positive masses on a logarithmic scale:

$$
Q^*(s,a):=\beta\log Z(s,a),
\qquad
V^*(s):=\beta\log Z(s).
$$

Substitution gives the backward recursion (terminal reward at EOS):

$$
Q^*(s,a)=
\begin{cases}
R(sa), & a\ \text{terminates},\\[2pt]
V^*(sa), & \text{else},
\end{cases}
\qquad
V^*(s)=\beta\log\sum_a q(a\mid s)\,e^{Q^*(s,a)/\beta}.
$$

These are called **soft values** because $V^*$ is a
$\beta$-log-sum-exp aggregation rather than a hard maximum. Their operational
meanings are:

- $Q^*(s,a)$: the reward-weighted quality of the entire future subtree reached
  after choosing token $a$;
- $V^*(s)$: the corresponding quality before choosing the next token, averaged
  over the teacher's choices in exponential space;
- $Q^*(s,a)-V^*(s)$: whether token $a$ leads to better or worse futures than
  the teacher-weighted baseline at prefix $s$.

Thus $Q^*-V^*$ is a token-level **soft advantage**. It is positive for a token
whose future subtree is better than the prefix baseline and negative for a
token whose subtree is worse.

### Convert those masses into the next-token target

Under the sequence-level posterior, the mass of all complete responses whose
next token is $a$ is the teacher's probability of choosing $a$, multiplied by
the reward-weighted mass below that choice. Normalizing across candidate tokens
therefore gives

$$
\begin{aligned}
\pi^*(a\mid s)
&=\frac{q(a\mid s)Z(s,a)}{Z(s)}\\
&=q(a\mid s)\,
  \exp\!\Big(\tfrac{Q^*(s,a)-V^*(s)}{\beta}\Big).
\end{aligned}
\tag{2a}
$$

This equation says exactly what “the teacher who saw the ending” means:

1. Begin with the teacher prior $q(a\mid s)$.
2. Increase it when choosing $a$ leads to above-baseline future outcomes.
3. Decrease it when choosing $a$ leads to below-baseline future outcomes.
4. Use $V^*(s)$ as the normalization that makes all adjusted probabilities
   sum to one.

Normalization can be checked directly:

$$
\sum_a\pi^*(a\mid s)
=e^{-V^*(s)/\beta}\sum_a q(a\mid s)e^{Q^*(s,a)/\beta}
=1.
$$

### A one-step example

Suppose only two next tokens are possible and either one terminates the
response. Let

$$
q(A\mid s)=0.75,\qquad q(B\mid s)=0.25,
$$

and let their verifier rewards be $R(sA)=0$ and $R(sB)=2$, with $\beta=1$.
The teacher prefers $A$, but $B$ leads to the better outcome. Their
unnormalized posterior masses are

$$
\begin{aligned}
A &: 0.75e^0=0.75,\\
B &: 0.25e^2\approx1.85.
\end{aligned}
$$

After normalization,

$$
\pi^*(A\mid s)\approx0.29,
\qquad
\pi^*(B\mid s)\approx0.71.
$$

The target does not discard the teacher: $B$ had to overcome its smaller
teacher prior. Nor does it ignore reward: the exponential reward tilt was
strong enough to reverse the ranking. For a nonterminal token, the same
calculation uses the reward-weighted mass of every possible future completion
below that token rather than one immediately observed reward.

### Why these token conditionals reproduce the sequence posterior

It remains to check that using (2a) at every position really gives the same
complete-response distribution as Step 1. Along a response $y$, multiply the
token conditionals:

$$
\prod_t \pi^*(y_t\mid s_t)
=\prod_t q(y_t\mid s_t)
 \exp\!\left(\frac{Q^*(s_t,y_t)-V^*(s_t)}{\beta}\right).
$$

For every nonterminal token,
$Q^*(s_t,y_t)=V^*(s_{t+1})$. Its positive value cancels the negative
$-V^*(s_{t+1})$ contributed at the next position. This is the “telescoping”:

$$
-V^*(x)
+\big[V^*(s_1)-V^*(s_1)\big]
+\big[V^*(s_2)-V^*(s_2)\big]
+\cdots+R(x,y).
$$

Only the initial $-V^*(x)$ and the terminal $Q^*=R(x,y)$ remain. Also,
$\prod_t q(y_t\mid s_t)=\pi_T(y\mid x)$, so

$$
\begin{aligned}
\prod_t \pi^*(y_t\mid s_t)
&=\pi_T(y\mid x)
  e^{\left(R(x,y)-V^*(x)\right)/\beta}\\
&=\frac{\pi_T(y\mid x)e^{R(x,y)/\beta}}{Z(x)}
=\pi^*(y\mid x),
\end{aligned}
$$

where $e^{V^*(x)/\beta}=Z(x)$. Therefore (2a) is not a heuristic token loss:
it is the exact autoregressive factorization of the whole-response target from
Step 1.

### What Step 2 contributes to the method

We can now replace ordinary OPD's teacher target $q(\cdot\mid s)$ with the
outcome-aware target $\pi^*(\cdot\mid s)$ and train on prefixes visited by the
student:

$$
L=\mathbb E_{s_t\sim d_{\pi_{\mathrm{old}}}}\Big[\mathrm{KL}\big(\pi^*(\cdot\mid s_t)\,\big\Vert\,\pi_\theta(\cdot\mid s_t)\big)\Big].
\tag{2}
$$

This has OPD's exact structure with $\pi_T$ replaced by $\pi^*$. Conceptually,
Step 2 establishes the ideal token target; computationally, it also exposes the
missing ingredient $Q^*-V^*$ that Step 3 must estimate from finite rollouts.

## Step 3: The Practical Surrogate

### What is the tilt vector?

At prefix $s_t$, Step 2's exact token target can be written as

$$
\pi^*(v\mid s_t)
\propto q(v\mid s_t)e^{\Delta_t^*(v)/\beta},
\qquad
\Delta_t^*(v):=Q^*(s_t,v)-V^*(s_t).
$$

$\Delta_t^*$ is a vector with one component for every vocabulary token $v$:

$$
\Delta_t^*
=\big(\Delta_t^*(v_1),\ldots,\Delta_t^*(v_{|\mathcal V|})\big).
$$

Each component answers a counterfactual question: “How good would the future
be, relative to the current-prefix baseline, if the next token were $v$?” A
positive component should raise that token above the teacher prior, a negative
component should lower it, and zero should leave its teacher weight unchanged
before normalization.

It is called a **tilt** vector because it tilts, or reweights, the teacher
distribution. Equivalently,

$$
\pi^*(\cdot\mid s_t)
=\operatorname{softmax}\!\left(
   \log q(\cdot\mid s_t)+\frac{\Delta_t^*}{\beta}
 \right).
$$

Thus the vector is simply an additive correction to the teacher logits. It is
not another model parameter and it is not a second loss. Only differences
between its components matter: adding the same constant to every component
would cancel in the softmax.

### Why can we observe only one component?

The exact vector $\Delta_t^*$ is unavailable because evaluating it would
require exploring the future after **every** possible next token. A sampled
rollout explores only one branch. If response $i$ contains token
$a=y_{i,t}$ at this prefix, the verifier eventually reveals how that one
chosen branch turned out, but it says nothing directly about the
counterfactual branches beginning with $v\ne a$.

GRPO summarizes the observed response outcome with the scalar group advantage
$\hat A_i$. Step 3 uses that scalar as a crude observation of the sampled
component $\Delta_t^*(a)$:

$$
\hat A_i\ \approx\ Q^*(s_t,a)-V^*(s_t).
$$

It then constructs the sparse estimate

$$
\widehat\Delta_t(v)
=\hat A_i\,\mathbf 1[v=a]
=
\begin{cases}
\hat A_i, & v=a,\\
0, & v\ne a.
\end{cases}
$$

Here $\mathbf 1[v=a]$ is an indicator: it equals one for the sampled token and
zero for every other token. For example, if the vocabulary at a position were
$[A,B,C]$, token $B$ was sampled, and $\hat A_i=1.2$, then

$$
\widehat\Delta_t=[0,1.2,0].
$$

This does **not** assert that the true advantages of $A$ and $C$ are zero. They
are unknown. Zero means “apply no verifier correction,” so those tokens retain
the teacher's relative preferences. In statistical language this is a sparse
shrinkage estimate: where direct evidence is absent, shrink back to the
teacher prior.

The same response-level $\hat A_i$ is used at every position in response $i$.
Consequently this is a cheap, biased credit assignment rule, not an exact
estimate of the Step 2 soft advantage. It inherits GRPO's assumption that all
tokens in a successful response deserve positive evidence and all tokens in a
failed response deserve negative evidence. Step 6 discusses how a value model
could supply more genuinely token-specific estimates. Whitening changes the
scale of $\hat A_i$, so $\beta$ acts as the conversion factor between its units
and a logit adjustment.

### How the estimated vector changes the teacher

Substituting $\widehat\Delta_t$ for $\Delta_t^*$ gives the practical target

$$
\tilde q(v)=\frac{q(v)\,e^{\hat A_i\mathbf 1[v=a]/\beta}}{Z_t},
\qquad
Z_t=1+q_a\!\left(e^{\hat A_i/\beta}-1\right),\quad q_a:=q(a).
$$

The formula is easier to read component by component:

$$
\tilde q(a)=\frac{q_a e^{\hat A_i/\beta}}{Z_t},
\qquad
\tilde q(v)=\frac{q(v)}{Z_t}\quad(v\ne a).
$$

The sampled token's teacher weight is multiplied by
$e^{\hat A_i/\beta}$; every other token's weight is multiplied by one. Their
total unnormalized mass is therefore

$$
q_a e^{\hat A_i/\beta}+(1-q_a)
=1+q_a\big(e^{\hat A_i/\beta}-1\big)=Z_t.
$$

After division by $Z_t$, the result is again a valid probability distribution.
If $\hat A_i>0$, the sampled token gains probability; if $\hat A_i<0$, it loses
probability; and if $\hat A_i=0$, then $\tilde q=q$. Among two unsampled tokens
$v$ and $w$, their relative odds remain exactly the teacher's:

$$
\frac{\tilde q(v)}{\tilde q(w)}=\frac{q(v)}{q(w)},
\qquad v,w\ne a.
$$

This is how the teacher answers “where should the probability removed from a
bad sampled token go?”: it is redistributed across the alternatives in the
teacher's existing proportions.

The training loss is (2) with $\tilde q$ in place of $\pi^*$ (targets are
stop-gradient, computed at rollout time):

$$
L=\mathbb E_{x,\{y_i\}\sim\pi_{\mathrm{old}}}\Big[\tfrac1G\sum_i\tfrac1{T_i}\sum_t
\mathrm{KL}\big(\tilde q^{(i,t)}\,\big\Vert\,\pi_\theta(\cdot\mid s_t)\big)\Big].
$$

**Implementation.** One-line change to an OPD trainer: add $\hat A_i/\beta$ to
the teacher's logit at the sampled token, re-softmax, cross-entropy. No second
loss, no routing, no ratio surgery.

**Intuition.** The verifier only ever has evidence about the token actually
taken; the teacher speaks about all the tokens *not* taken. The posterior
target stitches exactly those two together. GRPO suppresses a bad token without
saying where the freed mass should go; here the teacher answers that question
at every position.

## Step 4: Gradient Structure — the Exact Decomposition

### Why introduce $\kappa$?

Step 3 defines the target by exponentially reweighting the sampled token:

$$
\tilde q(a)=\frac{q_a e^{\hat A/\beta}}{Z_t},
\qquad
\tilde q(v)=\frac{q_v}{Z_t}\quad(v\ne a).
$$

This form is convenient for implementation, but it hides how far the target
actually moved from the teacher distribution $q$. Step 4 introduces $\kappa$
solely to expose that movement with one scalar.

For shorter notation, write

$$
u:=e^{\hat A/\beta},
\qquad
Z_t=1+q_a(u-1),
$$

and define

$$
\boxed{
\kappa:=\frac{q_a(u-1)}{Z_t}
=\frac{q_a\big(e^{\hat A/\beta}-1\big)}
       {1+q_a\big(e^{\hat A/\beta}-1\big)}
}.
$$

$\kappa$ is **not a new hyperparameter**. Once the advantage $\hat A$, the
temperature $\beta$, and the teacher probability $q_a$ are known, $\kappa$ is
fully determined. It is a more interpretable way to express the result of the
exponential reweighting and normalization.

### Deriving the mixture form

From this definition,

$$
1-\kappa
=1-\frac{q_a(u-1)}{Z_t}
=\frac{1}{Z_t}.
$$

Therefore, for any unsampled token $v\ne a$,

$$
\tilde q(v)=\frac{q_v}{Z_t}=(1-\kappa)q_v.
$$

For the sampled token,

$$
\begin{aligned}
(1-\kappa)q_a+\kappa
&=\frac{q_a}{Z_t}+\frac{q_a(u-1)}{Z_t}\\
&=\frac{q_a u}{Z_t}
=\tilde q(a).
\end{aligned}
$$

Combining both cases gives

$$
\tilde q=(1-\kappa)\,q+\kappa\,\mathbf 1_a .
\tag{3}
$$

Here $\mathbf 1_a$ is the one-hot distribution that places probability one on
the sampled token. Equation (3) is called an **affine combination** because its
coefficients sum to one. When $\kappa\in[0,1]$, it is also an ordinary convex
mixture between the teacher $q$ and the one-hot distribution $\mathbf 1_a$.
When $\kappa<0$, it extrapolates from $q$ in the direction away from
$\mathbf 1_a$; despite the negative coefficient, the original normalized form
guarantees that $\tilde q$ remains a valid distribution.

### What $\kappa$ measures

Subtracting $q$ from (3) yields the especially useful identity

$$
\boxed{\tilde q-q=\kappa(\mathbf 1_a-q).}
$$

Componentwise,

$$
\tilde q_a-q_a=\kappa(1-q_a),
\qquad
\tilde q_v-q_v=-\kappa q_v\quad(v\ne a).
$$

Thus $\kappa$ is the signed size of the verifier-induced probability transfer:

- If $\kappa>0$, the sampled token receives the fraction $\kappa$ of the
  probability mass it did not already have. Every alternative gives up the
  same fraction $\kappa$ of its teacher mass.
- If $\kappa=0$, no mass moves and $\tilde q=q$.
- If $\kappa<0$, mass is removed from the sampled token and distributed among
  the alternatives in their teacher proportions.

This is the “slider” interpretation. Zero is the untouched teacher. Moving
toward one approaches certainty on the sampled token. Moving below zero goes
in the opposite direction, away from that token.

For $0<q_a<1$ and $\beta>0$, the sign and limits follow from
$u=e^{\hat A/\beta}$:

$$
\begin{array}{c|c|c}
\text{advantage} & u & \kappa \\
\hline
\hat A>0 & u>1 & 0<\kappa<1 \\
\hat A=0 & u=1 & \kappa=0 \\
\hat A<0 & 0<u<1 & -\dfrac{q_a}{1-q_a}<\kappa<0.
\end{array}
$$

It also increases monotonically with the advantage:

$$
\frac{\partial\kappa}{\partial\hat A}
=\frac{q_a e^{\hat A/\beta}}{\beta Z_t^2}>0.
$$

As $\hat A\to+\infty$, $\kappa\to1$ and the target approaches
$\mathbf 1_a$. As $\hat A\to-\infty$, $\kappa\to-q_a/(1-q_a)$ and the sampled
token's target probability approaches zero. A negative $\kappa$ can have
magnitude greater than one when $q_a$ is large, but this does not mean more
than one unit of probability moves: the actual sampled-token change is
$\kappa(1-q_a)$, whose limiting value is $-q_a$.

For small $|\hat A|/\beta$,
$e^{\hat A/\beta}\approx1+\hat A/\beta$, so

$$
\kappa\approx q_a\frac{\hat A}{\beta}.
$$

This shows that the teacher's confidence also controls the initial response to
the verifier. If the teacher assigned the sampled token very little
probability, moderate evidence moves the target only a little; stronger
evidence is needed to overcome the prior.

### Numerical example

Suppose

$$
q=(0.5,0.3,0.2),
$$

the second token was sampled, and $e^{\hat A/\beta}=2$. Then

$$
Z_t=1+0.3(2-1)=1.3,
\qquad
\kappa=\frac{0.3}{1.3}\approx0.231.
$$

Direct exponential reweighting gives

$$
\tilde q
=\frac{(0.5,\,0.3\times2,\,0.2)}{1.3}
\approx(0.385,0.462,0.154).
$$

The mixture form gives exactly the same result:

$$
\tilde q
=(1-0.231)(0.5,0.3,0.2)+0.231(0,1,0).
$$

So $\kappa\approx0.231$ means that the target moved about $23.1\%$ of the way
from the teacher distribution toward certainty on the sampled token. Unlike a
hand-chosen interpolation weight, this value follows automatically from
$q_a$, $\hat A$, and $\beta$.

### Why this rewriting helps with the gradient

The forward-KL logit gradient is $p-\tilde q$. Using the identity above
separates it into the ordinary teacher-matching direction and the
verifier-induced displacement:

$$
\boxed{\;
\frac{\partial L_t}{\partial z}
=\underbrace{(p-q)}_{\text{pure OPD (forward-KL) gradient}}
\;-\;\underbrace{\kappa\,(\mathbf 1_a-q)}_{\text{REINFORCE-shaped verifier term}}
\;}
\tag{4}
$$

Equation (4) is the whole argument in one line:

- **Both pieces are bounded, with the same bound.** The OPD term is $p-q$, a
  difference of two distributions: elementwise in $(-1,1)$, total variation
  $\le1$. The verifier term is, by (3), exactly $q-\tilde q$ — *also* a
  difference of two distributions: elementwise in $[-1,1]$, and

  $$
  \mathrm{TV}(\tilde q,q)=|\kappa|(1-q_a)\;\le\;\max(q_a,\,1-q_a)\;\le\;1,
  $$

  saturating at $1-q_a$ as $\hat A\to+\infty$ (the verifier can at most grant
  the sampled token all remaining mass) and at $q_a$ as $\hat A\to-\infty$ (it
  can at most take all the sampled token's mass away). Note the bound is on the
  *displacement*, not on $\kappa$ itself — $\kappa$'s lower limit
  $-q_a/(1-q_a)$ diverges as $q_a\to1$, but the induced move on the simplex
  never exceeds TV $1$. Both signals are commensurate by construction; there is
  no scale on which one can dwarf the other. This is the structural fix for the
  additive method's imbalance. Moreover, norm domination cannot flip a
  direction even in principle: the gradient at each coordinate is the single
  fused quantity $p_v-\tilde q_v$, so there are no competing vectors whose
  relative magnitude decides anything.
- **The verifier term has GRPO's shape** $(\mathbf 1_a-q)$ — push up the
  sampled token, push down alternatives — except the subtractive distribution
  is the **teacher** $q$ rather than the student $p$: suppression is aimed
  where the teacher says mass should *not* go, and the freed mass lands on
  teacher-favored alternatives.
- **The effective advantage is $\kappa$, a saturating squash of $\hat A$**, not
  a free multiplier: it increases monotonically in $\hat A$
  ($\partial\kappa/\partial u=q_a/Z_t^2>0$), saturates at $1$ as
  $\hat A\to+\infty$ and at $-q_a/(1-q_a)$ as $\hat A\to-\infty$, and the
  displacement it induces obeys the TV bound above. "The verifier's influence
  is bounded" is thus a theorem, not a hope. Contrast with RLSD: there a
  positive weight $w_t$ rescales $\hat A$ by design choice; here the bounded
  modulation *follows directly from normalization*.

### The cross-entropy mixture view

Since $\mathrm{KL}(\tilde q\Vert p)$ and $\mathrm{CE}(\tilde q,p)=-\sum_v\tilde
q_v\log p_v$ differ by the $\theta$-independent entropy $H(\tilde q)$, the loss
can equally be written, using (3),

$$
\mathrm{CE}(\tilde q,p)
=(1-\kappa)\,\underbrace{\mathrm{CE}(q,p)}_{\text{distillation CE}}
\;+\;\kappa\,\underbrace{\big(-\log p_a\big)}_{\text{sampled-token NLL}}.
\tag{3$'$}
$$

**Intuition.** Practitioners already interpolate distillation and
SFT/REINFORCE-style NLL losses with hand-tuned coefficients. Here the
interpolation weights $(1-\kappa,\kappa)$ are *derived*, per token, from the
verifier's advantage and the teacher's confidence — and they sum to one, unlike
the additive method's free $(1,\beta)$ pair. For $\hat A<0$ the NLL coefficient
$\kappa$ is negative (an unlikelihood-flavored term), yet the total remains a
bona fide cross-entropy toward the genuine distribution $\tilde q$, which is
what keeps the gradient bounded — a property unlikelihood terms do not have on
their own.

### Consistency checks

- $\hat A=0$ (all-correct/all-wrong group): $\kappa=0$; the loss is exactly
  forward-KL OPD. A dense gradient survives precisely where GRPO has none —
  HDPO's routing, but as a smooth limit.
- $\pi_T\equiv\pi_\theta$ (teacher agrees with student): $p=q$, the OPD term
  vanishes and (4) reduces to $-\kappa(\mathbf 1_a-p)$ — pure policy gradient
  with squashed advantage $\kappa(\hat A)$. The method degenerates to
  bounded-advantage RL exactly when the teacher has nothing to add.
- $\beta\to\infty$: $\kappa\to0$, pure OPD.
- $\beta\to0$ with $\hat A>0$: $\tilde q\to\mathbf 1_a$, pure cross-entropy on
  the sampled token.
- Fixed point of each token term is $p=\tilde q$, which carries a **support
  floor**: for $\hat A\ge0$, (3) gives $\tilde q_v\ge(1-\kappa)q_v$ for every
  $v$, and for $\hat A<0$ every non-sampled token *gains* mass
  ($\tilde q_v=q_v/Z_t>q_v$) while the sampled token keeps
  $\tilde q_a=q_au/Z_t>0$. Since $\kappa<1$ strictly for finite
  $\hat A/\beta$, no token's target is ever zeroed, and the target's entropy
  stays bounded away from zero unless $|\hat A|/\beta\to\infty$ persistently.
  Unlike GRPO's $(\mathbf 1_a-p)$ updates, the loss cannot collapse the policy
  to a delta; the teacher acts as a per-prefix entropy floor for free.

## Step 5: Who Controls the Direction — an Explicit Arbitration Rule

At the sampled token, the update raises its logit iff $\tilde q_a>p_a$.
Solving $u\,q_a/Z_t>p_a$:

$$
\text{reinforce } y_t
\quad\Longleftrightarrow\quad
\frac{\hat A_i}{\beta}\;>\;\operatorname{logit}(p_a)-\operatorname{logit}(q_a),
\qquad \operatorname{logit}(x)=\log\tfrac{x}{1-x}.
\tag{5}
$$

**Intuition.** Direction is decided by comparing the verifier's advantage (in
units of $\beta$) against the **log-odds gap between student and teacher**. The
verifier can overrule the teacher, but must "pay" $\beta$ nats of tilt per nat
of teacher–student log-odds disagreement; symmetrically the teacher can soften
but not indefinitely resist a strong advantage. Authority is negotiated at an
explicit exchange rate — as opposed to additive (whoever has the bigger
gradient norm wins), TRRD/RLSD (verifier always wins), or naive OPD (teacher
always wins).

$\beta$ is dimensionless-interpretable: $\hat A$ is whitened ($O(1)$) and
log-odds are in nats, so $\beta\approx1$ means "one standard deviation of group
advantage buys one nat of log-odds against the teacher." A sensible default
range is $\beta\in[0.5,2]$, and $\beta$ can be auto-set by matching the average
tilt $|\hat A|/\beta$ to the average teacher–student disagreement
$|\log(q_a/p_a)|$ in the batch.

## Interlude: Is This RL? The Two Limits Are Not Symmetric

Step 4's consistency checks established one retraction exactly:
$\hat A=0\Rightarrow\tilde q=q$, so zero advantage gives pure OPD. The
symmetric question — does a *rich* verifier signal recover the original RL
update? — has a more interesting answer: **no**, at three successively deeper
levels, and the failure at the deepest level is a design property with its own
control parameter.

### In gradient form: yes

With an uninformative teacher ($\pi_T=\pi_\theta$, so $q=p$), eq. (4) reduces
to $-\kappa(\mathbf 1_a-p)$ — exactly the GRPO per-token gradient
$-\hat A(\mathbf 1_a-p)$ with the advantage replaced by its squashed version
$\kappa$. Structurally the method *is* policy gradient with a smooth trust
region, plus a distillation pull $(p-q)$ when the teacher disagrees. RL's
gradient geometry is inherited.

### In the strong-signal limit: no — it becomes reward-filtered imitation

As $\hat A/\beta\to+\infty$, $\kappa\to1$ and the loss saturates at
$-\log p_a$ with **unit weight**: the advantage's magnitude beyond saturation
is discarded. The strong-verifier limit is therefore not GRPO (linear in
$\hat A$) but **RAFT / best-of-$n$-style imitation** of successful tokens.
Symmetrically, $\hat A/\beta\to-\infty$ gives $Z_t\to1-q_a$ and

$$
\tilde q_v\;\longrightarrow\;\frac{q_v\,\mathbf 1[v\neq y_t]}{1-q_a},
$$

teacher-shaped unlikelihood: cross-entropy toward the teacher with the sampled
token excised and the freed mass redistributed in the teacher's proportions.
Two consequences:

- The update is **not an unbiased estimator of $\nabla\mathbb E[R]$ in any
  regime**: $\kappa$ is nonlinear in $\hat A$, so even the pure-PG part
  estimates a projection step, not the reward gradient. GRPO's linear
  advantage weighting is exactly what made it unbiased; the tilt trades
  unbiasedness for boundedness — the same trade PPO's clip makes, but smooth
  and prior-aware.
- In the *practical* regime the saturation is not binding: whitened advantages
  live in roughly $[-3,3]$, and with $\beta\in[0.5,2]$, $\kappa$ sits on the
  responsive part of its curve ($q_a=0.3$, $\beta=1$: $\hat A=2\Rightarrow
  \kappa\approx0.66$; $\hat A=3\Rightarrow\kappa\approx0.85$). The degenerate
  imitation limit is reached only by driving $\beta\to0$.

### In its asymptotics: no — a fixed prior is a ceiling

The deepest difference. KL-regularized RL is **iterated tilting of a moving
prior**: policy iteration sets $\pi_{k+1}\propto\pi_k\,e^{A_k/\beta}$, and
because the anchor moves each round the tilts compound — in the sequence-level
bandit view the baselines cancel into $Z$ and $\pi_k\propto\pi_0\,e^{kR/\beta}$
— so the process *anneals onto the reward maximizer*. The proposal is
**repeated projection onto a tilt of a *fixed* prior**: a baseline shift
changes only $Z$, and whitening changes only the effective temperature
$\beta\sigma_x$, so at every iteration the target stays inside the
one-parameter family $\pi_T\,e^{R/(\beta\sigma_x)}/Z$ — a tempered posterior
of the *same* teacher. Rich verifier signal speeds convergence *to* that
posterior; it never moves the student *past* it. The objective says so
directly: reward may buy at most $R/\beta$ nats of departure from the teacher.

This ceiling is not a hidden defect but a stated modeling choice — and it is
the same ceiling the literature measures from the other side: Sang et al.'s
condition C1 (a teacher that is not itself reward-shaped underperforms direct
GRPO) and ExOPD's extrapolation exist precisely because a fixed teacher anchor
bounds reward-driven departure from the teacher.

### The $\alpha$ parameter: from one Bayes update to full iterated RL

The geometric mixture prior (which reappears in Step 6 as an exploration
mitigation) is in fact the parameter that controls the asymptotics:

$$
q\;\propto\;\pi_T^{\alpha}\,\pi_{\mathrm{old}}^{\,1-\alpha}.
\tag{6}
$$

- **$\alpha=1$:** fixed teacher prior — one exact Bayes update per round,
  teacher-anchored, bounded; the tempered posterior is the terminus.
- **$\alpha=0$:** the prior *is* the old policy, and at the objective level the
  method becomes exactly **AWR / mirror descent** — genuine iterated RL that
  converges toward reward maximization (with $\kappa$-squashed steps; the
  one-coordinate surrogate estimator is unchanged).
- **$0<\alpha<1$:** the anchor tracks the student at rate $1-\alpha$ while
  staying tethered to the teacher — reward-following where the verifier keeps
  paying, teacher-regularized where it does not.

$\beta$-annealing also reaches reward-only asymptotics, but through the
saturated RAFT-style updates above; $\alpha$ is the more principled control
because it changes *whose posterior is computed*, not how sharply. A natural
schedule follows: $\alpha$ high early (the teacher supplies most of the
signal while the student cannot yet earn reward), decayed as group success rates
rise. Notably, this *derives* the anneal-to-zero schedules that the additive
family (KDRL, TGPO, SDPG, CEPO) converged to by hand-tuning.

### Dense verifiers only help

If "rich" means *dense* (per-step verifier, PRM) rather than *large*, the
framework improves rather than degrades: the ideal target tilts each token by
its own value increment (Step 6), the banked-credit bias vanishes, and the
method moves closer to well-credited RL, not further from it.

**Summary.** The method is OPD when the advantage is zero, soft-clipped
policy gradient when the teacher is uninformative, and in between it is one
exact Bayes update against a fixed teacher prior — it inherits RL's gradient
geometry but not RL's asymptotics, and $\alpha$ in (6) is the parameter moving from "one
posterior update" ($\alpha=1$) to "full iterated RL" ($\alpha=0$).

## Step 6: What the Surrogate Gets Wrong (Bias Accounting)

Comparing $\widehat\Delta_t$ to the exact tilt $Q^*-V^*$:

1. **Unsampled-token advantages are set to zero.** True
   $Q^*(s_t,v)-V^*(s_t)\neq0$ for $v\neq y_t$; the surrogate substitutes the
   teacher prior. The error is largest where the teacher misranks alternatives
   that the verifier would distinguish — invisible to any method that only
   rolls out once per prefix.
2. **The sampled-token tilt re-applies banked credit.** Token appending is a
   deterministic transition, so the hard action value satisfies
   $Q(s_t,y_t)=V(s_{t+1})$ *exactly*, and per-prefix hard advantages telescope
   just like the exact soft tilts:

   $$
   \sum_t\big[V(s_{t+1})-V(s_t)\big]=R-V(x).
   $$

   The ideal per-position tilt is therefore the value **increment**
   $V(s_{t+1})-V(s_t)$ — how much this move shifted the evaluation bar, in
   the chess picture of Step 0. GRPO's $\hat A_i$ instead has conditional
   mean

   $$
   \mathbb E\big[\hat A_i\mid s_t,y_t\big]
   \;\propto\;V(s_{t+1})-V(x)
   \;=\;\underbrace{V(s_{t+1})-V(s_t)}_{\text{ideal increment}}
   +\underbrace{V(s_t)-V(x)}_{\text{credit already banked}},
   $$

   i.e., credit earned by earlier tokens is re-applied at every later
   position. In a score-function policy gradient this substitution is
   harmless — baselines never bias a linear estimator — but here the tilt
   enters the target *nonlinearly* (inside $\exp$ and the normalizer), so it
   is a genuine bias: on a trajectory whose fate is already sealed
   ($V(s_t)\approx R$, ideal increment $\approx0$), late tokens still receive
   the full tilt and get over-reinforced or over-suppressed relative to the
   exact posterior. It also estimates a hard (mean) advantage under
   $\pi_{\mathrm{old}}$ rather than the soft ($\beta$-log-sum-exp) advantage
   under $\pi_T$, and whitening changes units ($\beta$ absorbs the scale but
   not the position-dependence).

Both are estimator deficiencies, not framework deficiencies: the loss accepts
any tilt vector $\widehat\Delta_t(v)$. A value head supplying increment
estimates $\hat V(s_{t+1})-\hat V(s_t)$ fixes the banked-credit bias;
teacher-completion counterfactuals or prefix-tree Monte Carlo estimates densify
the tilt over alternatives. Either slots in without changing anything above —
the target just becomes a (better-)tilted teacher. Step 8 turns both upgrades
into concrete algorithm variants (v2 and v3), with the critic's cold start
solved by the teacher-potential warm start of `ppo_opd_value_bridge.md`.

Relatedly, the one-hot $\mathbf 1[v=y_t]$ is a property of the **estimator**,
not the objective: the exact target (Step 2) tilts every token, and the
surrogate concentrates on the sampled one only because that is where the
verifier's evidence sits. No discrete routing decision is ever made on the
loss, so this does not reintroduce hard token selection.

One genuine structural caveat from (4): the verifier's per-step leverage is
mediated by $q_a$ (for moderate $\hat A/\beta$,
$\kappa\approx q_a(e^{\hat A/\beta}-1)$). If the teacher assigns the sampled
token tiny mass, a single update moves it little — the posterior view says a
flat-prior region needs accumulated likelihood. If that suppresses exploration
too much, widen the prior to the geometric mixture
$q\propto\pi_T^\alpha\pi_{\mathrm{old}}^{1-\alpha}$ of eq. (6) — TRRD's anchor
$m$, reused as a *prior* rather than a ratio denominator, and per the Interlude
also the parameter that controls whether the method behaves as a one-shot posterior
or as iterated RL — or use ExOPD-style extrapolated teacher logits.

**Off-policy reuse.** The target $\tilde q$ is fixed at rollout time, so
multi-epoch reuse behaves like SFT regression; a clipped $\rho$ weight can be
added for PPO-style conservatism, but the update direction no longer depends
on it.

## Step 7: Relation to Existing Methods

- **SD-ZERO:** the tilted target is an *analytic* outcome-conditioned teacher.
  SD-ZERO learns a reviser $\pi_T(a\mid s,\text{outcome})$ in a separate
  training phase; Bayes' rule says
  $\pi_T(a\mid s,\text{outcome})\propto\pi_T(a\mid s)\cdot\text{likelihood}$,
  and $e^{\hat A/\beta}$ plays the likelihood-ratio role in closed form, with
  no phase-1 training.
- **HDPO:** its hard routing (distill only when GRPO has zero gradient) emerges
  here as the smooth $\hat A=0$ limit.
- **TRRD:** its geometric anchor reappears as an optional prior-widening
  device, not as a ratio denominator.
- **RLSD/RLRT:** the saturating $\kappa$ is the principled counterpart of their
  hand-designed positive weights — except it is signed, bounded, and derived
  from normalization rather than chosen.
- **ExOPD:** extrapolated teacher logits compose directly with the tilt as an
  alternative prior.
- More broadly this is the control-as-inference posterior with a teacher prior
  (advantage-weighted / soft-Q family), applied per-prefix on student rollouts.
  The switch from score-function descent of the reverse projection to a
  forward per-prefix projection is the same move DPO makes relative to
  PPO-RLHF: same optimum, different geometry.

## Step 8: The Algorithm

Everything above compresses into a short procedure. Three versions, ordered by
commitment. **v2 and v3 change only where the per-position credit comes
from** — the target construction, the loss, rule (5), and the $\beta$
semantics are identical across all three.

### v1 — minimal (critic-free, GRPO-level cost)

Hyperparameters: temperature $\beta$ (default $1$, range $[0.5,2]$, or
auto-set per Step 5); prior mixture $\alpha$ (eq. (6), default $1$); group
size $G$.

Per iteration:

1. **Rollouts.** Sample $G$ responses per prompt from
   $\pi_{\mathrm{old}}=$ current student.
2. **Credits.** Verifier scores each response; whiten within the group
   $\to\hat A_i$ (all-equal groups get $\hat A_i=0$, i.e. pure OPD there).
3. **Prior.** One frozen teacher forward pass per rollout gives
   $q(\cdot\mid s_t)$ at every prefix; with $\alpha<1$, use
   $q\propto\pi_T^\alpha\pi_{\mathrm{old}}^{1-\alpha}$ renormalized.
4. **Target (the tilt).** At each position, add $\hat A_i/\beta$ to the
   *sampled token's* teacher logit and re-softmax:

   $$
   \tilde q_t(v)\;=\;\frac{q(v)\,e^{\hat A_i\mathbf 1[v=y_t]/\beta}}{Z_t},
   \qquad Z_t=1+q_a\big(e^{\hat A_i/\beta}-1\big),
   $$

   and stop-gradient the target.
5. **Loss.** $L(\theta)=\sum_{i,t}\mathrm{CE}\big(\tilde q_{i,t},\,
   p_\theta(\cdot\mid s_{i,t})\big)$. For multi-epoch reuse, an optional
   clipped $\rho_t$ weight adds PPO-style conservatism (per Step 6, the
   update direction does not depend on it).
6. **Step.** One optimizer step; refresh $\pi_{\mathrm{old}}$ on the usual
   cadence.

Reference implementation is a few lines on top of any OPD loop:

```python
t_logits = teacher(x, y).logits              # [T, V], frozen
t_logits[arange(T), y] += A_hat / beta       # one scalar per rollout (v1)
q_tilde  = softmax(t_logits, -1).detach()    # the tilted target
loss = -(q_tilde * log_softmax(s_logits, -1)).sum(-1).mean()
```

Monitoring: the $\kappa$ histogram (should sit in the responsive zone, not
saturated); $\mathrm{TV}(\tilde q,q)=|\kappa|(1-q_a)$; the fraction of
positions where rule (5) flips the teacher's direction; entropy at
high-entropy (fork) positions.

### v2 — value-increment tilt (teacher-warm-started critic)

The PPO–OPD bridge (`ppo_opd_value_bridge.md`) makes Step 6's "value head
increments" fix implementable. Full spec:

**Architecture and indexing.** A small MLP on the shared trunk's last hidden
states, detached from the trunk (or given a small auxiliary coefficient) to
avoid representation interference. Standard decoder indexing: $h_t$ has
consumed $y_{\le t}$, so

$$
V_\phi(s_{t+1})=\mathrm{MLP}(h_t),\qquad
V_\phi(x)=\mathrm{MLP}(h_{\text{last prompt pos}}),
$$

and the increment for token $y_t$ is $\mathrm{MLP}(h_t)-\mathrm{MLP}(h_{t-1})$
— consecutive positions, one forward pass. The terminal value is pinned,
$V(s_T):=R$, so the last increment is $R-V_\phi(s_{T-1})$.

**Parameterize in probability units.** For binary rewards, have the head
output $p_\phi(s)\in[0,1]$ (sigmoid) and define
$V_\phi=\beta\log\!\big(1+p_\phi\,(e^{1/\beta}-1)\big)$ via the link (bridge
note, eq. (6)). Targets are then bounded and the soft-vs-hard calibration is
built into the parameterization rather than patched afterward.

**Warm start as a byproduct, not a phase.** The teacher potential is a
cumulative sum of log-ratios already gathered for the prior:

$$
\widehat V_T(s_t)\;=\;\hat c_x+\beta\sum_{j<t}\log
\frac{\pi_T(y_j\mid s_j)}{\pi_{\mathrm{ref}}(y_j\mid s_j)},
\qquad
\hat c_x=\beta\log\!\big(1+\hat p_x(e^{1/\beta}-1)\big)
$$

with $\hat p_x$ the group success rate and $\pi_{\mathrm{ref}}=$ the teacher's
pre-RL checkpoint if available, else the student init (bridge note, eq. (5)).
Invert the link to teacher-implied success probabilities $\hat p_T(s_t)$, clip
to $[\varepsilon,1-\varepsilon]$, and regress $p_\phi$ on the **mixed target**

$$
p^{\mathrm{tgt}}(s_t)\;=\;(1-w)\,\hat p_T(s_t)\;+\;w\,\mathbf 1[R_i=1],
\qquad
w:\ \text{small}\ \longrightarrow\ 1\ \text{annealed by explained variance.}
$$

This unifies "Phase 0, then returns" into a single annealed regression:
$w\approx0$ is the warm start, $w\to1$ is the transient-prior guarantee
(bridge note, Step 3 — teacher error in the critic must decay, so the
teacher's weight in the mix may not persist).

**The tilt via GAE — v1 is a corner case of v2.** With $\gamma=1$,
terminal-only reward, and $\delta_t=V(s_{t+1})-V(s_t)$, define the per-position
credit $\mathrm{GAE}_\lambda(t)=\sum_l\lambda^l\,\delta_{t+l}$, whiten across
the batch, and use it as the tilt:

- $\lambda=1$ telescopes to $R-V(s_t)$; with an untrained or constant critic,
  whitening reduces this to **exactly v1's $\hat A_i$**;
- $\lambda=0$ gives the pure one-step increments.

So the implementation is one code path,
$\text{tilt}_t=\mathrm{whiten}\big(\mathrm{GAE}_\lambda(t)\big)/\beta$, with
v1 the $(\lambda{=}1,\,V{\equiv}\text{const})$ corner and $\lambda$ scheduled
downward as the critic's explained variance on returns rises. There is no
algorithm switch — one parameter anneals as the critic's explained variance
rises.
Everything downstream — $\tilde q$, $\kappa$, eq. (4), rule (5) — keeps its
exact form with the credit substituted.

**Monitoring.** Explained variance of $V_\phi$ vs returns (drives both $w$ and
$\lambda$); and the free diagnostic — the persistent residual between
teacher-potential increments and TD residuals is a per-token map of teacher
bias; fork positions should show large residuals (an online measurement of
the fork-suppression effect).

v2 removes the banked-credit bias (Step 6, bullet 2) and gives fork tokens
per-token credit; the cost is the value head plus one frozen reference forward
pass per batch.

### v3 — dense Q-head tilt (research extension)

Identity (2) of the bridge note supplies advantage estimates for *every*
vocabulary entry: $\beta\log\frac{\pi_T(v\mid s)}{\pi_{\mathrm{ref}}(v\mid s)}$
is a per-vocab $Q-V$ estimate. v3 refines these on verifier data and tilts
densely, attacking the remaining bias (Step 6, bullet 1: unsampled-token
zeros), which no sampled-token estimator can touch.

**Head: residual parameterization, not an LM-head copy.**

$$
\frac{Q_\phi(s,v)}{\beta}
\;=\;\eta\,\big(z_T(v\mid s)-z_{\mathrm{ref}}(v\mid s)\big)\;+\;g_\phi(s,v),
\qquad g_\phi\ \text{zero-initialized},
$$

where $z_T,z_{\mathrm{ref}}$ are teacher and frozen-reference logits. The
zero-init of $g_\phi$'s output layer *is* the initialization: the head equals
the teacher-implied advantages exactly at start, with no pretraining, and
$z_T$ is already computed for the prior — only the frozen reference pass is
extra. The tempting alternative — initializing a Q-head by copying the model's
output (unembedding) weights — outputs the *student's logits*, which are
advantage-like only against a uniform reference and double-count the reference
when $\pi_{\mathrm{ref}}$ is the student init; avoid it as a Q init. Where the
output weights *are* the right reuse: as a feature basis for the correction.
Keep $g_\phi$ low-rank, $g_\phi(s,v)=u_\phi(h_s)^{\!\top}B_v$ with rank
$r\ll d$ ($B$ initializable from the top right-singular directions of the
unembedding). Task credit is far lower-dimensional than language modeling —
small $r$ is a feature, not a compromise.

**Training targets — the mixed-target discipline.** Which component of the
verifier/teacher mix may *persist* depends on the coordinate:

- **Sampled $(s_t,y_t)$:** the deterministic-transition identity makes the TD
  target exact, with no bootstrap approximation:
  $Q(s_t,y_t)=V(s_{t+1})$, so the regression target is
  $\mathrm{sg}[V_\phi(s_{t+1})]$ from the v2 head ($R$ at the terminal). The
  teacher's weight in the mix must **decay** (annealed by explained variance):
  a permanent teacher component here means the critic never converges to the
  true value and the tilt double-counts the teacher.
- **Unsampled $(s_t,v\neq y_t)$:** a masked L2 pull $g_\phi\to0$,
  **persistent** — no verifier signal ever arrives at these coordinates, and
  the teacher anchor is the only thing preventing a head with no training
  signal from drifting as trunk features shift.

**Tilt.** Per-position constants absorb into the softmax, so no $V$
subtraction is needed; the tilted target logits are

$$
z_{\tilde q}\;=\;z_T\;+\;\eta\,(z_T-z_{\mathrm{ref}})\;+\;g_\phi,
$$

renormalized. **The $\eta$ parameter and the extrapolation fact:** at
$g_\phi=0,\ \eta=1$ the target logits are $2z_T-z_{\mathrm{ref}}$, i.e.
$\tilde q\propto q^2/\pi_{\mathrm{ref}}$ — exactly ExOPD's extrapolated
teacher. The anchor at full strength therefore presumes teacher
soft-optimality (Sang's condition C1 is the caveat) and *starts* by
extrapolating it; $\eta=0$ starts at pure OPD and learns the tilt from
verifier data alone; intermediate $\eta$ trades warm-start speed against init
bias. No whitening in v3 — the TD regression in reward units sets the scale,
and $\beta$ keeps its exchange-rate role.

**Cost over v2.** One frozen reference forward pass per batch, the
$[d\times r]+[V\times r]$ correction head, and the masked-L2 term. No extra
rollouts, no counterfactual sampling.

**Honest caveat.** Unsampled-coordinate credit is still teacher-implied,
refined only where sampling reaches; the evidence's reach widens only insofar
as the trunk generalizes $g_\phi$ across coordinates. Genuinely *measured*
credit for never-sampled alternatives needs counterfactual rollouts — v3 is
the cheap approximation of that, not the thing itself.

### Choosing a version

| Version | Use when | Extra cost over GRPO + OPD |
|---|---|---|
| v1 | default first experiment; every structural claim (boundedness, arbitration, $\hat A=0\to$ OPD) is testable here | none |
| v2 | long-horizon credit is the binding bias: thinking models, fork-suppression regime, long rollouts | value head + one frozen reference pass (warm start is a byproduct); v1 is its $(\lambda{=}1, V{\equiv}\text{const})$ corner |
| v3 | the teacher misranks alternatives that the verifier would distinguish | low-rank Q-correction head + reference pass |

Schedules, from the Interlude: keep $\alpha=1$ for a teacher-anchored run
(one-shot posterior); decay $\alpha\to0$ as group success rates rise if the
goal is to pass the teacher (mirror-descent limit). $\beta$-annealing also
sharpens the verifier's authority, but drives updates toward saturated
RAFT-style imitation — prefer $\alpha$.

## Summary Against the Three Objections

| Objection | Mechanism that answers it |
|---|---|
| Additive: KL gradient dominates GRPO | One loss; by (4) both signals enter as differences of distributions, each with TV $\le1$; the verifier's displacement obeys $\mathrm{TV}(\tilde q,q)\le\max(q_a,1-q_a)$; identity (1) shows the same optimum is reached via a different projection geometry |
| TRRD: discards dense signal | Full-vocabulary teacher distribution in the target at every prefix; $\hat A=0$ limit is exactly OPD |
| RLSD: teacher reduced to a positive weight | Teacher enters as the prior of a posterior, contributes the signed distributional term $(p-q)$, supplies the baseline in $(\mathbf 1_a-q)$, and can flip the sampled token's update direction per rule (5) — but only within the bound set by $\beta$ |

No hard selection anywhere: $\tilde q$, $\kappa$, and rule (5) are smooth in
$\hat A$, $q$, $p$, $\beta$.

**One-sentence summary.** Stop summing a reward gradient and a KL gradient;
distill toward the single distribution the objective already says is optimal —
the teacher tilted by the verifier's advantage. The observed imbalance is an
artifact of descending the reverse projection by score function, not of the
objective; the forward per-prefix projection shares its optimum (under
realizability) and has bounded gradients by construction.

## Appendix: Verified Claims

The following were checked numerically (5000 random $(q,p,a,\hat A,\beta)$
draws, plus a random depth-4 ternary tree for the sequence-level claims):

- $Z_t=1+q_a(u-1)$ matches the logit-shift implementation.
- Decompositions (3), (3$'$), and (4) hold exactly.
- $\kappa\in\big(-\tfrac{q_a}{1-q_a},\,1\big)$, increasing in $\hat A$.
- The verifier term $q-\tilde q$ is elementwise in $[-1,1]$;
  $\mathrm{TV}(\tilde q,q)=|\kappa|(1-q_a)\le\max(q_a,1-q_a)$.
- Arbitration rule (5) predicts $\operatorname{sign}(\tilde q_a-p_a)$ exactly.
- Support floor: $\tilde q_v\ge(1-\kappa)q_v$ for $\hat A\ge0$; non-sampled
  tokens gain mass for $\hat A<0$.
- Soft-value factorization: the per-prefix conditionals
  $q\,e^{(Q^*-V^*)/\beta}$ are normalized, their product recovers
  $\pi^*(y\mid x)$ leaf-exactly, $e^{V^*(x)/\beta}=Z(x)$, and the per-position
  tilts telescope to $R-V^*(x)$.
