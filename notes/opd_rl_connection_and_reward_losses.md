# From GRPO to Teacher-Guided RL and Distillation Losses

This note compares the objectives used by the papers in
`papers/opd_rl_connection_and_reward`. The common theme is the tension between
dense teacher supervision and reward-grounded policy optimization.

## Reading Path

The easiest way to understand these objectives is to add one idea at a time:

1. Start with GRPO: reward decides whether an entire response should become more
   or less likely.
2. Add OPD: a teacher supplies dense next-token judgments, but now the teacher
   can determine the update direction.
3. Combine the two signals additively: reward and teacher each contribute an
   independent gradient.
4. Integrate the teacher into GRPO: TRRD changes the update's guardrails, while
   RLSD changes token-level credit magnitude.
5. Route, gate, debias, extrapolate, or cache the teacher signal when the basic
   forms are too indiscriminate or expensive.

## Step 1: GRPO—One Outcome, One Advantage for the Whole Response

For a prompt $x$, GRPO samples a group of responses
$\{y_1,\ldots,y_G\}$ and assigns each completed response a verifier reward
$R(x,y_i)$. It first converts that raw reward into a relative advantage:

$$
A_i=\frac{R(x,y_i)-\operatorname{mean}_{G}R}
          {\operatorname{std}_{G}R}.
$$

Thus $A_i>0$ means “better than the other sampled responses for this prompt,”
while $A_i<0$ means “worse.” If every response receives the same reward, there
is no within-group preference; implementations handle the zero standard
deviation by producing zero advantages or skipping the group. This is why GRPO
has no reward gradient on an all-correct or all-wrong group.

GRPO then compares the current probability of each sampled token with the
probability under the policy that generated the rollout:

$$
\rho_{i,t}
=\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}
       {\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,<t})}.
$$

Ignoring clipping for a moment, the token contribution is simply
$\rho_{i,t}A_i$. A positive $A_i$ increases the probability of every sampled
token in response $i$; a negative $A_i$ decreases it. PPO-style clipping limits
how far a single update can change the ratio. Written as an objective to
maximize:

$$
J_{\mathrm{GRPO}}
=\mathbb E_{x,\{y_i\}\sim\pi_{\mathrm{old}}}
\left[
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}\sum_{t=1}^{T_i}
\min\!\left(
  \rho_{i,t}A_i,
  \operatorname{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)A_i
\right)
\right].
$$

This is the common response-normalized form. Some implementations average over
all non-padding tokens in the batch instead of using $1/T_i$ per response, and
some add a separate reference-policy KL penalty. Those choices affect length
weighting or regularization, but the clipped reward surrogate analyzed below is
the same. The expression above is a **maximization objective**. Code that is
written in terms of a loss to minimize uses $L_{\mathrm{GRPO}}=-J_{\mathrm{GRPO}}$;
the optimization direction is identical.

### Reading the loss one token at a time

Define the contribution of token $t$ in response $i$ as

$$
\ell(\rho,A)
=\min\!\left(
  \rho A,
  \operatorname{clip}(\rho,1-\varepsilon,1+\varepsilon)A
\right).
$$

The full objective just averages and sums these token contributions:

| Part | Operational meaning |
|---|---|
| $\mathbb E$ | Average over prompts and response groups sampled from $\pi_{\mathrm{old}}$. |
| $G^{-1}\sum_i$ | Give each sampled response an equal group weight. |
| $T_i^{-1}\sum_t$ | Average token contributions within response $i$, so a longer response does not automatically receive more total weight. |
| $A_i$ | A response-level signed label: reinforce if positive, penalize if negative. |
| $\rho_{i,t}$ | How much more or less likely the current policy makes this sampled token than the rollout policy did. |
| $\rho_{i,t}A_i$ | The ordinary, unclipped policy-improvement score. |
| $\operatorname{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)A_i$ | The same score after limiting the ratio to the trust interval. |
| $\min$ | Use the more pessimistic of the unclipped and clipped scores. |

The word “pessimistic” is important. GRPO is maximizing $J$, so taking the
minimum prevents clipping from making an update look *better* than its actual
unclipped score.

### Why the sign of the advantage changes which side is clipped

The compact expression hides two different cases. It is exactly equivalent to

$$
\ell(\rho,A)=
\begin{cases}
A\min(\rho,1+\varepsilon), & A\ge 0,\\[3pt]
A\max(\rho,1-\varepsilon), & A<0.
\end{cases}
$$

For a **positive-advantage** response, maximizing the objective tries to
increase $\rho$, hence increase the probability of its sampled tokens. Once
$\rho>1+\varepsilon$, the contribution stays at $(1+\varepsilon)A$: making the
good token still more likely receives no further first-order reward from this
batch.

For a **negative-advantage** response, maximizing the objective tries to
decrease $\rho$, hence decrease the probability of its sampled tokens. Because
$A<0$, multiplication reverses inequalities. Once $\rho<1-\varepsilon$, the
contribution stays at $(1-\varepsilon)A$: making the bad token still less likely
receives no further first-order reward from this batch.

The clipping is deliberately one-sided for each sign:

- If $A>0$, clipping stops an excessively large *increase*, but it does not hide
  an erroneous decrease. If $\rho$ is far below $1-\varepsilon$, the objective
  still has an incentive to raise it.
- If $A<0$, clipping stops an excessively large *decrease*, but it does not hide
  an erroneous increase. If $\rho$ is far above $1+\varepsilon$, the objective
  still strongly penalizes it.

This explains why the loss uses
$\min(\rho A,\operatorname{clip}(\rho)A)$ instead of simply replacing $\rho$
with its clipped value everywhere. Always using the clipped ratio would also
erase the corrective gradient when the policy moves far in the *wrong*
direction.

### A numerical clipping example

Let $\varepsilon=0.2$, so the nominal trust interval is $[0.8,1.2]$.

For $A=+1$,

$$
\ell(\rho,+1)=\min(\rho,1.2).
$$

| $\rho$ | $0.5$ | $0.8$ | $1.0$ | $1.2$ | $1.5$ |
|---:|---:|---:|---:|---:|---:|
| $\ell(\rho,+1)$ | $0.5$ | $0.8$ | $1.0$ | $1.2$ | $1.2$ |

The score rises with the token probability until $\rho=1.2$, then plateaus.
Notice that $\rho=0.5$ is **not** clipped up to $0.8$ in the objective; GRPO
keeps the incentive to repair this wrong-direction movement.

For $A=-1$,

$$
\ell(\rho,-1)=-\max(\rho,0.8).
$$

| $\rho$ | $0.5$ | $0.8$ | $1.0$ | $1.2$ | $1.5$ |
|---:|---:|---:|---:|---:|---:|
| $\ell(\rho,-1)$ | $-0.8$ | $-0.8$ | $-1.0$ | $-1.2$ | $-1.5$ |

Because the objective is maximized, moving from $-1.0$ to $-0.8$ is an
improvement. Further decreasing $\rho$ below $0.8$ gives no additional
improvement, whereas increasing a bad token to $\rho=1.5$ remains fully visible
as the worse score $-1.5$.

### Why use a policy ratio at all?

At rollout time, the current and old policies initially coincide, so $\rho=1$.
Training commonly takes multiple gradient steps on the same rollout batch;
after the first step, $\pi_\theta$ has changed but the sampled data still came
from $\pi_{\mathrm{old}}$. The ratio serves two related roles:

1. It accounts for this mismatch between the policy being optimized and the
   policy that generated the token.
2. Together with clipping, it measures and limits how far the optimizer reuses
   the batch to move each sampled-token probability.

In the active, unclipped region, the gradient of one token term is

$$
\nabla_\theta(\rho A)
=A\rho\,\nabla_\theta\log\pi_\theta(y_{i,t}\mid x,y_{i,<t}).
$$

At the start of the batch, $\rho=1$, so this reduces to
$A\nabla_\theta\log\pi_\theta$: the advantage is the signed force, while the
score-function gradient says which parameters would make the sampled token more
likely. On the clipped plateau the token term has zero gradient, so that batch
temporarily stops pushing the token farther in the already-favored direction.

**Intuition.** GRPO first labels an entire response “better than its siblings”
or “worse than its siblings,” then paints that same label onto every token in
the response. The verifier reliably tells us whether the completed reasoning
worked, but not which intermediate token caused success or failure. Its strength
is reward grounding; its weakness is coarse credit.

For example, suppose four sampled math responses receive rewards
$[1,1,0,0]$. After group normalization, the two correct responses have positive
advantages and the two incorrect responses have negative advantages. GRPO does
not ask whether a particular token was a sound algebraic step; it only asks
which completed response contained that token. Every token in either correct
response is reinforced, and every token in either incorrect response is
penalized.

The key limitation is now visible: $A_i$ depends on the whole response, but it
is copied unchanged to every token. If a 500-token derivation gets the final
answer wrong because of one arithmetic error, GRPO penalizes the useful setup,
the arithmetic mistake, and the final conclusion with the same sign.

This coarse-credit problem is the motivation for introducing a teacher.

## Step 2: OPD—Dense Teacher Feedback at Every Student Prefix

On-policy distillation uses student rollouts, then matches a teacher on those
rollout prefixes:

$$
L_{\mathrm{OPD}}
=\mathbb E_{y\sim\pi_S(\cdot\mid x)}
 \sum_t D\!\left(
   \pi_T(\cdot\mid x,y_{<t})
   \mathbin\Vert
   \pi_S(\cdot\mid x,y_{<t})
 \right).
$$

**Intuition.** OPD follows the student into the states the student actually
visits, then asks the teacher, “At this exact prefix, what would your next-token
distribution be?” It is therefore like a teacher correcting each line of the
student's own draft, rather than giving the student only a finished teacher
solution to copy.

Variants differ in divergence direction: forward KL, reverse KL, JSD, or
mixtures. OPSD uses the same model as teacher and student, but gives the teacher
privileged context such as feedback, a correct sibling rollout, a solution, or a
diagnostic.

Reverse-KL OPD can be read as a policy-gradient update with dense token reward:

$$
r_t
=\log\pi_T(y_t\mid x,y_{<t})
-\log\pi_S(y_t\mid x,y_{<t}).
$$

**Intuition.** The log-ratio is local evidence about the sampled token. A
positive value says “the teacher finds this token more plausible than the
student did”; a negative value says the opposite. This gives a dense judgment at
every position, which explains OPD's sample efficiency. It also means a
privileged, stale, or noisy teacher can inject a misleading judgment at every
position.

OPD therefore solves GRPO's density problem by giving up GRPO's exclusive
control over direction. The next question is whether we can retain both the
verifier's reliable outcome judgment and the teacher's fine-grained token
information.

## Step 3: Three Basic Ways to Combine Reward and Teacher Information

Steps 1 and 2 give us two useful but incomplete signals:

```text
GRPO: reliable response-level direction, but coarse token credit.
OPD:  dense token-level direction, but the teacher can override reward.
```

There are three natural ways to combine them, distinguished by where the teacher
appears mathematically:

```text
3A. Separate loss:     add a teacher divergence beside GRPO.
3B. Importance ratio:  put the teacher in GRPO's update geometry.
3C. Token advantage:   use the teacher to redistribute GRPO credit.
```

Where the signal appears and who controls its direction are related but distinct
questions. Step 3 focuses on placement; Step 4 will compare directional
authority explicitly.

For the derivations, fix one prefix $s=(x,y_{<t})$ and the sampled token
$a=y_t$, and write

$$
\begin{aligned}
p   &:= \pi_\theta(a\mid s) && \text{current student probability},\\
p_0 &:= \pi_{\mathrm{old}}(a\mid s) && \text{rollout / old-policy probability},\\
q   &:= \pi_T(a\mid s) && \text{teacher probability},\\
A   &:= \operatorname{stopgrad}(\widehat A) && \text{GRPO advantage of the sampled response},\\
\rho&:= \frac{p}{p_0} && \text{ordinary GRPO importance ratio}.
\end{aligned}
$$

All objectives below are written as objectives to **maximize**, and clipping is
temporarily omitted when taking gradients.

### Step 3A: Additive KL—Two Independent Gradients

`dGRPO` and `HDPO` use the most literal hybrid form. Schematically,

$$
J_{\mathrm{add}}(\theta)
=J_{\mathrm{GRPO}}(\theta)
-\beta D(\pi_\theta\Vert\pi_T).
$$

**Intuition.** Two training objectives act on the same policy simultaneously
and independently. GRPO
says “change the policy in the direction that improves verified reward,” while
the divergence says “look more like the teacher.” The coefficient $\beta$ sets
the relative weight of the teacher term. Crucially, neither term must wait
for or agree with the other.

For an unclipped, sampled GRPO term,

$$
\nabla_\theta J_{\mathrm{GRPO}}
=\mathbb E\!\left[A\rho\,\nabla_\theta\log p\right].
$$

The KL gradient is present whether or not $A$ is informative. For example, for
forward KL over the whole vocabulary, with student and teacher probabilities
$p_v$ and $q_v$ and student logit $z_v$,

$$
\begin{aligned}
D_{\mathrm{FKL}}(q\Vert p)
  &=\sum_v q_v\log\frac{q_v}{p_v},\\
\frac{\partial[-\beta D_{\mathrm{FKL}}]}{\partial z_v}
  &=\beta(q_v-p_v).
\end{aligned}
$$

Thus even when $A=0$, the teacher still moves the student toward $q$. With
reverse KL, as used by the simple dGRPO formulation,

$$
\begin{aligned}
D_{\mathrm{RKL}}(p\Vert q)
  &=\sum_v p_v\log\frac{p_v}{q_v},\\
\frac{\partial D_{\mathrm{RKL}}}{\partial z_v}
  &=p_v\left[\log\frac{p_v}{q_v}-D_{\mathrm{RKL}}(p\Vert q)\right].
\end{aligned}
$$

This gradient is also independent of $A$. The total gradient is a vector sum:

$$
\nabla J_{\mathrm{add}}
=\underbrace{\nabla J_{\mathrm{GRPO}}}_{\text{reward gradient}}
+\beta\underbrace{\nabla J_{\mathrm{teacher}}}_{\text{teacher-matching gradient}}.
$$

**Intuition.** The first vector is the verifier's vote and the second is the
teacher's vote. Adding the losses literally adds these votes in parameter space.
If they point in the same direction, learning accelerates; if they oppose each
other, they partially cancel; if the teacher term is larger, imitation wins even
when the verifier objects.

The two vectors may agree, cancel, or point in opposing directions. Therefore a
sufficiently large $\beta$ can make the net update imitate the teacher even on a
negative-advantage trajectory. This is what "objective interference" and
"coefficient-sensitive" mean mathematically; it is not merely a difference in
loss scale.

### Step 3B: TRRD—Move GRPO's Trust-Region Anchor

`RLAD/TRRD` puts the teacher inside the same advantage-weighted surrogate as the
reward. Using $\gamma$ as the weight on the old policy,

$$
\begin{aligned}
r_{\mathrm{TRRD}}
  &=\left(\frac{\pi_\theta}{\pi_{\mathrm{old}}}\right)^\gamma
    \left(\frac{\pi_\theta}{\pi_T}\right)^{1-\gamma}\\
  &=\frac{p}{p_0^\gamma q^{1-\gamma}}
   =\frac{p}{m},\\
m&:=p_0^\gamma q^{1-\gamma}.
\end{aligned}
$$

**Intuition.** TRRD does not add a second teacher objective. Instead, it changes
the ruler used by the existing reward objective: the sampled probability $p$ is
now measured relative to the teacher/old-policy reference $m$, rather than only
relative to $p_0$. The reward still says “increase” or “decrease”; the teacher
changes how much movement counts as small, acceptable, or already too large.

#### What “anchor” means

For a ratio of the generic form

$$
r(a\mid s)=\frac{\pi_\theta(a\mid s)}{b(a\mid s)},
$$

I call the denominator $b(a\mid s)$ the **anchor**. This has a precise,
operational meaning:

1. $r=1$ when the current probability equals the anchor, $\pi_\theta=b$.
2. Clipping the ratio around one translates into a probability interval around
   the anchor:

   $$
   1-\varepsilon\le r\le1+\varepsilon
   \quad\Longleftrightarrow\quad
   (1-\varepsilon)b\le\pi_\theta(a\mid s)\le(1+\varepsilon)b.
   $$

3. Equivalently, the anchor is the origin for measuring log-probability change:

   $$
   \log r=\log\pi_\theta(a\mid s)-\log b(a\mid s).
   $$

In ordinary GRPO, $b=p_0$. Thus “no relative change” means $p=p_0$, and the
positive-advantage clipping boundary is $p=(1+\varepsilon)p_0$. In TRRD,
$b=m=p_0^\gamma q^{1-\gamma}$, so ratio one and the clipping boundaries are
defined relative to $m$ instead. The teacher has literally changed the reference
point from which the size of the policy update is measured.

The interpolation is geometric, or linear in log-probability space:

$$
\log m=\gamma\log p_0+(1-\gamma)\log q.
$$

Hence, for $0<\gamma<1$, $m$ lies between $p_0$ and $q$ for the sampled token.
If $q>p_0$, the anchor moves upward; if $q<p_0$, it moves downward. As written,
$m(a\mid s)$ is a per-token anchor score and need not sum to one over the whole
vocabulary, so “anchor policy” should be understood as shorthand for the
denominator used by the token-level ratio.

The anchor is also not necessarily the policy at which the optimization epoch
starts. Usually the current student initially equals the old student, so
$p=p_0$. Ordinary GRPO then starts at $\rho=p_0/p_0=1$. TRRD instead starts at

$$
r_{\mathrm{TRRD}}\big|_{p=p_0}
=\frac{p_0}{p_0^\gamma q^{1-\gamma}}
=\left(\frac{p_0}{q}\right)^{1-\gamma},
$$

which is generally not one. If the teacher assigns more probability than the
old student ($q>p_0$), the TRRD ratio starts below one; if $q<p_0$, it starts
above one. Thus $m$ is the ratio's reference point, not necessarily its initial
point, and teacher agreement affects the clipped update from the first step.

Most importantly, an anchor is **not a destination that the optimizer always
tries to reach**. It is more like the origin and fence used by the clipped
reward update:

- A KL **target** creates a gradient toward the teacher even when $A=0$.
- A ratio **anchor** determines how the reward gradient is measured and where
  clipping saturates; when $A=0$, it creates no gradient at all.

PPO clipping is also a surrogate-objective mechanism, not a hard projection of
the complete policy into this interval. For $A>0$, the upper boundary is the
one that stops further reinforcement; for $A<0$, the lower boundary is the one
that stops further penalization.

With this terminology, $m$ is the per-token geometric anchor. $\gamma=1$
recovers the ordinary GRPO anchor $p_0$; $\gamma=0$ is fully teacher-anchored
with $m=q$. The unclipped token surrogate and its gradient are

$$
\begin{aligned}
J_{\mathrm{TRRD},t}&=A r_{\mathrm{TRRD}},\\
\nabla_\theta J_{\mathrm{TRRD},t}
  &=A r_{\mathrm{TRRD}}\nabla_\theta\log p.
\end{aligned}
$$

because $p_0$, $q$, and $A$ are detached. Unlike additive KL, there is no
teacher-only gradient. In particular,

$$
A=0\quad\Longrightarrow\quad\nabla_\theta J_{\mathrm{TRRD},t}=0.
$$

**Intuition.** Think of $A$ as the accelerator/brake decision and
$r_{\mathrm{TRRD}}$ as teacher-adjusted road geometry. If $A=0$, neither pedal
is pressed and the teacher cannot move the car. This is fundamentally different
from additive KL, where the teacher has its own pedal.

Applying PPO clipping, the relevant probability interval is centered on $m$:

$$
(1-\varepsilon)m\le p\le(1+\varepsilon)m,
$$

**Intuition.** The interval is a teacher-shaped permission range:

- If $A>0$ and the teacher likes the token ($q$ is high), the upper fence moves
  up, giving reward optimization more room to reinforce it.
- If $A<0$ but the teacher likes the token, the lower fence also moves up, so the
  token is protected from a large penalty.
- If the teacher dislikes the token, both effects reverse: positive
  reinforcement saturates sooner and negative suppression can go farther.

Thus the teacher changes the *geometry and saturation point* of a
reward-directed update; it does not independently declare the token good or bad.

One subtle point is that TRRD can also change the local gradient magnitude
through $r_{\mathrm{TRRD}}$. Calling it "geometry" does not mean that magnitude is
unchanged. It means that the magnitude change arises from a new ratio and new
clip boundary, rather than from an explicit token-credit multiplier.

### Step 3C: RLSD—Rescale Token Credit Without Changing Its Sign

`RLSD` leaves the ordinary GRPO ratio and its trust-region anchor alone. It
instead constructs a positive, detached token weight:

$$
\begin{aligned}
\Delta_t&=\log q-\log p_S,\\
w_t&=\operatorname{clip}\!\left(
  \exp\!\left(\operatorname{sign}(A)\Delta_t\right)\right)>0,\\
A_t&=Aw_t,\\[2pt]
J_{\mathrm{RLSD},t}&=\rho A_t,\\
\nabla_\theta J_{\mathrm{RLSD},t}
  &=\rho A w_t\nabla_\theta\log p.
\end{aligned}
$$

**Intuition.** RLSD keeps GRPO's road and fences fixed, but redistributes the
response-level credit across tokens. The verifier supplies one reliable thumbs
up/down for the whole trajectory; the privileged teacher acts like a highlighter
that marks which sampled tokens deserve a darker or lighter version of that same
thumbs up/down.

Here $p_S$ is the unprivileged student's probability used to compute the credit
signal; the weight is stop-gradient. Since $w_t>0$,

$$
\operatorname{sign}(A_t)=\operatorname{sign}(A).
$$

**Intuition.** For a successful response ($A>0$), $w_t\approx q/p_S$: tokens
that become more plausible when the teacher sees privileged information receive
more praise. For a failed response ($A<0$), $w_t\approx p_S/q$: tokens that the
privileged teacher finds less plausible receive more blame. Teacher/reward
agreement amplifies the update; disagreement attenuates it. Because $w_t$ is
always positive, the highlighter can change emphasis but cannot turn praise into
blame or blame into praise. It also does not move the ordinary GRPO clip
boundary; it only changes how quickly the update reaches it.

### Direct Comparison: TRRD vs. RLSD

Both are "integrated RL-distillation" in the broad sense and both preserve the
reward sign. Their precise difference is:

- **TRRD:** the teacher changes the denominator and clipping interval:
  $\rho=p/p_0$ becomes $r_{\mathrm{TRRD}}=p/m$.
- **RLSD:** the teacher changes the detached token advantage: $A$ becomes
  $A_t=Aw_t$, while $\rho=p/p_0$ is unchanged.

For a numerical example, let $p_0=0.2$, $q=0.8$, $\varepsilon=0.2$, and
$\gamma=1/2$. Ordinary GRPO has an upper positive-advantage boundary of
$1.2(0.2)=0.24$. TRRD has $m=\sqrt{0.2(0.8)}=0.4$, so its upper boundary
is $1.2(0.4)=0.48$: the teacher allows much farther movement toward this
token. At the initial $p=p_0$, however, $r_{\mathrm{TRRD}}=0.2/0.4=0.5$, so its
*local* gradient is initially smaller. By contrast, unclipped RLSD gives the
positive-advantage token weight $w=q/p_S=4$; it pushes harder locally but
retains the ordinary $0.24$ boundary. This example is why "new geometry" and
"credit magnitude" should not be treated as synonyms.

## Step 4: Decide Who Controls the Update Direction

This second axis is about *authority over direction*. A convenient diagnostic is
to set the environment advantage to zero:

> If $A=0$, can the teacher still produce a policy gradient?

### Teacher as target: the teacher can create a gradient

Naive OPD/OPSD directly optimizes a divergence:

$$
J_{\mathrm{target}}
=-D(\pi_\theta\Vert\pi_T)
\quad\text{or}\quad
-D(\pi_T\Vert\pi_\theta).
$$

**Intuition.** Here the teacher distribution is an answer key, not merely advice
about how strongly to apply a reward update. Minimizing the divergence directly
redistributes probability mass so that the student's whole next-token
distribution resembles the teacher's, whether or not the sampled trajectory
received a useful verifier reward.

For reverse-KL OPD, its score-function form is

$$
\nabla_\theta J_{\mathrm{target}}
=\mathbb E_{a\sim\pi_\theta}\!\left[
  \left(\log\pi_T(a\mid s)-\log\pi_\theta(a\mid s)\right)
  \nabla_\theta\log\pi_\theta(a\mid s)
\right].
$$

The implicit token signal

$$
\Delta(a,s)=\log\pi_T(a\mid s)-\log\pi_\theta(a\mid s)
$$

comes entirely from teacher/student disagreement, not from $A$. Thus it can
reinforce a teacher-preferred token on a verifier-failed rollout, suppress a
teacher-disfavored token on a verifier-successful rollout, and keep updating
when all group rewards are identical. In an additive hybrid such as dGRPO, this
teacher-directed vector is added to the reward-directed vector, so the final
direction depends on their relative sizes.

**Intuition.** Each student-sampled token receives a teacher-authored signed
signal. If the teacher assigns it more probability than the student does,
$\Delta>0$ and the token tends to be reinforced; if the teacher assigns less,
$\Delta<0$ and it tends to be suppressed. Notice that no environment advantage
$A$ appears in this decision.

This is powerful when the teacher is reliable and has the same information as
the student. It is risky with a privileged teacher: matching
$\pi_T(\cdot\mid s,\text{privileged info})$ asks $\pi_S(\cdot\mid s)$ to reproduce a conditional
distribution without observing the variable on which it is conditioned.

### Reward-directed use: the teacher can only reshape a gradient

RLSD explicitly satisfies

$$
\nabla_\theta J_{\mathrm{RLSD},t}
=\underbrace{(\rho w_t)}_{>0}A\nabla_\theta\log p,
\qquad
\operatorname{sign}(A_t)=\operatorname{sign}(A).
$$

TRRD has the same directional property before clipping:

$$
\nabla_\theta J_{\mathrm{TRRD},t}
=\underbrace{r_{\mathrm{TRRD}}}_{>0}A\nabla_\theta\log p.
$$

**Intuition.** Both equations have the form “positive teacher-dependent number
$\times$ verifier advantage.” The teacher can make the verifier's vote louder,
quieter, or clipped to zero, but multiplication by a positive number cannot
reverse that vote. This is the precise sense in which reward retains authority
over direction.

Clipping can suppress either gradient to zero, but it does not reverse its
reward-determined sign. Hence TRRD and RLSD are in the same family on this
second axis, despite putting the teacher in different locations on the first
axis. More specifically:

- **TRRD:** reward-directed, teacher-shaped geometry.
- **RLSD:** reward-directed, teacher-shaped token magnitude.

Returning to the numerical example $p_0=0.2$, $q=0.8$: a target-matching
loss pushes probability toward $0.8$ even if $A=0$ or $A<0$. With RLSD, a
negative advantage instead remains a penalty, but its weight is
$p_S/q=0.25$; the teacher can soften the penalty, not turn it into a reward.
With TRRD, the teacher-shifted lower clip boundary can stop that penalty early;
again it can suppress the reward update, not create an opposite one.

### RLRT: reward-directed emphasis on successful surprises

Structurally, RLRT and RLSD use the **same modification to GRPO**. Both leave the
ordinary importance ratio $\rho$ unchanged and replace the response advantage
with a positively weighted token advantage:

$$
A_t^{\mathrm{method}}=A\,w_t^{\mathrm{method}},
\qquad
J_t^{\mathrm{method}}=\rho A_t^{\mathrm{method}}.
$$

Their main algebraic difference is the definition of $w_t$. Before weight
clipping,

$$
\begin{aligned}
w_t^{\mathrm{RLSD}}
  &=\exp\!\left(
      \operatorname{sign}(A)\log\frac{p_T(y_t)}{p_S(y_t)}
    \right),\\
w_t^{\mathrm{RLRT}}
  &=\exp\!\left(
      \operatorname{sign}(A)\log\frac{p_S(y_t)}{p_T(y_t)}
    \right)
   =\frac{1}{w_t^{\mathrm{RLSD}}}.
\end{aligned}
$$

For a positive-advantage rollout, this reduces to

$$
w_t^{\mathrm{RLSD}}=\frac{p_T(y_t)}{p_S(y_t)},
\qquad
w_t^{\mathrm{RLRT}}=\frac{p_S(y_t)}{p_T(y_t)}.
$$

**Intuition.** Ordinary RLSD treats teacher agreement as evidence that a token
deserves stronger credit. RLRT asks a different question on a successful
trajectory: “Which correct choices were *surprising to the teacher*?” If the
student gave a successful token more probability than the teacher did, RLRT
amplifies it. The method treats successful disagreement as a potentially novel
strategy worth preserving rather than an error to imitate away.

The paper also differs in **routing**: RLRT applies its reversed weighting to
successful / reward-1 rollouts, where “student over teacher” can be interpreted
as successful exploration. Thus the concise comparison is:

| Property | RLSD | RLRT |
|---|---|---|
| GRPO modification | $A_t=A w_t$ | $A_t=A w_t$ |
| Positive-advantage weight | $p_T/p_S$ | $p_S/p_T$ |
| Emphasis | Teacher-endorsed successful tokens | Student-over-teacher successful tokens |
| Interpretation | Imitative credit | Exploratory credit |
| Typical routing | Positive and negative advantages | Reversed weighting on reward-1 rollouts |

Because both weights remain positive, both preserve
$\operatorname{sign}(A_t)=\operatorname{sign}(A)$. Neither changes the ordinary
GRPO ratio or clipping boundary, and neither produces a teacher-driven gradient
when $A=0$. RLRT is therefore best viewed as **RLSD's advantage-weighting
skeleton with a reversed weight and different routing**, not as a different
place to insert the teacher signal.

### Cross-classification

| Method | Mental picture | Placement of teacher signal | Who determines direction? | Teacher update when $A=0$? |
|---|---|---|---|---|
| OPD / OPSD | Teacher as answer key | Standalone divergence | Teacher | Yes |
| dGRPO / additive KL | Reward and teacher cast separate votes | Separate term beside GRPO | Reward and teacher can compete | Yes |
| TRRD | Teacher moves the guardrails | Inside importance ratio / anchor | Reward; teacher changes geometry | No |
| RLSD | Teacher highlights where credit matters | Inside detached token advantage | Reward; teacher changes magnitude | No |
| RLRT | Preserve successful surprises | Inside detached token advantage | Reward; reversed teacher weighting changes magnitude | No |

## Step 5: Route the Teacher Signal Only Where It Is Useful

The basic objectives above apply their chosen teacher mechanism broadly. The
next refinement asks a more local question: even if teacher information is
useful in principle, does every response and every token need it? Routing keeps
the base GRPO update and activates distillation only on selected regions.

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

## Step 6: Gate Noisy Teacher Advice

Routing is a hard decision: a sample or span either receives distillation or it
does not. Gating is the soft counterpart. It continuously varies the strength of
teacher advice when relevance is uncertain, which is especially useful for
multi-turn agents with noisy retrieved context.

`SDAR` keeps the loss additive:

$$
L_{\mathrm{total}}=L_{\mathrm{GRPO}}+\lambda L_{\mathrm{SDAR}}.
$$

but gates each token's auxiliary distillation by a detached sigmoid gate, usually
based on the teacher-student gap:

$$
\begin{aligned}
\Delta_t
  &=\log p_T(y_t\mid\text{privileged context})-\log p_S(y_t),\\
g_t&=\sigma(\alpha\Delta_t).
\end{aligned}
$$

**Intuition.** SDAR still has two objectives—reward learning and teacher
matching—but puts a dimmer switch on the teacher at every sampled token. When
the privileged teacher strongly endorses a token, $g_t$ is near one and the
distillation correction is admitted. When endorsement is weak or negative,
$g_t$ approaches zero and RL is left mostly alone. Because the gate is detached,
training cannot game the gate itself to reduce the loss.

The main idea is asymmetric trust. Positive gaps mean the teacher endorses a
token the student already sampled, so they are relatively trustworthy. Negative
gaps are ambiguous in multi-turn agents because retrieved skills or privileged
context may be irrelevant. The gate prevents negative teacher rejections from
overwhelming RL.

## Step 7: Learn a Reviser Before Distilling It

So far, the teacher was assumed to already provide useful token distributions.
SD-ZERO moves one step earlier in the pipeline: first learn a model that can
repair failed attempts, then use that repair behavior as the dense teacher.

`SD-ZERO` is different from the KL-routing papers because it first trains a
reviser. Phase 1, SRT, uses two supervised losses:

$$
L_{\mathrm{SRT}}=L_{\mathrm{revision}}+L_{\mathrm{generation}}.
$$

**Intuition.** The first loss teaches “given a flawed attempt and feedback, edit
it into something better.” The second teaches “produce that better behavior
directly from the original input.” Together they turn the model into both an
editor and a generator before either role is frozen as a teacher.

$L_{\mathrm{revision}}$ trains the model to revise an initial attempt given the
reward prompt. $L_{\mathrm{generation}}$ trains it to produce the improved
response from the input.

Phase 2 then freezes the reviser and performs on-policy self-distillation:

$$
L_{\mathrm{SD}}
=\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\sum_t
 \operatorname{KL}\!\left(
   \pi_\theta(\cdot\mid x,y_{<t})
   \mathbin\Vert
   \pi_{\mathrm{SRT}}(\cdot\mid x,y,\text{reward prompt},y_{<t})
 \right).
$$

**Intuition.** During phase 2, the current generator writes a draft and the
frozen reviser reads that very draft with extra outcome information. The KL then
transfers the reviser's token-by-token corrections back into the generator. The
dense teacher is therefore not just a privileged scorer; it is a learned
outcome-conditioned editor of the student's own mistakes.

## Step 8: Extrapolate Beyond the Teacher

The preceding methods ask how to imitate or safely use teacher preferences.
ExOPD asks a more aggressive question: if the teacher is an improvement over a
reference policy, can the student continue in that improvement direction beyond
the teacher itself?

`ExOPD` shows that standard OPD can be rewritten as dense KL-constrained RL:

$$
\max_\theta\;
\mathbb E\!\left[
  \log\frac{\pi_T(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
  -\operatorname{KL}(\pi_\theta\Vert\pi_{\mathrm{ref}})
\right].
$$

It then introduces a reward scaling factor:

$$
\max_\theta\;
\mathbb E\!\left[
  \alpha\log\frac{\pi_T}{\pi_{\mathrm{ref}}}
  -\operatorname{KL}(\pi_\theta\Vert\pi_{\mathrm{ref}})
\right].
$$

**Intuition.** The log-ratio $\log(\pi_T/\pi_{\mathrm{ref}})$ asks, “Which
behaviors did the teacher promote relative to the starting policy?” The KL term
is a penalty pulling the new student back toward the reference. The coefficient
$\alpha$ controls how hard we pull in the teacher-improvement direction:
$\alpha<1$ stops between reference and teacher, $\alpha=1$ aims at the teacher,
and $\alpha>1$ continues past it. Extrapolation is therefore not copying the
teacher more accurately; it is extending the displacement from reference to
teacher.

$\alpha=1$ recovers standard OPD. $0<\alpha<1$ interpolates between the
reference and teacher. $\alpha>1$ extrapolates beyond the teacher in the
direction from reference to teacher.

`Extrapolation_Cliff` is the cautionary companion. In structured-output
settings, extrapolation can be beneficial just below a threshold but collapse
when the sharpened fixed point exits the clip-safe region. Its ListOPD advantage
is base-relative:

$$
A(s,a;\alpha)
=\alpha\log\pi_T(a\mid s)
-\log\pi_S(a\mid s)
-(\alpha-1)\log\pi_B(a\mid s).
$$

The method applies this advantage with a GRPO-style importance-sampling clip.

**Intuition.** Each term answers a different question. $\log\pi_T$ says how much
the teacher likes the action, $-\log\pi_S$ says how much of that preference the
student has already absorbed, and the base correction prevents “extrapolation”
from merely rewarding actions that were already common before distillation. As
$\alpha$ grows, these relative preferences sharpen; once clipping can no longer
represent the required move safely, behavior can change abruptly—the
extrapolation cliff. The paper derives a threshold $\alpha^*(p,b,c)$ from
teacher modal probability $p$, warm-start/base modal probability $b$, and clip
strength $c$. The main message is that reward extrapolation is not a free parameter;
it has a geometry-dependent cliff.

## Step 9: Debias What the Teacher Reward Means

Clipping and gating control how much a dense signal matters, but they assume the
signal measures the right thing. CREDIT instead audits its semantics: is the
teacher rewarding reasoning that is specific to this input, or merely generic
language correlated with feedback?

`CREDIT` starts from the raw OPSD token reward:

$$
r_t(\hat y)
=\log p_{\mathrm{ref}}(\hat y\mid x,y_{<t},z)
-\log p_\theta(\hat y\mid x,y_{<t}).
$$

**Intuition.** The raw score treats any token that becomes more likely after the
teacher sees feedback $z$ as useful. But some tokens—generic phrases such as
“therefore” or “the answer is”—may correlate with that feedback across many
questions without carrying question-specific reasoning.

It interprets this as a Bayesian filtering increment: how much the token makes
feedback $z$ more predictable. But this can reward generic tokens that correlate
with feedback across many inputs.

CREDIT decomposes the teacher logprob into:

$$
\log p_{\mathrm{ref}}(\hat y\mid x,y_{<t},z)
=S_t(\hat y,x)+G_t(\hat y).
$$

Here $S_t$ is input-specific and $G_t$ is input-generic. CREDIT estimates $G_t$
by running the same response/feedback against contrastive, mismatched inputs.

**Intuition.** $S_t$ is the part of teacher endorsement that depends on the
actual input; $G_t$ is common-mode endorsement that would appear for many
unrelated inputs. Deliberately mismatching the input estimates the background
component while holding the response and feedback pattern fixed. CREDIT then
subtracts it:

$$
\begin{aligned}
R_{\mathrm{CREDIT}}(\hat y)
  ={}&\log p_{\mathrm{ref}}(\hat y\mid x,y_{<t},z)\\
    &-\alpha\,\mathbb E_{x'}
       \log p_{\mathrm{ref}}(\hat y\mid x',y_{<t},z)\\
    &-\log p_\theta(\hat y\mid x,y_{<t}).
\end{aligned}
$$

**Intuition.** This removes the input-independent component of teacher
credit. The first term is
the teacher's score on the correct input; the contrastive expectation estimates
the score the teacher produces even on wrong inputs; subtracting it keeps only
the input-specific component. CREDIT therefore changes *what the dense reward means*,
rather than merely gating or clipping it.

## Step 10: Move Teacher Computation Offline

All previous choices concern the meaning or placement of the teacher signal.
The final axis is computational: must the current student query the teacher on
fresh rollouts at every iteration, or can those rollouts and annotations be
cached?

`Lightning OPD` keeps the usual OPD advantage:

$$
A_t
=\log p_T(y_t\mid\text{prefix})
-\log p_\theta(y_t\mid\text{prefix}).
$$

but fixes the rollout distribution to the SFT reference policy and precomputes
teacher logprobs:

$$
\begin{aligned}
\text{online OPD:}\quad &y\sim\pi_\theta,\\
\text{Lightning OPD:}\quad &y\sim\pi_{\mathrm{ref}}.
\end{aligned}
$$

**Intuition.** Online OPD repeatedly asks a live teacher to annotate the
student's newest drafts. Lightning OPD instead freezes a stack of older drafts
and caches the teacher's red-pen marks on them. Training becomes much cheaper,
but the annotations no longer follow the student's changing state distribution;
teacher consistency and coverage of the cached drafts become crucial.

This makes OPD trainable without a live teacher server. Its crucial condition is
teacher consistency: the teacher used to generate SFT data should be the same
teacher used for OPD logits. Otherwise, the offline objective inherits a
teacher-mismatch bias.

`Power Distribution Bridges` is also offline, but its target is not an external
teacher. It defines a self-reward:

$$
r_{\mathrm{self}}(x,y)=\log\pi(y\mid x).
$$

The KL-regularized RL optimum under that reward is the power distribution:

$$
\pi_\gamma(y\mid x)\propto\pi(y\mid x)^\gamma.
$$

**Intuition.** Because the reward is the policy's own log-probability, already
likely responses reward themselves more. The KL constraint prevents arbitrary
collapse, and the resulting power distribution is simply a sharpened version of
the original policy: for $\gamma>1$, high-probability modes gain relative mass
and low-probability modes lose it.

Power self-distillation then samples from $\pi_\gamma$ and trains by MLE / forward
KL to those samples. This amortizes expensive power sampling into the model.

## Reference: Quick Taxonomy

After following the progression above, the papers can be located along two
orthogonal axes:

1. **What determines the update direction?**

   - Teacher distribution matching: the teacher controls the direction.
   - Verifiable reward / GRPO: the environment controls the direction.
   - Hybrid: reward controls the coarse direction, while teacher signals reshape
     magnitude, trust region, routing, or auxiliary-loss strength.

2. **Where does dense token-level credit enter?**

   - As a KL/JSD loss against teacher logits.
   - As a token-level advantage or reward.
   - As a multiplicative reweighting of GRPO advantages.
   - As a routed/gated auxiliary loss on selected samples or tokens.
   - As an offline distillation target.

## Reference: Paper-by-Paper Summary

| Paper | Main loss/objective | What is distinctive |
|---|---|---|
| `Survey_OPD_LLM.pdf` | Not a new loss; frames OPD as $\mathbb E_{y\sim p_\theta}D_f(p_T\Vert p_\theta)$ over student-sampled trajectories. | Provides the high-level axes: divergence choice, signal source, and stabilization. |
| `dGRPO_Combining_Optimization_Distillation.pdf` | $J_{\mathrm{dGRPO}}=J_{\mathrm{GRPO}}-\beta\operatorname{KL}(\pi_\theta\Vert\pi_{\mathrm{teacher}})$ on student rollouts. | The simplest additive hybrid: sparse reward optimization plus dense teacher regularization. |
| `Reinforcement_Aware_KD.pdf` | Replaces the GRPO ratio with a teacher/old-policy mixture ratio: $r_{\mathrm{TRRD}}=r_{\mathrm{GRPO}}^\gamma r_T^{1-\gamma}$. | Folds distillation into the PPO/GRPO trust-region ratio instead of adding a separate KL. |
| `RLSD_Self_Distilled_RLVR.pdf` | GRPO with token advantage $A_t=A\,\operatorname{clip}(\exp(\operatorname{sign}(A)\Delta_t))$, where $\Delta_t=\log p_T(y_t)-\log p_S(y_t)$. | Reward controls update sign; privileged teacher controls only magnitude. No auxiliary distillation loss. |
| `RLRT_Rebellious_Student_Reversed_Teacher.pdf` | GRPO with reversed teacher weight $w_t=\exp(\operatorname{sign}(A)\log[p_S(y_t)/p_T(y_t)])$ on correct rollouts. | Treats student-over-teacher disagreement on successful rollouts as exploration, not imitation. |
| `SRPO_Unifying_Group_Relative_Self_Distillation.pdf` | Routed objective: correct samples use GRPO; incorrect samples with teacher info use dynamically weighted SDPO. | Sample-level routing avoids applying self-distillation to already-correct rollouts. |
| `TRACE_Token_Routed_Self_OPD_Alignment.pdf` | GRPO plus span-routed KL: FKL on annotated key spans, optional RKL on error spans, no KL elsewhere; KL decays to zero. | Routes by token span and divergence direction, not just by sample correctness. |
| `SDAR_Self_Distilled_Agentic_RL.pdf` | $L=L_{\mathrm{GRPO}}+\lambda L_{\mathrm{SDAR}}$, with $L_{\mathrm{SDAR}}$ a gated sampled-token OPSD loss; the gate is often $\sigma(\alpha\Delta_t)$. | For multi-turn agents, keeps RL primary and gates auxiliary distillation by token-level trust. |
| `Self_Distillation_Zero.pdf` | Two phases: SRT uses two NLL losses, then phase 2 minimizes KL from a frozen reviser teacher to the current generator on generator rollouts. | Converts binary reward into revision-conditioned dense supervision without external teacher demonstrations. |
| `HDPO_Hybrid_Distillation_Policy_Optimization.pdf` | $L_{\mathrm{HDPO}}=L_{\mathrm{GRPO}}+\lambda L_{\mathrm{JSD}}$, but JSD only on cliff prompts where all standard rollouts fail and privileged rollouts pass. | Uses privileged self-distillation only when GRPO has zero gradient. Uses JSD to broaden support. |
| `REOPOLD_Scaling_Reasoning_via_Relaxed_OPD.pdf` | Reverse-KL OPD as policy gradient with reward $R_t=\log[p_T(y_t)/p_S(y_t)]$, then stop-gradient, reward clipping, and token masks. | Stabilizes OPD by clipping heavy negative teacher rewards and sampling informative high-entropy tokens. |
| `ExOPD_Learning_Beyond_Teacher_Reward_Extrapolation.pdf` | Generalized OPD: $\max\mathbb E[\alpha\log(\pi_{\mathrm{teacher}}/\pi_{\mathrm{ref}})-\operatorname{KL}(\pi_\theta\Vert\pi_{\mathrm{ref}})]$; $\alpha>1$ extrapolates reward. | Shows OPD is dense KL-constrained RL; extrapolating teacher-vs-reference reward can push beyond teacher behavior. |
| `Extrapolation_Cliff_OPD_Structured_Outputs.pdf` | ListOPD uses $A=\alpha\log\pi_T-\log\pi_S-(\alpha-1)\log\pi_B$, with IS clipping. | Studies when extrapolation becomes unstable; derives a clip-safe threshold $\alpha^*$ for structured outputs. |
| `CREDIT_Input_Specific_Credit_OPSD.pdf` | Replaces $\log p_T(y\mid x,z)-\log p_S(y\mid x)$ with contrastively debiased reward subtracting teacher logits under mismatched inputs. | Removes input-generic shortcut credit; keeps input-specific feedback credit. |
| `Lightning_OPD_Offline.pdf` | Same OPD advantage $\log p_T(y_t)-\log p_\theta(y_t)$, but rollouts and teacher log-probabilities are precomputed from the SFT reference policy. | Offline OPD; the key condition is teacher consistency between SFT data generation and OPD teacher. |
| `Power_Distribution_Bridges.pdf` | Power self-distillation samples from $\pi_\gamma(y\mid x)\propto\pi(y\mid x)^\gamma$, then trains by MLE / forward KL. | Connects power sampling, KL-regularized self-reward RL, and offline self-distillation. |

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
