# The PPO–OPD Bridge: Teacher Logits as an Implicit Critic

Companion to `advantage_tilted_opd.md` (the tilted-posterior proposal, referenced
below as "the proposal note") and `rl_opd_hybrids_survey_update_2026_07.md`.
Question addressed: PPO and OPD both assign token-level scores — PPO through a
learned value network (with its cold-start problem), OPD through teacher
log-probabilities (with its noisy/drifting scores). What is the exact
mathematical relation, and is "use OPD to initialize PPO's value network" the
right combination?

**Answer in one paragraph.** PPO's critic and OPD's teacher are the *same
mathematical object obtained two different ways*: token-level estimates of a
soft value function. The teacher's log-ratios against a fixed reference are an
implicit critic, readable by forward pass. Their famous failure modes are dual —
PPO's cold start is *statistical* error that data cures; OPD's drift is
*epistemic* error that data cannot cure **in the channel where OPD normally puts
it** (the target/reward). A classical equivalence (potential-based shaping ≡
value initialization) shows that moving the teacher into the critic's
*initialization* makes its error transient instead of permanent — so the
proposed combination is not a heuristic but the uniquely correct channel.

Steps 1–5 develop the theory; **Step 6 is the complete end-to-end algorithm**
— read that first if you want the procedure before the justification. **Step 7
contrasts the actor update with plain PPO's, gradient by gradient** — the
critic subsystem converges to PPO's, and Step 7 is the proof that the actor
subsystem never does.

---

## Step 1: The Identity — a Soft-Optimal Policy's Logits Are Q-Values

Consider the soft-RL problem: maximize terminal verifier reward $R$ under KL
regularization to a fixed reference $\pi_{\mathrm{ref}}$ at temperature
$\beta$:

$$
\max_\pi\;\mathbb E_\pi[R]\;-\;\beta\,\mathrm{KL}\!\left(\pi\,\Vert\,\pi_{\mathrm{ref}}\right).
$$

With states $s_t=(x,y_{<t})$ and deterministic token appending, the soft
Bellman equations are

$$
V^*(s)\;=\;\beta\log\sum_a \pi_{\mathrm{ref}}(a\mid s)\,e^{V^*(sa)/\beta},
\qquad
V^*(y_{\text{complete}})\;=\;R(y),
\tag{1}
$$

and the soft-optimal policy is
$\pi^*(a\mid s)=\pi_{\mathrm{ref}}(a\mid s)\,e^{(V^*(sa)-V^*(s))/\beta}$.
Rearranged:

$$
\beta\log\frac{\pi^*(a\mid s)}{\pi_{\mathrm{ref}}(a\mid s)}
\;=\;Q^*(s,a)-V^*(s)
\qquad\big(Q^*(s,a)=V^*(sa)\text{ by determinism}\big).
\tag{2}
$$

Because the right-hand side is a difference of values at consecutive prefixes,
the log-ratios **telescope along any rollout**:

$$
\beta\sum_{j<t}\log\frac{\pi^*(y_j\mid s_j)}{\pi_{\mathrm{ref}}(y_j\mid s_j)}
\;=\;V^*(s_t)-V^*(x),
\qquad
\beta\sum_{j}\log\frac{\pi^*(y_j\mid s_j)}{\pi_{\mathrm{ref}}(y_j\mid s_j)}
\;=\;R(y)-V^*(x),
\tag{3}
$$

with $V^*(x)=\beta\log\mathbb E_{\pi_{\mathrm{ref}}}\!\left[e^{R/\beta}\right]$.
These are exactly the Step-2 equations of the proposal note with the roles
relabeled (there the tilted posterior played $\pi^*$; here the teacher does),
so the telescoping is already covered by the machine-verified appendix there.
Externally this is the "your language model is secretly a Q-function" identity
(Rafailov et al.) and the parameterization behind Implicit PRM. A
self-contained derivation of (1)–(3) and of link (6) — chain rule for KL,
Gibbs variational lemma, backward induction — is in this note's Appendix.

**The reframing.** To the extent the teacher $\pi_T$ approximates the
soft-optimal policy for this task and some fixed reference:

- $\beta\log\frac{\pi_T(a\mid s)}{\pi_{\mathrm{ref}}(a\mid s)}$ **is a soft
  advantage** at every token, and
- its partial sums **are a value function** over prefixes —

both computed by a forward pass, no learning. Therefore:

| | PPO | OPD |
|---|---|---|
| Token-level advantage from | explicit critic $V_\phi$ learned from returns | implicit critic read off teacher logits |
| Available at | after critic training | immediately (forward pass) |
| Error type | statistical (few samples, untrained head) | epistemic (teacher misjudgment) |
| Cured by | more data / more TD steps | **nothing, in the reward/target channel** |

Same object, two sources. Everything else follows from this table's last row.

## Step 2: "Drift," Diagnosed — the Non-Conservative Component

Decompose the standard OPD score against any *frozen* reference:

$$
\log\frac{\pi_T(y_t\mid s_t)}{\pi_\theta(y_t\mid s_t)}
\;=\;
\underbrace{\log\frac{\pi_T(y_t\mid s_t)}{\pi_{\mathrm{ref}}(y_t\mid s_t)}}_{\text{potential difference }\;\frac1\beta[\Phi(s_{t+1})-\Phi(s_t)]}
\;+\;
\underbrace{\log\frac{\pi_{\mathrm{ref}}(y_t\mid s_t)}{\pi_\theta(y_t\mid s_t)}}_{\text{KL regularizer to ref}}.
\tag{4}
$$

If the teacher were exactly soft-optimal, the first term is a **conservative
field**: a potential difference with $\Phi=V^*$, whose sum along any
trajectory telescopes to $R-V^*(x)$ by (3) — perfect dense credit with zero
net fabrication. The failure mode the field calls "drifted scores" is precisely
the **non-conservative component**: wherever $\pi_T$ deviates from
soft-optimality (style preferences, privileged-context artifacts, the
anti-fork bias of thinking-model self-teachers), its log-ratio field contains
a part that is *not* the difference of any task-relevant potential. A
non-conservative field injects the same spurious reward on every pass through
similar states — it does not telescope away, and more data does not average it
out. Teacher bias reads as drift, not noise, because it is systematic.

Two corollaries:

- OPD's "noise" is epistemic, PPO's early "noise" is statistical. Data fixes
  the second; only real returns *placed in a channel that can override the
  teacher* fix the first.
- The privileged self-teacher case makes the analysis exact: with
  $\pi_{\mathrm{ref}}=$ the same model without the hint $c$, the score is
  $\log\frac{\pi(y_t\mid s_t,c)}{\pi(y_t\mid s_t)}=\mathrm{PMI}(y_t;c\mid s_t)$
  — the network's *amortized* Bayes posterior evidence. Its shape is a value
  increment (that is why OPSD works at all); its biases (fork suppression) are
  amortization error — fine in an initializer, dangerous in a target.

## Step 3: The Channel Theorem — Why Initialization Is the Right Place

Classical results pin down where teacher error is survivable:

- **Shaping theorem** (Ng–Harada–Russell 1999): adding a potential-based term
  $\Phi(s_{t+1})-\Phi(s_t)$ to the reward never changes the optimal policy.
- **Equivalence theorem** (Wiewiora 2003): potential-based shaping and
  initializing the value function at $\Phi$ produce identical learning
  trajectories under the same experience.

Combined with Step 2, this yields the decisive asymmetry:

1. **Teacher in the loss/reward channel** (additive hybrids, naive OPD+RL):
   the teacher term is a *regularizer*, not a potential — it moves the fixed
   point. Teacher error is **permanent**: it lives in the objective. (This is
   why every 2026 additive-family paper anneals its coefficient to zero by
   hand — manual simulation of a decay the channel does not provide.)
2. **Teacher in the critic-initialization channel**: the critic keeps
   regressing on *real returns*, so teacher error is **transient** — it decays
   at the rate TD/MC regression overwrites it — while the correct part of the
   teacher's value shape (which prefixes are promising) provides dense credit
   from step one.

Even the fork-suppression bias becomes self-correcting in channel 2: the
teacher-initialized critic starts out believing forks are bad; real returns on
hard prompts contradict it; the critic updates. The same opinion enforced
through the OPD target (channel 1) is enforced forever.

**Design principle (one line): put epistemic priors where data can override
them — the critic's initialization, not the objective.**

## Step 4: The Construction — Teacher-Potential Warm Start

Define the **teacher potential** along student rollouts (two forward passes,
no gradients):

$$
\widehat V_T(s_t)\;=\;c_x\;+\;\beta\sum_{j<t}\log
\frac{\pi_T(y_j\mid s_j)}{\pi_{\mathrm{ref}}(y_j\mid s_j)},
\tag{5}
$$

with the per-prompt constant $c_x$ an estimate of $V^*(x)$ (which is what,
by (3), it must equal). Any small set of scored rollouts of $x$ suffices to
fit it: record the success rate and pass it through link (6) below.

For binary rewards the soft value has a closed form, derived in two steps.
The free-energy identity (stated under (3); Appendix (A.5)) gives the value
of any prefix as an expectation over *reference* completions:

$$
V^*(s)\;=\;\beta\log\,\mathbb E_{\pi_{\mathrm{ref}}}\!\big[e^{R/\beta}\,\big|\,s\big].
$$

With $R\in\{0,1\}$, the random variable $e^{R/\beta}$ takes only two
values: $e^{1/\beta}$ with probability $p(s)$ — the success rate of
reference completions of prefix $s$ — and $e^0=1$ with probability
$1-p(s)$. Its expectation is therefore $1+p(s)\,(e^{1/\beta}-1)$, and
taking $\beta\log$ of it yields an invertible link between soft value and
success probability:

$$
V^*(s)\;=\;\beta\log\!\big(1+p(s)\,(e^{1/\beta}-1)\big)
\quad\Longleftrightarrow\quad
p(s)\;=\;\frac{e^{V^*(s)/\beta}-1}{e^{1/\beta}-1},
\tag{6}
$$

so the potential can be converted to calibrated success-probability targets
if the critic is parameterized in probability units. The two limits of (6)
locate the calibration: as $\beta\to\infty$, $V^*(s)\to p(s)$ — the value
is just the mean return; as $\beta\to0$, $V^*(s)\to1$ whenever $p(s)>0$ —
any success at all counts fully. The soft value sits between the average
case and the best case, which is the soft-vs-hard calibration referred to
in Level 1 below. (Appendix (A.5)–(A.6) for the full derivation.)

Three commitment levels:

- **Level 0 — no critic at all.** Use $\widehat V_T$ directly as an analytic
  baseline / per-token increment source in early training. (vOPD's closed-form
  value $-\mathrm{KL}(\pi_\theta\Vert\pi_T)$ is this idea for the
  pure-distillation reward; PBSD's "Bayesian evidence" is the privileged
  self-teacher case of (5).)
- **Level 1 — warm start (the proposal of this note).** Before RL, pretrain
  the value head by regressing $V_\phi(s_t)$ on $\widehat V_T(s_t)$ over
  student rollouts, mixing in a small batch of Monte-Carlo returns to fix the
  soft-vs-hard calibration in (6). Then run standard PPO/GAE. The critic
  begins training already encoding the teacher's credit map and spends
  training *correcting* it
  rather than *discovering* it — the cold start is gone.
- **Level 2 — decaying prior.** Keep an auxiliary increment-consistency loss

  $$
  \Big\|\big(V_\phi(s_{t+1})-V_\phi(s_t)\big)-\beta\log
  \frac{\pi_T(y_t\mid s_t)}{\pi_{\mathrm{ref}}(y_t\mid s_t)}\Big\|^2
  \tag{7}
  $$

  with weight annealed by the critic's explained variance on real returns — a
  TD update with a teacher prior instead of a zero prior. GAE's $\lambda$
  becomes a principled control parameter: schedule it down (trust dense critic credit) as
  explained variance rises.

**Reference choice.** The pair $(\pi_T,\pi_{\mathrm{ref}})$ defines the
implicit reward. For an RL-trained teacher, the principled reference is the
teacher's own pre-RL checkpoint — then $\log(\pi_T/\pi_{\mathrm{ref}})$ is the
teacher's RL delta, exactly the quantity Direct-OPD (2607.05394) distills.
Lacking that checkpoint, the student init works, at the cost of recording
teacher-vs-init *capability* gaps as "value" — a bias, but a transient one by
Step 3. What must be avoided is a *moving* reference (the current student):
that is what turns the score field non-conservative in Step 2.

**Free diagnostic.** The persistent residual between teacher-potential
increments and learned TD residuals is a per-token **map of teacher bias**:
positions where the teacher's potential drop contradicts the learned value
rise. Fork positions should show large residuals — this turns the Princeton
fork-suppression finding (2607.05184) into an online measurement.

## Step 5: Closure With Advantage-Tilted OPD

This bridge is the missing concrete realization of the proposal note's Step-6
upgrade path ("value head increments"):

- **Step 2 of the proposal note** showed the exact tilt needs $Q^*-V^*$, which
  requires solving the soft Bellman equation — intractable, hence the
  one-sample surrogate and its banked-credit bias.
- **Identity (2) here** removes that obstacle: if the teacher is approximately
  soft-optimal, its log-ratios *are* the soft values Step 2 asked for — no
  solving, just a warm start — and TD on real returns then corrects the
  approximation.
- The upgraded tilt uses whitened value increments,
  $\widehat\Delta_t\propto V_\phi(s_{t+1})-V_\phi(s_t)$, which fixes banked
  credit *and* gives fork tokens per-token credit (the targeted fix for
  fork suppression identified in the Interlude discussion).

The full stack, each component doing the one thing it is structurally good at:

| Component | Supplies | Solves |
|---|---|---|
| Tilted target (proposal note, Step 3) | fusion of teacher and verifier in bounded distribution space | the units/domination problem |
| Critic on real returns (PPO machinery) | per-token value increments for the tilt | banked credit; per-token fork credit |
| Teacher potential (OPD's implicit critic, eq. 5) | critic initialization | the cold start that led GRPO to drop the critic |

PPO cures OPD's drift (returns overwrite the prior); OPD cures PPO's cold
start (the prior is dense and instantly available); the tilt is where the two
signals combine without conflicting. The teacher enters twice, and both entries are
safe by construction: as a **prior on the simplex** (bounded, with influence quantified by the
arbitration rule (5) of the proposal note) and as a **critic init** (transient
under TD).

## The Combination Space: Where OPD Can Enter RL, Mathematically

Steps 4–5 commit to specific choices. This section states the full space of
mathematically distinct combinations, for implementations whose constraints
differ. There are exactly three slots where the teacher can enter a
policy-gradient method — the objective, the estimator, and the update rule.
The first changes the fixed point; the second provably cannot; the third
changes only the path taken toward it.

**Slot 1 — the objective.** The principled form is a single KL-regularized
functional, not a sum of two losses:

$$
J(\theta)=\mathbb E_{\pi_\theta}\big[R(y)\big]
-\beta_S\,\mathbb E_{\pi_\theta}\sum_t
\mathrm{KL}\big(\pi_\theta(\cdot\mid s_t)\,\Vert\,\pi_T(\cdot\mid s_t)\big).
\tag{C1}
$$

The additive form $L_{\mathrm{RL}}+\lambda L_{\mathrm{OPD}}$ couples two
gradients at an arbitrary exchange rate; (C1) instead has the closed-form
optimum

$$
\pi^*(y\mid x)\;\propto\;\pi_T(y\mid x)\,e^{R(y)/\beta_S}
\tag{C2}
$$

— the teacher's distribution reweighted by outcome. Note
$\beta_S$ is the *student problem's* temperature, a design choice; it is
distinct from the teacher's training temperature $\beta$ of eqs. (1)–(6).

*Derivation of (C2), two steps.* First, the chain rule for KL turns the
per-step sum in (C1) into one sequence-level divergence: expanding
$\log\frac{\pi(y\mid x)}{\pi_T(y\mid x)}$ into its per-token factors and
taking expectations,

$$
\mathbb E_{\pi}\sum_t
\mathrm{KL}\big(\pi(\cdot\mid s_t)\,\Vert\,\pi_T(\cdot\mid s_t)\big)
=\mathrm{KL}\big(\pi(\cdot\mid x)\,\Vert\,\pi_T(\cdot\mid x)\big),
$$

so (C1) is a single variational problem over completion distributions,
$J(\pi)=\mathbb E_{y\sim\pi}[R]-\beta_S\,\mathrm{KL}(\pi\Vert\pi_T)$.
Second, complete the KL (the Gibbs lemma, soft primer (4.2) / A.2 here):
write $R=\beta_S\log e^{R/\beta_S}$, fold $\pi_T\,e^{R/\beta_S}$ into the
normalized distribution $\pi^*=\pi_T\,e^{R/\beta_S}/Z$, and (C1) becomes,
for *every* policy,

$$
J(\pi)=\beta_S\log Z(x)-\beta_S\,\mathrm{KL}(\pi\Vert\pi^*),
\qquad
Z(x)=\mathbb E_{y\sim\pi_T}\big[e^{R(y)/\beta_S}\big].
\tag{C2$'$}
$$

The first term does not depend on $\pi$; the second is nonnegative and
vanishes only at $\pi=\pi^*$. Hence the maximum of (C1) is
$\beta_S\log Z(x)$, attained uniquely at (C2).

*Interpretation.* Three readings of the same formula. (i) *Bayesian:*
$\pi^*$ is a posterior — prior $\pi_T$, likelihood $e^{R/\beta_S}$. For
binary $R$, as $\beta_S\to0$ it is the teacher conditioned on success
(the rejection-sampling limit); as $\beta_S\to\infty$ it returns to the
teacher unchanged. $\beta_S$ interpolates imitate $\leftrightarrow$
maximize. (ii) *Exchange rate:*
$\log\pi^*-\log\pi_T=(R-\beta_S\log Z)/\beta_S$ — reward moves
log-probability at $1/\beta_S$ nats per reward unit, so with bounded $R$
the tilt multiplies any probability by at most $e^{\Delta R/\beta_S}$:
the teacher's support is never escaped, only reweighted. (iii)
*Projection:* (C2$'$) holds for every $\pi$, so maximizing (C1) *is*
minimizing $\mathrm{KL}(\pi\Vert\pi^*)$ — KL-regularized RL is
distribution matching to the tilted posterior, which is the fact slot 3's
tilted regression descends directly. The achieved value
$\beta_S\log Z(x)$ is the prompt's soft value with the teacher as
reference, bounded by
$\mathbb E_{\pi_T}[R]\le\beta_S\log Z(x)\le\max_y R(y)$.

**The exact reduction.** (C1) needs no new machinery: its gradient is
ordinary policy gradient on an augmented per-token reward,

$$
\tilde r_t \;=\;
\beta_S\log\frac{\pi_T(y_t\mid s_t)}{\pi_\theta(y_t\mid s_t)}
\Big|_{\text{stop-grad}},
\qquad
\tilde r_{\text{last}} \mathrel{+}= R(y).
\tag{C3}
$$

This is exact, not a heuristic: differentiating the KL term yields a
score-function part (captured by the sampled log-ratio in the reward) plus
a pass-through part $\mathbb E_{a\sim\pi_\theta}[\nabla\log\pi_\theta]=0$
that vanishes by the baseline lemma. Feed (C3) into any advantage
estimator and update (GAE then a clipped surrogate, or a group baseline)
and (C1) is optimized exactly. The sampled log-ratio is the unbiased $k_1$
KL estimator; the $k_3$ variant ($\log r + 1/r - 1$, $r=\pi_T/\pi_\theta$)
trades a small bias for lower variance and nonnegativity. This is the
mathematical core of the dense-KL-reward family (dGRPO, KDRL).

**Slot 2 — the estimator (policy-invariant).** Independently of slot 1,
add the teacher potential as shaping, or equivalently initialize the
critic with it:

$$
r'_t = \tilde r_t + \Phi(s_{t+1})-\Phi(s_t),
\qquad \Phi=\widehat V_T \text{ of (5)}.
$$

Potential-based shaping cannot move the optimum of (C1)
(Ng–Harada–Russell), and equals critic initialization for TD learners
(Wiewiora) — it changes credit assignment only. This is Step 4 restated as
a slot: the channel where teacher knowledge is safe because return
regression overwrites it wherever it is wrong.

**Slot 3 — the update rule.** With the objective and estimator fixed, the
actor update can be the clipped score-function surrogate (PPO/GRPO style:
zero gradient at zero-advantage positions, sampled tokens only) or the
tilted regression of Step 6 (full $(p-\tilde q)$ gradient, uses
unsampled-token information, bounded). This slot selects the path toward
the optimum of (C1), not the optimum itself; Step 7 compares the two paths
gradient by gradient.

**Setting $\beta_S$: the constraint form.** Rather than fixing a
coefficient, solve
$\max_\theta\,\mathbb E[R]$ subject to
$\overline{\mathrm{KL}}(\pi_\theta\Vert\pi_T)\le\varepsilon$, with
$\beta_S$ the Lagrange multiplier updated by dual ascent ($\beta_S$ rises
while the measured KL exceeds $\varepsilon$, falls otherwise). The
teacher's influence becomes a measurable budget in nats.

**Caveats.** (a) In (C2) the teacher multiplies the tilt, so the objective
taxes *every* departure from the teacher's behavior at $\beta_S$ per nat —
including departures from its biases (style, verbosity, fork suppression) —
and those errors sit in the objective, where data cannot override them
(Step 3's principle: slot 1 is a target channel). Keep $\varepsilon$
generous, or route teacher knowledge through slot 2.
(b) The reverse direction $\mathrm{KL}(\pi_\theta\Vert\pi_T)$ is the right
one: it penalizes only probability the student places where the teacher
places little, and does not force the student to cover all teacher
behavior. (c) The algorithm of Step 6 chooses slots 2+3 (critic prior +
tilted actor); an implementation that wants to keep a stock PPO/GRPO actor
should choose slots 1+2 and leave its update rule untouched.

## Step 6: The Complete Algorithm, End to End

Everything above, assembled into one runnable procedure — **PPO–OPD** viewed
as a standalone algorithm. Two facts to hold onto while reading, because they
are the usual sources of confusion:

1. **There is one code path and no phase switch.** The same loop *starts out*
   behaving exactly like GRPO-credit tilted distillation (v1 of the proposal
   note) and continuously becomes dense-credit as the critic's explained variance rises. The
   transition is carried entirely by two scalars, $w$ (which source supplies
   the critic's targets) and $\lambda$ (how much the critic's increments are
   trusted for credit) — both *measured from the run*, not hand-scheduled
   (step 8, and the subsection after the loop).
2. **The critic loop below is PPO's; the actor loop never is.** PPO's clipped
   surrogate is replaced, permanently, by cross-entropy toward the tilted
   teacher. Even with a perfect critic, positions with zero advantage receive
   the full OPD gradient $(p-q)$ — PPO's update would be zero there.

### Given

- Prompt set with verifier $R\in\{0,1\}$; a held-out split for monitoring.
- Student $\pi_\theta$ (trainable). Frozen **teacher** $\pi_T$. Frozen
  **reference** $\pi_{\mathrm{ref}}$: the teacher's pre-RL checkpoint if it
  exists, else the student init. (Never a moving reference — Step 2.)
- A **value head** on the student trunk (detached from it), outputting
  $p_\phi(s)\in[0,1]$, with $V_\phi(s)=\beta\log\!\big(1+p_\phi(s)(e^{1/\beta}-1)\big)$
  per link (6). **Zero-initialize its output layer** so $p_\phi\equiv\tfrac12$
  (constant) at start — this makes round 1 *exactly* the critic-free v1, see
  the timeline below.
- Estimated state: per-prompt rollout counts $n_x=0$ (these drive the mixing
  weight $w_x=n_x/(n_x+m)$, so $w_x=0$ at start); teacher equivalent sample
  size $m$, fit once after round 1; GAE parameter $\lambda=1$; prior mixture
  $\alpha=1$ (keep 1 in low-data regimes).
- Constants: $\beta\in[0.5,2]$; group size $G$; critic coefficient $c_v$;
  KL refresh threshold; $\lambda_{\min}$.

### Round $k$ of the training loop

1. **Rollout.** Sample $G$ responses per prompt from
   $\pi_{\mathrm{old}}:=\pi_\theta$; verifier returns $R_i$.
2. **Frozen forward passes.** Teacher logits $z_T(\cdot\mid s_t)$ at every
   prefix (these are also the prior $q$); reference log-probs
   $\log\pi_{\mathrm{ref}}(y_t\mid s_t)$ of the sampled tokens only.
3. **Group statistics.** Per-prompt success rate $\hat p_x$;
   $\hat c_x=\beta\log\!\big(1+\hat p_x(e^{1/\beta}-1)\big)$ — the group-mean
   estimate of $V^*(x)$.
4. **Critic targets** (teacher potential → mixed target):

   $$
   \widehat V_T(s_t)=\hat c_x+\beta\sum_{j<t}\log
   \frac{\pi_T(y_j\mid s_j)}{\pi_{\mathrm{ref}}(y_j\mid s_j)},
   \qquad
   \hat p_T(s_t)=\mathrm{clip}\big(\text{link}^{-1}(\widehat V_T),\,
   \varepsilon,\,1-\varepsilon\big),
   $$

   $$
   p^{\mathrm{tgt}}_t=(1-w_x)\,\hat p_T(s_t)+w_x\,\mathbf 1[R_i=1],
   \qquad
   w_x=\frac{n_x}{n_x+m}.
   \tag{8}
   $$

   The per-prompt weight $w_x$ is *estimated*, not scheduled: $n_x$ is the
   number of rollouts accumulated on prompt $x$ so far, $m$ the teacher's
   equivalent sample size (step 8; derivation just after the loop).

5. **Credit** (the tilt scalar): $\delta_t=V_\phi(s_{t+1})-V_\phi(s_t)$ with
   terminal value pinned at $R_i$;
   $\mathrm{GAE}_\lambda(t)=\sum_l\lambda^l\delta_{t+l}$; whiten across the
   batch and stop-gradient:
   $\widehat\Delta_t=\mathrm{whiten}(\mathrm{GAE}_\lambda(t))$.
6. **Actor target** (the tilt): with prior logits $z_q=\alpha z_T+(1-\alpha)
   z_{\mathrm{old}}$ (the geometric mixture is a convex combination of
   logits; $\alpha=1$ gives $z_q=z_T$), add the credit to the sampled token's
   logit and renormalize:

   $$
   \tilde q_t(v)=\frac{q(v)\,e^{\widehat\Delta_t\mathbf 1[v=y_t]/\beta}}
   {1+q(y_t)\big(e^{\widehat\Delta_t/\beta}-1\big)},
   \qquad \tilde q_t\ \text{stop-gradient.}
   \tag{9}
   $$

   ($\kappa$, the TV bound, and the arbitration rule of the proposal note
   apply verbatim with $\widehat\Delta_t$ in place of $\hat A_i$.)
7. **Updates** — inner epochs on this batch, targets held fixed:

   $$
   L(\theta,\phi)=\sum_t \mathrm{CE}\big(\tilde q_t,\;
   \pi_\theta(\cdot\mid s_t)\big)
   \;+\;c_v\sum_t \mathrm{BCE}\big(p_\phi(s_t),\,p^{\mathrm{tgt}}_t\big),
   \tag{10}
   $$

   actor gradient per position $=p-\tilde q$ (bounded); critic head on the
   detached trunk. Optionally weight the actor term by a clipped $\rho_t$ in
   late inner epochs. Break to a fresh rollout (step 1) when
   $\mathrm{KL}(\pi_\theta\Vert\pi_{\mathrm{old}})$ exceeds the threshold.
8. **Parameter estimates** (once per round; measured, not scheduled). *After
   round 1 only*: fit the teacher equivalent sample size $m$ by empirical
   Bayes — maximize over $m$ the Beta-Binomial marginal likelihood of the
   observed group outcomes $\{(k_x,G)\}$ under the prior
   $\mathrm{Beta}\big(m\hat p_T(x),\,m(1-\hat p_T(x))\big)$; a 1-D search.
   Every round: add the rollouts to the counts and read both parameters off the data,

   $$
   n_x\mathrel{+}=G,
   \qquad
   w_x=\frac{n_x}{n_x+m},
   \qquad
   \lambda=1-\mathrm{EV}_+\,(1-\lambda_{\min}),
   $$

   where $\mathrm{EV}_+=\mathrm{clip}(\mathrm{EV},0,1)$ is the explained
   variance of $V_\phi$ against realized returns on this round's data.
   $\alpha$ stays at 1 unless deliberately optimizing beyond the teacher
   (proposal note, Interlude).
9. **Stop** when the actor loss plateaus ($p\approx\mathbb E[\tilde q]$: the
   posterior is reached) and/or held-out pass@k peaks.

### Why $w$ and $\lambda$ are estimates, not schedules

An earlier version of this note increased a single global $w$ monotonically
with explained variance and tied $\lambda$ to it. Both are better *read off
the run*, and the channel theorem (Step 3) is what licenses doing so on
purely statistical grounds: the mixture (8) shapes only the critic's
*targets* — the initialization channel — so no choice of $w$ can bias the
final policy. $w$ controls variance and transient speed, nothing else, and
should therefore be the precision-optimal fusion weight. That weight has a
closed form.

**The conjugate form.** In the binary link the critic target is a
probability, so the natural model is Beta–Bernoulli. Treat the teacher
potential's prediction $\hat p_T(x)$ as a prior worth $m$ pseudo-rollouts,
$\mathrm{Beta}\big(m\hat p_T,\,m(1-\hat p_T)\big)$, and each real rollout as
a Bernoulli observation. The posterior mean is exactly (8):

$$
\frac{m\,\hat p_T(x)+n_x\bar R_x}{m+n_x}
=(1-w_x)\,\hat p_T(x)+w_x\,\bar R_x,
\qquad
w_x=\frac{n_x}{n_x+m};
$$

and because BCE is linear in its target, regressing on the per-trajectory
$\mathbf 1[R_i=1]$ with weight $w_x$ — which is what (8) does at every state
of the trajectory — equals posterior-mean regression in expectation. So $w$
was never a free parameter. It is the consequence of one interpretable scalar,
$m$ = *how many rollouts the teacher's prediction is worth*, and empirical
Bayes measures $m$ directly from round-1 calibration: a well-calibrated
teacher earns a large $m$ (slow handoff), a misspecified or overconfident
one a small $m$ (fast handoff). Per-prompt structure comes free: a prompt
sampled heavily for five rounds has $w_x\to1$ while a rarely-visited one still
leans on the teacher.

**The general form, and two refinements it provides.** Fusing a
biased-but-deterministic estimate (the teacher potential, bias $b$) with an
unbiased-but-noisy one (the MC return, variance $v=\sigma_R^2/n$) has
MSE-optimal weight

$$
w^*=\frac{b^2}{b^2+v},
\qquad
\hat b^2=\max\!\Big(0,\ \overline{(\hat p_T-\bar R)^2}-\hat v\Big)
$$

(the subtraction debiases the squared residual). Modeling the teacher's bias
as prior variance $\propto1/m$ recovers $w^*=n/(n+m)$ — the same formula,
derived twice. The extra freedom provides what the pseudo-count form cannot:
(i) **non-monotonicity** — when the student surpasses the teacher,
$\hat b^2$ grows and $w\to1$ on its own (this *is* Step 2's drift showing up
in the residuals); when returns are noisy on hard prompts, $w$ backs off.
(ii) **position dependence** — the teacher potential integrates Step 2's
non-conservative component along the prefix, so its systematic error grows
roughly linearly in position, $b(t)\approx\delta t$, with $\delta$ estimable
by regressing residuals on $t$. That gives
$w_t=(\delta t)^2/\big((\delta t)^2+v\big)$: early tokens trust the teacher
longest, deep tokens hand off first, one scalar controlling the whole
profile.

**Decoupling $\lambda$ from $w$.** The old tie
$\lambda=1-w(1-\lambda_{\min})$ conflated two different trusts. $w$ is
*teacher-vs-verifier* trust in the head's targets, governed by counts and
calibration as above. $\lambda$ is *head-vs-MC* trust in the credit
estimator, governed by how well the head actually fits — which explained
variance measures directly. Each parameter is now determined by its own
statistic, and neither is forced to be monotone.

### The same loop as pseudocode

```python
# frozen: teacher, ref.  trainable: student, value head p_phi (zero-init output).
n_x, m, lam = zeros(P), None, 1.0                             # P = #prompts
for round in range(K):
    batch = sample_rollouts(student, prompts, G)              # y, R
    zT    = teacher.logits(batch)                             # [N,T,V] = prior
    lpT   = gather_logprob(zT, batch.y)                       # [N,T]
    lpref = ref.logprob_sampled(batch)                        # [N,T]

    c_x   = beta * log(1 + group_success_rate(batch) * (exp(1/beta) - 1))
    V_T   = c_x + beta * cumsum(lpT - lpref, exclusive=True)  # teacher potential
    p_T   = clip(inv_link(V_T), eps, 1 - eps)
    w     = 0.0 if m is None else (n_x / (n_x + m))[batch.pid][:, None]
    p_tgt = (1 - w) * p_T + w * batch.R[:, None]              # eq. (8) = Beta posterior mean

    for _ in range(inner_epochs):
        V      = link(p_phi(hidden.detach()))                 # V(s_T) := R pinned
        credit = whiten(gae(diff(V, terminal=batch.R), lam)).detach()
        # round 1: V const (zero-init) and lam=1  =>  credit == GRPO's A_hat

        z_tilt = zT.clone()                                   # eq. (9)
        z_tilt.scatter_add_(-1, batch.y, credit / beta)       # sampled coord only
        q_tilde = softmax(z_tilt, -1).detach()

        loss = CE(q_tilde, student.logits(batch)) \
             + c_v * BCE(p_phi_out, p_tgt)                    # eq. (10)
        step(loss)
        if kl(student, pi_old) > kl_max: break                # refresh rollouts

    if m is None:                                             # once, after round 1:
        m = fit_beta_binomial_ess(p_T[:, 0], batch.R)         # teacher ESS by empirical Bayes
    n_x += G                                                  # teacher weight in critic targets decays as counts grow
    ev  = explained_variance(V, batch.R)
    lam = 1 - clip(ev, 0.0, 1.0) * (1 - lam_min)              # bootstrap weight rises with explained variance
```

### Timeline: what actually happens over training

- **Round 1.** The value head is zero-initialized, so $V_\phi$ is constant and
  $\lambda=1$: the whitened GAE credit *equals* GRPO's $\hat A_i$, and the
  round is exactly v1 — tilted distillation with sequence-level credit. No
  critic information is used because none exists yet.
- **Early rounds (few rollouts banked, $w_x$ small).** The critic regresses
  mostly toward the teacher potential (eq. 8 with $w_x\approx0$) — it is
  being *warm-started as a byproduct of training*, not in a separate phase.
  Credit is still MC-dominated ($\lambda$ near 1), now with a per-position
  baseline as $V_\phi$ departs from constant.
- **Mid rounds ($n_x$ outgrowing $m$, EV rising).** $w_x\to1$ prompt by
  prompt: real returns replace the teacher in the critic's targets — the
  prior exits the critic exactly as the channel theorem (Step 3) prescribes,
  transiently, and fastest on the prompts visited most.
  $\lambda\to\lambda_{\min}$ as the head's explained variance rises: credit
  shifts from banked whole-outcome to per-token value increments.
- **Late rounds.** PPO-grade credit (trained critic, low $\lambda$) drives
  the actor — **whose loss has never changed**: CE toward the tilted teacher,
  full OPD gradient wherever credit is zero, mass redistributed in teacher
  proportions wherever it is negative. This is the sense in which the mature
  algorithm is "PPO's credit machinery attached to an MPO/AWR-style actor
  with a teacher prior," not PPO. (Step 7 makes this precise, gradient by
  gradient.)
- **What remains of GRPO.** Only the group statistics (whitening, $\hat c_x$)
  remain from GRPO, and even they are transitional: a mature critic's
  $V_\phi(x)$ can replace $\hat c_x$ and batch whitening can replace group
  whitening, at which point groups are a variance-reduction convenience, not
  a requirement.

## Step 7: The Same Credit, Used Two Ways — PPO's Actor Update vs. Ours

Step 6 showed the *critic* subsystem converging to PPO's. This step pins down
the other half of the claim. Both algorithms hand the actor the **identical
scalar** $\widehat\Delta_t$ — same critic, same GAE, same whitening; the divergence
is entirely in how that scalar becomes a gradient on the position-$t$ token
distribution. Everything below is per position, in logit space, so the two
updates can be laid side by side coordinate by coordinate.

### What plain PPO does

PPO maximizes the clipped importance-ratio surrogate over several inner epochs
on the same batch:

$$
L^{\mathrm{CLIP}}(\theta)=\mathbb E_t\Big[\min\big(r_t(\theta)\,\widehat\Delta_t,\
\mathrm{clip}(r_t(\theta),\,1-\epsilon,\,1+\epsilon)\,\widehat\Delta_t\big)\Big],
\qquad
r_t(\theta)=\frac{\pi_\theta(y_t\mid s_t)}{\pi_{\mathrm{old}}(y_t\mid s_t)},
\tag{11}
$$

usually plus a small entropy bonus. Two mechanics matter here. First, this is a
**score-function update**: on the first inner epoch ($r_t=1$, clip inactive)
the per-position gradient is $\widehat\Delta_t\nabla_\theta\log\pi_\theta(y_t\mid
s_t)$ — it directly updates *one coordinate* (the sampled token) and lets softmax mass
conservation handle the rest. Second, the **clip is the trust region**: the
ratio is measured against the moving snapshot $\pi_{\mathrm{old}}$, and once
$r_t$ exits the band on the profitable side the gradient is hard-zeroed; this
is what licenses multi-epoch reuse of stale rollouts.

### The per-token gradient, side by side

Write both as *loss* gradients with respect to the logits $z$ at one position
($p=\pi_\theta(\cdot\mid s_t)$, $q$ = prior from eq. 9, PPO taken at $r_t=1$,
unclipped):

$$
\nabla_z L^{\mathrm{PPO}}_t=-\,\widehat\Delta_t\,(\mathbf 1_{y_t}-p),
\qquad\qquad
\nabla_z L^{\mathrm{ours}}_t=p-\tilde q_t
=\underbrace{(p-q)}_{\text{distillation pull}}
-\underbrace{\kappa_t\,(\mathbf 1_{y_t}-q)}_{\text{squashed reinforcement}},
\tag{12}
$$

$$
\kappa_t=\frac{q_{y_t}\big(e^{\widehat\Delta_t/\beta}-1\big)}
{1+q_{y_t}\big(e^{\widehat\Delta_t/\beta}-1\big)}
\;\in\;\Big(-\tfrac{q_{y_t}}{1-q_{y_t}},\;1\Big).
$$

Reading eq. (12): **three substitutions transform PPO's update into ours** —
(i) an always-on distillation term $(p-q)$ appears; (ii) the linear, unbounded
credit coefficient $\widehat\Delta_t$ is replaced by the saturating $\kappa_t$;
(iii) the mass-conservation counterweight changes from the student's own $p$ to
the teacher's $q$. The reinforcement terms are otherwise structurally parallel.

Coordinate by coordinate (each row is the loss gradient on that logit; the
target's coordinates are $\tilde q_t(y_t)=q_{y_t}+\kappa_t(1-q_{y_t})$ and
$\tilde q_t(v)=(1-\kappa_t)\,q_v$ for $v\ne y_t$):

| logit coordinate | PPO | ours |
|---|---|---|
| sampled token $y_t$ | $-\widehat\Delta_t\,(1-p_{y_t})$ | $(p_{y_t}-q_{y_t})-\kappa_t\,(1-q_{y_t})$ |
| each unsampled $v\ne y_t$ | $+\widehat\Delta_t\,p_v$ | $(p_v-q_v)+\kappa_t\,q_v$ |

### What the algebra implies, token by token

1. **At zero credit** ($\widehat\Delta_t=0\Rightarrow\kappa_t=0$): PPO's
   gradient is exactly zero — the actor receives *no update* at any position the
   critic scores as neutral. Ours is $(p-q)$, the full OPD gradient: the
   teacher keeps teaching everywhere, and credit is added on top only where it
   exists. This single row is the reason the combination can run on hundreds
   of prompts when pure PG cannot.
2. **Reinforcement strength is squashed, asymmetrically.** PPO's coefficient
   is linear and unbounded in $\widehat\Delta_t$ (until the clip zeroes it).
   Ours saturates: at $\kappa_t\to1$ the target degenerates to
   $\mathbf 1_{y_t}$ — *the strongest possible positive update is one SFT step
   on the sampled token* (the unit-weight RAFT limit of the proposal note's
   Interlude); at the lower bound the target becomes
   $\tilde q_t(y_t)\to0,\ \tilde q_t(v)\to q_v/(1-q_{y_t})$ — the teacher with
   the sampled token deleted and renormalized. Both extremes are proper
   distributions, so the update is bounded no matter how large the credit.
3. **The prior gates the credit.** For a token the teacher disfavors
   ($q_{y_t}$ small), $\kappa_t\approx q_{y_t}(e^{\widehat\Delta_t/\beta}-1)$
   — the reinforcement is *attenuated by the teacher's mass*, and the net
   sampled-coordinate gradient flips sign only when the credit clears the
   student–teacher logit gap (the arbitration rule, proposal note eq. 5). PPO
   is prior-blind: any $\widehat\Delta_t>0$ reinforces, at full linear
   strength, regardless of how implausible the teacher finds the token.
4. **Where the displaced mass goes.** PPO's unsampled-coordinate gradient is
   $\widehat\Delta_t\,p_v$: mass is removed from (or added to) alternatives *in
   proportion to the student's own current probabilities* — the update says
   "not this" (or "more of this") with no information about which alternatives
   should receive the difference. Ours moves every unsampled coordinate toward
   $(1-\kappa_t)q_v$: the correction says "*these* instead," in teacher
   proportions. This is information PPO structurally cannot use, because a
   score-function update only ever addresses one coordinate directly.
5. **The trust regions are in different places.** PPO: a hard clip on the ratio
   against the *moving* $\pi_{\mathrm{old}}$, non-smooth, gradient dead
   outside the band. Ours: the target itself is a bounded distribution near
   the *fixed* teacher, and CE toward a fixed target is self-limiting
   ($p\to\tilde q\Rightarrow$ gradient $\to0$). The only $\pi_{\mathrm{old}}$
   machinery we keep is the $\mathrm{KL}(\pi_\theta\Vert\pi_{\mathrm{old}})$
   rollout-refresh trigger (and the optional clipped $\rho_t$ weight in late
   inner epochs), which guards state-distribution staleness, not per-token
   step size.
6. **Multi-epoch reuse.** PPO needs the ratio-clip to make reuse of a stale
   batch valid at all. Our inner loop is supervised regression toward a fixed
   target — reuse is free by construction; staleness enters only through the
   state distribution (the DAgger-style argument), hence the refresh trigger.
7. **Bias, by choice.** PPO's first-epoch gradient is unbiased for
   $\nabla\mathbb E[R]$ (given the baseline); its fixed point is the reward
   maximizer, and entropy collapse must be countered with additional terms. Ours is
   *deliberately biased* for $\nabla\mathbb E[R]$ — $\kappa_t$ saturates — because
   the declared objective is $\mathbb E[R]-\beta\,\mathrm{KL}(\pi\Vert\pi_T)$
   and the update is a projection onto its closed-form optimum
   $\propto\pi_T e^{R/\beta}$, not ascent on $\mathbb E[R]$. The fixed point
   is a full distribution with teacher-inherited entropy.

**Summary of the fork.** After warm-up, both algorithms compute the same
$\widehat\Delta_t$; PPO then multiplies it into an importance-ratio surrogate
that updates one coordinate with unbounded linear strength and is inactive
without credit, while ours exponentiates it into a one-coordinate tilt of the
teacher's logits — bounded, prior-gated, teacher-shaped on the remaining
coordinates, and reverting to pure OPD when credit vanishes. That is the exact
sense in which the mature algorithm is *PPO's credit machinery driving an
MPO/AWR-style projection actor with a teacher prior* — and never PPO.

## Caveats and Validity Conditions

1. **Soft-optimality of the teacher is the assumption everything else depends
   on.** Identity
   (2) holds exactly only if $\pi_T$ is the soft-optimal policy for *this*
   reward w.r.t. *some* fixed reference. Sang et al.'s condition C1
   (2605.12483) is the empirical manifestation of its failure: a non-reward-shaped
   teacher's potential still initializes *shape*, but with more to unlearn.
2. **Soft vs hard calibration.** The potential estimates the optimistic
   log-sum-exp value (1), a monotone transform of the hard value; the link (6)
   or return-mixing in Level 1 is not optional.
3. **Self-teacher amortization bias.** For OPSD, (5) inherits the network's
   amortized-Bayes errors (PMI biases, fork suppression). Acceptable in an
   initializer, self-correcting under Level 1/2; never move it back into the
   target.
4. **Frozen reference only.** A reference that moves with $\theta$ re-creates
   the non-conservative drift field of Step 2 inside the initializer's own
   definition.

## References (external; spot-check IDs before citing)

- Rafailov et al., *From r to Q\*: Your Language Model is Secretly a
  Q-Function* (arXiv 2404.12358) — identity (2) for DPO policies.
- Ng, Harada, Russell, *Policy invariance under reward transformations* (ICML
  1999) — potential-based shaping theorem.
- Wiewiora, *Potential-based shaping and Q-value initialization are
  equivalent* (JAIR 2003) — the channel equivalence of Step 3.
- Yuan et al., *Free Process Rewards without Process Labels* (arXiv
  2412.01981) — Implicit PRM: log-ratio parameterization yields per-token
  Q-estimates (the converse direction of this note).
- vOPD (arXiv 2605.07865) — analytic value for the distillation reward.
- PBSD (arXiv 2606.09348) — privileged-self-teacher case of eq. (5).
- Direct-OPD (arXiv 2607.05394) — teacher RL delta as dense reward.
- Sang et al. (arXiv 2605.12483) — condition C1.
- VinePPO (arXiv 2410.01679) — documented difficulty of learned critics in
  LLM RL; the motivation for warm starts.
- Kaur et al. (arXiv 2607.05184) — fork suppression; measurable via the
  Step-4 diagnostic.

## Appendix: Soft Bellman Equations and the Soft-Optimal Policy, From Scratch

Step 1 asserts the soft Bellman equation (1), the policy identity (2), and
the telescoping (3); Step 4 asserts the binary link (6). This appendix
derives all four from first principles, in full detail. The route:

- **A.1** turns the sequence-level objective into a token-level one (chain
  rule for KL), defines the regularized value functions, and derives the
  *evaluation* equation (A.1) — true for every policy, no optimality yet.
  It ends with a reconciliation against the textbook (entropy-regularized,
  SAC-style) soft Bellman equations, which look superficially different.
- **A.2** solves the only optimization problem the whole theory contains —
  a single state's token distribution — in closed form, via a
  one-substitution lemma ("completing the KL").
- **A.3** assembles A.1 and A.2 into the soft Bellman *optimality* equation
  and the soft-optimal policy, by backward induction over prefixes.
- **A.4** derives the four downstream facts the body uses: telescoping, the
  sequence-level posterior, the path-integral form of the value, and the
  binary link.

No Lagrange multipliers and no function-space calculus appear anywhere; and
because generation halts at EOS or a length cap, backward induction
replaces any contraction-mapping argument.

**Setting.** Generation is a finite-horizon MDP:

- A **state** is a prefix $s=(x,y_{<t})$ — the prompt plus everything
  generated so far. The policy conditions on exactly this, so it is a
  legitimate Markov state.
- An **action** is the next token $a\in\mathcal V$; taking it moves the
  state to the longer prefix $sa$ with certainty — deterministic dynamics.
- An **episode** ends when EOS is emitted or a length cap is hit, so every
  trajectory has bounded length and only finitely many prefixes are
  reachable.
- The **reward** is terminal only: the verifier score $R(y)$ on the
  complete sequence, kept general (real-valued) until A.4(iv), where it
  becomes binary. Intermediate rewards are zero; A.1 notes how to
  reinstate them.

The problem is

$$
\max_\pi\ J(\pi),\qquad
J(\pi)=\mathbb E_{y\sim\pi(\cdot\mid x)}[R(y)]
-\beta\,\mathrm{KL}\big(\pi(\cdot\mid x)\,\Vert\,\pi_{\mathrm{ref}}(\cdot\mid x)\big),
$$

with the KL taken at the **sequence** level: it compares full distributions
over completions $y$.

### A.1 From the sequence objective to the per-state evaluation equation

**Step 1: the chain rule makes the objective token-additive.** Both $\pi$
and $\pi_{\mathrm{ref}}$ assign a sequence its probability as a product of
per-token conditionals, $\pi(y\mid x)=\prod_t\pi(y_t\mid s_t)$, so the
sequence-level log-ratio splits into a sum:

$$
\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
=\sum_t\log\frac{\pi(y_t\mid s_t)}{\pi_{\mathrm{ref}}(y_t\mid s_t)}.
$$

Taking $\mathbb E_{y\sim\pi}$ of both sides — the left side becomes the
sequence KL by definition — and substituting into $J$:

$$
J(\pi)=\mathbb E_{y\sim\pi}\Big[R(y)-\beta\sum_t
\log\frac{\pi(y_t\mid s_t)}{\pi_{\mathrm{ref}}(y_t\mid s_t)}\Big].
$$

The sequence-level problem is thereby an ordinary token-level RL problem
whose per-step "reward" is the KL penalty
$-\beta\log(\pi/\pi_{\mathrm{ref}})$ of the token just emitted, plus the
terminal $R$. Additivity over steps is the property dynamic programming
needs.

**Step 2: values.** For any policy $\pi$, define the regularized value —
"expected terminal reward minus all *remaining* KL cost, continuing with
$\pi$ from here":

$$
V^\pi(s)=\mathbb E_\pi\Big[R(y)-\beta\!\!\sum_{j\ge|s|}\!\log
\frac{\pi(y_j\mid s_j)}{\pi_{\mathrm{ref}}(y_j\mid s_j)}\ \Big|\ s\Big],
\qquad V^\pi(y_{\text{complete}})=R(y),
$$

and the Q-function — the same quantity when the next token is *committed*
to be $a$, under the convention that the KL cost of that commitment is
**not** charged to $Q$ (it remains assigned to the state where the choice was
made).
In general $Q^\pi(s,a)=r(s,a)+\mathbb E_{s'}[V^\pi(s')]$; here the
intermediate reward is zero and the transition is deterministic, so the
backup collapses to a composition:

$$
Q^\pi(s,a)=V^\pi(sa).
$$

**Step 3: split off the first token.** Split the defining sum of $V^\pi(s)$ at its
first position, $j=|s|$. Conditioned on the first action being $a$,
everything after position $|s|$ depends only on the new prefix $sa$ (Markov
structure), and its conditional expectation is exactly $V^\pi(sa)$:

$$
V^\pi(s)
=\mathbb E_{a\sim\pi(\cdot\mid s)}\Big[
\underbrace{-\beta\log\frac{\pi(a\mid s)}{\pi_{\mathrm{ref}}(a\mid s)}}_{\text{this token's KL cost}}
\;+\;\underbrace{V^\pi(sa)}_{=\,Q^\pi(s,a)}\Big].
$$

The first term's expectation is $-\beta$ times the KL of the current token
distribution, giving the **soft Bellman evaluation equation**:

$$
V^\pi(s)=\mathbb E_{a\sim\pi(\cdot\mid s)}\big[Q^\pi(s,a)\big]
-\beta\,\mathrm{KL}\big(\pi(\cdot\mid s)\,\Vert\,\pi_{\mathrm{ref}}(\cdot\mid s)\big).
\tag{A.1}
$$

It holds for **every** policy — there is no optimality in it yet.

**Reconciliation with the textbook soft Bellman equations.** (A.1) can look
unfamiliar next to the max-ent/SAC equations
($Q=r+\gamma\mathbb E[V]$, $V=\mathbb E_\pi[Q-\alpha\log\pi]$,
$V^*=\alpha\log\sum_a e^{Q^*/\alpha}$). Three substitutions connect them:

1. *Move the KL inside the expectation.* (A.1) is identically
   $V^\pi(s)=\mathbb E_{a\sim\pi}\big[Q^\pi(s,a)
   -\beta\log\tfrac{\pi(a\mid s)}{\pi_{\mathrm{ref}}(a\mid s)}\big]$ —
   SAC's soft value equation with the entropy term $-\alpha\log\pi$
   generalized to a log-ratio against the reference.
2. *Entropy is KL to uniform.*
   $-\mathrm{KL}(\pi\Vert U)=H(\pi)-\log|\mathcal V|$, so setting
   $\pi_{\mathrm{ref}}=$ uniform recovers the entropy-regularized equations
   exactly, up to a constant $\beta\log|\mathcal V|$ per step that shifts
   $V$ and $Q$ equally and cancels from every policy and every advantage.
   The reference-weighted log-sum-exp of (A.3) below likewise reduces to
   the familiar bare log-sum-exp.
3. *The missing backup line.* The textbook first equation
   $Q=r+\gamma\,\mathbb E[V(s')]$ is present but invisible: with
   $r\equiv0$, $\gamma=1$, and deterministic appending, it degenerates to
   $Q^\pi(s,a)=V^\pi(sa)$ — a definition rather than an update. Reinstate
   intermediate rewards or stochastic transitions and every derivation
   below goes through with $r(s,a)+\mathbb E_{s'}[V^\pi(s')]$ in place of
   $V^\pi(sa)$.

The familiar *log-sum-exp* soft Bellman equation is what (A.1) becomes at
the optimal policy; getting there requires solving a one-step maximization,
which is the next lemma.

### A.2 The Gibbs variational lemma: the one-step problem in closed form

Backward induction (A.3) will repeatedly hand us this problem: at a single
state, with the values of all successor prefixes already known, choose the
token distribution. Abstractly: given any function $f$ on the vocabulary
(read $f(a)$ as "the value of landing on $sa$"), solve

$$
\max_{\pi(\cdot\mid s)}\ \mathbb E_{a\sim\pi}[f(a)]
-\beta\,\mathrm{KL}\big(\pi\,\Vert\,\pi_{\mathrm{ref}}\big).
$$

The lemma solves it exactly, for arbitrary $f$, by an algebraic identity —
the KL analogue of completing the square.

**Lemma.** Let $Z=\sum_a\pi_{\mathrm{ref}}(a\mid s)\,e^{f(a)/\beta}$ (a
normalizer) and $\pi_f(a)=\pi_{\mathrm{ref}}(a\mid s)\,e^{f(a)/\beta}/Z$
(the *Gibbs tilt* of the reference by $f$ at temperature $\beta$). Then for
**every** distribution $\pi$ over the vocabulary,

$$
\mathbb E_{a\sim\pi}[f(a)]-\beta\,\mathrm{KL}(\pi\Vert\pi_{\mathrm{ref}})
=\beta\log Z-\beta\,\mathrm{KL}(\pi\Vert\pi_f).
\tag{A.2}
$$

*Proof.* Take the log of $\pi_f$'s definition and solve for $f$: for every
token $a$,

$$
f(a)=\beta\log\frac{\pi_f(a)}{\pi_{\mathrm{ref}}(a\mid s)}+\beta\log Z.
$$

Substitute this into $\mathbb E_\pi[f]$, and subtract
$\beta\,\mathrm{KL}(\pi\Vert\pi_{\mathrm{ref}})
=\beta\,\mathbb E_\pi\log\frac{\pi}{\pi_{\mathrm{ref}}}$:

$$
\mathbb E_\pi[f]-\beta\,\mathbb E_\pi\log\frac{\pi}{\pi_{\mathrm{ref}}}
=\beta\log Z
+\beta\,\mathbb E_\pi\Big[\log\frac{\pi_f}{\pi_{\mathrm{ref}}}
-\log\frac{\pi}{\pi_{\mathrm{ref}}}\Big]
=\beta\log Z-\beta\,\mathbb E_\pi\log\frac{\pi}{\pi_f};
$$

the reference cancels inside the bracket, and the remaining expectation is
$\mathrm{KL}(\pi\Vert\pi_f)$. $\square$

**Reading off the solution.** The identity says: *any* candidate $\pi$
scores $\beta\log Z$ minus a non-negative penalty measuring exactly how far
$\pi$ is from the tilt $\pi_f$. Since $\mathrm{KL}\ge0$ with equality iff
its arguments coincide,

$$
\max_\pi\big\{\mathbb E_\pi[f]-\beta\,\mathrm{KL}(\pi\Vert\pi_{\mathrm{ref}})\big\}
=\beta\log Z,
\qquad\text{attained uniquely at }\pi=\pi_f.
$$

Both the optimal value (a log-partition function) and the optimizer (a
Gibbs distribution) follow from one substitution — no derivatives, no
Lagrange multipliers, and uniqueness is free. This finite Donsker–Varadhan
duality is the entire optimization content of soft RL; everything else
below is bookkeeping.

### A.3 Backward induction: the optimality equations

Define $V^*(s)=\max_\pi V^\pi(s)$ — finitely many reachable prefixes and a
bounded horizon, so the maximum is attained. Order prefixes by tokens
remaining and induct backward, longest first.

**Base case.** At a complete sequence there is nothing left to choose:
$V^*(y_{\text{complete}})=R(y)$.

**Inductive step.** Fix $s$, and assume $V^*$ is known — and attained by
some continuation policy — for every strict extension of $s$, in
particular every $sa$. Two inequalities pin $V^*(s)$ down.

*Upper bound: no policy beats the log-sum-exp.* For an arbitrary $\pi$,
start from (A.1), replace $Q^\pi(s,a)=V^\pi(sa)$ by its ceiling $V^*(sa)$,
then apply the Lemma with $f(a)=V^*(sa)$:

$$
V^\pi(s)
\;\overset{\text{(A.1)}}{=}\;
\mathbb E_{a\sim\pi}\big[V^\pi(sa)\big]
-\beta\,\mathrm{KL}(\pi\Vert\pi_{\mathrm{ref}})
\;\le\;
\mathbb E_{a\sim\pi}\big[V^*(sa)\big]
-\beta\,\mathrm{KL}(\pi\Vert\pi_{\mathrm{ref}})
\;\overset{\text{(A.2)}}{\le}\;
\beta\log\sum_a\pi_{\mathrm{ref}}(a\mid s)\,e^{V^*(sa)/\beta}.
$$

*Achievability: the bound is tight.* Choose at $s$ the Lemma's maximizer
$\pi_f$ (with $f(a)=V^*(sa)$), and from each successor $sa$ follow the
optimal continuation guaranteed by the inductive hypothesis. Because the
objective from $sa$ onward depends only on $sa$ — additivity plus the
Markov state — this composite policy turns both inequalities into
equalities.

Hence, with $Q^*(s,a):=V^*(sa)$, the **soft Bellman optimality equation**

$$
V^*(s)=\beta\log\sum_a\pi_{\mathrm{ref}}(a\mid s)\,e^{Q^*(s,a)/\beta}
\tag{A.3}
$$

— which is (1) — and the **soft-optimal policy**, unique at every state:

$$
\pi^*(a\mid s)=\pi_{\mathrm{ref}}(a\mid s)\,
e^{\big(Q^*(s,a)-V^*(s)\big)/\beta}.
\tag{A.4}
$$

Normalization needs no separate check: by (A.3), $e^{V^*(s)/\beta}$ *is*
the partition function $Z$. Taking logs of (A.4) gives identity (2) —
"value $=$ log-normalizer" is the exact sense in which the logits of a
soft-optimal policy are Q-values.

**Remarks.**

- *The two limits.* As $\beta\to0$ the log-sum-exp sharpens into
  $\max_{a\in\mathrm{supp}\,\pi_{\mathrm{ref}}}Q^*(s,a)$ and $\pi^*$
  collapses onto the argmax — hard RL. As $\beta\to\infty$ it flattens into
  $\mathbb E_{\pi_{\mathrm{ref}}}[Q^*]$ and $\pi^*\to\pi_{\mathrm{ref}}$ —
  pure imitation. The parameter $\beta$ interpolates between the endpoints.
- *Consistency with A.1.* Evaluate (A.1) at $\pi^*$: by (A.2) with
  $f=Q^*(s,\cdot)$, the penalty $\mathrm{KL}(\pi^*\Vert\pi_f)$ vanishes —
  they are the same distribution — leaving exactly (A.3). So the
  evaluation equation *at the optimum* is the optimality equation: (A.1)
  looks different from "the soft Bellman equation" of the textbooks only
  because it sits one maximization upstream of it.

### A.4 Consequences used in the body of the note

**(i) Telescoping — eq. (3).** Take logs of (A.4) at the sampled token
$y_j$ and use $Q^*(s_j,y_j)=V^*(s_jy_j)=V^*(s_{j+1})$:

$$
\beta\log\frac{\pi^*(y_j\mid s_j)}{\pi_{\mathrm{ref}}(y_j\mid s_j)}
=V^*(s_{j+1})-V^*(s_j).
$$

Each term is a difference of the *same* function at consecutive prefixes,
so summing over $j<t$ cancels every interior term, leaving
$V^*(s_t)-V^*(x)$; summing over the whole rollout and using the terminal
condition $V^*=R$ leaves $R(y)-V^*(x)$. That is (3). In the language of
Step 2: the log-ratio field of a soft-optimal policy is a *discrete
gradient* of the potential $V^*$ — a conservative field, which is exactly
the property drift destroys when the teacher is not soft-optimal.

**(ii) The sequence-level posterior.** Multiply (A.4) over the positions of
a complete sequence $y$: the reference factors reassemble into
$\pi_{\mathrm{ref}}(y\mid x)$ by the chain rule, and the exponents add into
the telescoped sum from (i):

$$
\pi^*(y\mid x)=\prod_t\pi^*(y_t\mid s_t)
=\pi_{\mathrm{ref}}(y\mid x)\,
\exp\Big(\tfrac1\beta\textstyle\sum_t\big(Q^*(s_t,y_t)-V^*(s_t)\big)\Big)
=\pi_{\mathrm{ref}}(y\mid x)\,e^{\big(R(y)-V^*(x)\big)/\beta}.
$$

Since $V^*(x)$ does not depend on $y$, it is precisely the normalizer:
$\pi^*(y\mid x)\propto\pi_{\mathrm{ref}}(y\mid x)\,e^{R(y)/\beta}$ — the
tempered posterior. Per-token soft-optimality and sequence-level
exponential tilting are the *same statement*, seen locally and globally;
this is the distribution the tilted actor projects onto (Step 7, item 7).

**(iii) The path-integral form of the value.** Exponentiate (A.3):

$$
e^{V^*(s)/\beta}
=\sum_a\pi_{\mathrm{ref}}(a\mid s)\,e^{V^*(sa)/\beta}
=\mathbb E_{a\sim\pi_{\mathrm{ref}}(\cdot\mid s)}\big[e^{V^*(sa)/\beta}\big].
$$

In words: the *exponentiated* value is preserved in expectation when one
token is sampled from the **reference** — it is a martingale under
reference rollouts. Apply the same identity at $sa$, then at $saa'$, and so
on; each application pushes the expectation one token deeper,

$$
e^{V^*(s)/\beta}
=\mathbb E_{a\sim\pi_{\mathrm{ref}}}\,
\mathbb E_{a'\sim\pi_{\mathrm{ref}}}\big[e^{V^*(saa')/\beta}\big]
=\cdots,
$$

until every branch reaches a complete sequence, where the terminal
condition substitutes $V^*=R$:

$$
e^{V^*(s)/\beta}
=\mathbb E_{y\sim\pi_{\mathrm{ref}}}\big[e^{R(y)/\beta}\,\big|\,s\big].
\tag{A.5}
$$

The soft value is $\beta$ times the log moment-generating function of the
reward under the **reference** continuation — computable, in principle,
without ever knowing $\pi^*$. (This is the "linearly solvable" structure
of KL-regularized control: the *nonlinear* Bellman recursion (A.3) became
a *linear* recursion in $e^{V/\beta}$.)

**(iv) The binary link — eq. (6).** For $R\in\{0,1\}$ the expectation in
(A.5) is over a two-point distribution. With
$p_{\mathrm{ref}}(s)=\Pr_{\pi_{\mathrm{ref}}}(R=1\mid s)$,

$$
\mathbb E\big[e^{R/\beta}\,\big|\,s\big]
=(1-p_{\mathrm{ref}})\,e^{0}+p_{\mathrm{ref}}\,e^{1/\beta}
=1+p_{\mathrm{ref}}(s)\,\big(e^{1/\beta}-1\big),
$$

so

$$
V^*(s)=\beta\log\!\big(1+p_{\mathrm{ref}}(s)\,(e^{1/\beta}-1)\big).
\tag{A.6}
$$

This is link (6) with its probability identified precisely: **the success
rate of the reference continuation from $s$**, not of the current student.
Range check: $p\in[0,1]\iff V^*\in[0,1]$ for every $\beta$, which is why a
sigmoid head behind this link spans exactly the achievable value range. The
link's shape also explains Step 4's soft-vs-hard calibration caveat: as
$\beta\to\infty$, $V^*\approx p$ (value $=$ success rate); as $\beta\to0$,
$V^*\to\mathbf 1[p>0]$ (any reachable success is worth full value). The
mixed target (8) moves the head from the first reading (teacher potential,
value units through the link) toward realized student returns; the residual
between "reference success" and "student success" semantics is absorbed by
that same regression.

**Provenance.** All of this is classical: maximum-entropy / soft-optimal
control (Ziebart 2010), linearly-solvable MDPs and path-integral control
(Todorov 2007; Kappen 2005), the RL-as-inference view (Levine 2018, survey).
The finite-horizon token-MDP specialization above is everything this note
needs.
