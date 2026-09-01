# VTR-OPD: Value-Triggered Rewind for Teacher–Student Mixed Rollouts

Design note, 2026-08-03. Sources: TCOD
(`papers/agentic_self_distillation/TCOD_Temporal_Curriculum_OPD.pdf`) and
DAgger
(`papers/foundations/DAgger_Reduction_of_Imitation_Learning_to_No_Regret_Online_Learning.pdf`).
Estimators for the quantities used here are in
`teacher_intervention_blueprint.md`.

## 1. Why the switching rule can be state-based

TCOD switches control between teacher and student on a clock:
$k = k_{\text{start}} + \lfloor n/\eta \rfloor$, where $n$ is the training
step. F2B truncates the student's rollout at turn $k$; B2F has the teacher
execute the first $L-k$ turns of a pre-collected success trajectory. The
schedule is open-loop — it does not look at what is happening in the
current trajectory — and the paper's own limitations section proposes an
adaptive replacement based on measured student progress.

DAgger tells us which part of the design carries the guarantee. Its rollout
policy is a mixture $\pi_i = \beta_i \pi^* + (1-\beta_i)\hat\pi_i$, and the
analysis (Lemma 4.1, Theorem 4.1) requires only that the average expert
usage $\bar\beta_N = \frac{1}{N}\sum_i \beta_i \to 0$; the extra cost of
mixing is bounded by $\frac{2\ell_{\max}}{N}[n_\beta + T\sum_{i>n_\beta}
\beta_i]$, which vanishes for any decaying schedule. Nothing in the bound
constrains *which states* the expert controls. So the placement of teacher
turns is a free design variable, and we can spend it where a measured
signal says the student needs it — provided a budget forces the total
teacher usage to anneal to zero.

VTR-OPD (Value-Triggered Rewind OPD) is that algorithm: TCOD-B2F's
mechanics (teacher navigates, student continues, teacher turns carry no
gradient) with the clock replaced by a value-based trigger, plus a DAgger
budget that preserves the guarantee.

## 2. Algorithm overview

Notation follows TCOD: history state $h_t$, student $\pi_\theta$, teacher
$\pi_\phi$, horizon $T$. Two value estimates: $\hat V(h)$ = success-to-go
if the **student** continues from $h$, and $\hat V_T(h)$ = success-to-go if
the **teacher** continues from $h$.

One iteration in words: the student rolls out as usual. A monitor watches
$\hat V(h_t)$ after every turn. If the student has spent $m$ consecutive
turns below a value threshold — it is not merely in a hard state, it is
*staying* in a low-value region — the trajectory is rewound to the latest
earlier state the teacher can still recover from, and the teacher generates
turns from there (no gradient) until the student's value recovers, at which
point the student resumes. The abandoned low-value segment is kept as
training data (those are exactly the mistake states DAgger says to label),
subject to a per-turn off-support filter. A global annealed budget caps the
fraction of teacher turns per batch, which is what makes the DAgger
guarantee go through.

Five rules define the algorithm:

**Trigger.** Fire at turn $t$ when $\hat V(h_j) < \delta$ for all
$j \in \{t-m+1, \dots, t\}$. Low value alone is not enough (a hard task
starts with low value); the $m$-turn persistence is the "repeatedly
encountering the same low-value region" condition, and it is what
distinguishes stuck from slow. When $\hat V$ is expensive, a cheap
pre-filter (a near-duplicate observation or action in the last $m$ turns)
decides when to evaluate it.

**Anchor (rewind target).** Search backward from $t$ for the latest turn
whose state the teacher can recover from:
$t^* = \max\{\, t' \le t : \hat V_T(h_{t'}) \ge \delta_T \,\}$.
Reset the environment to $h_{t^*}$ (snapshot, or deterministic replay of
$a_0,\dots,a_{t^*-1}$). If no such $t'$ exists, the task is beyond the
teacher from every visited state — truncate and drop the trajectory, since
teacher supervision is unreliable everywhere on it. Anchoring on
$\hat V_T$ rather than $\hat V$ is the point: the rewind must land where
the teacher's continuation (and its labels) are trustworthy.

**Handback.** From $h_{t^*}$ the teacher generates turns with stopped
gradient. After each teacher turn, if $\hat V(h) \ge \delta'$ with
$\delta' > \delta$ (hysteresis, so control does not oscillate), the student
resumes. If the teacher spends $M$ turns without reaching $\delta'$, the
anchor test was wrong — truncate the trajectory.

**Budget.** Per batch at iteration $n$, teacher turns may not exceed a
fraction $b_n$ of total turns, with $b_n \to 0$ (e.g. linear decay to zero
over the first $N_0$ iterations); additionally at most $C$ interventions
per trajectory (1–2). Once the budget is spent, triggers are ignored and
rollouts are pure student. This cap is the $\bar\beta_N \to 0$ condition;
late in training it rarely binds because a better student triggers less,
but it holds regardless of estimator errors, and $b_n = 0$ at the end
removes the train–test mismatch exactly as TCOD-B2F's anneal to a zero
prefix does.

**Loss.** Standard OPD objective over **student-generated turns only**,
including the abandoned segment $t^*..t$ from before the rewind:

$$
\mathcal L(\theta) = \sum_{t \,\in\, \text{student turns},\ \kappa\text{-kept}}
D_{KL}\big(\pi_\phi(\cdot \mid h_t) \,\|\, \pi_\theta(\cdot \mid h_t)\big),
$$

where a turn is $\kappa$-kept iff its per-turn KL is $\le \kappa$. The
filter is TCOD's finding applied at turn granularity instead of by clock:
per-turn KL grows as the student drifts off the teacher's support
(TCOD Fig. 2d), and turns past the cap carry unreliable labels. The KL is
computed for the loss anyway, so the filter is free. Teacher turns
contribute no gradient.

```
Algorithm VTR-OPD — one training iteration n
Input: student π_θ, teacher π_φ, value estimators V̂, V̂_T, environment E
Params: trigger threshold δ, window m, handback threshold δ' > δ,
        teacher-recovery threshold δ_T, per-intervention cap M,
        per-trajectory cap C, batch budget b_n (annealed to 0),
        off-support KL cap κ

for each task in the batch:
    h_0 ← reset(E);  t ← 0;  c ← 0;  control ← student
    while not terminal and t < T:
        if control = student:
            a_t ~ π_θ(·|h_t)
        else:
            a_t ~ π_φ(·|h_t)                        # stop gradient
        execute a_t;  observe h_{t+1};  t ← t+1
        if control = teacher:
            if V̂(h_t) ≥ δ': control ← student       # handback
            else if M teacher turns spent: truncate trajectory
        else if V̂(h_j) < δ for all j ∈ {t−m+1..t}:  # trigger
            if c < C and batch teacher-turn fraction < b_n:
                t* ← max{ t' ≤ t : V̂_T(h_{t'}) ≥ δ_T }   # anchor
                if t* undefined: truncate trajectory       # beyond teacher
                keep turns t*..t as failed-branch training data
                rewind E to h_{t*};  t ← t*;  c ← c+1
                control ← teacher
            else:
                truncate trajectory

L(θ) = Σ over student turns (kept path + failed branches) with
       per-turn KL ≤ κ of D_KL(π_φ(·|h_t) ∥ π_θ(·|h_t))
θ ← θ − ∇_θ L
```

## 3. Why it is principled

**DAgger reduction.** The rollout policy at iteration $n$ is a
state-gated mixture $\pi_n(\cdot \mid h) = g_n(h)\,\pi_\phi +
(1-g_n(h))\,\pi_\theta$ with $g_n \in \{0,1\}$ set by
trigger/anchor/handback. Let $\beta_n$ be the expected per-turn teacher
rate; the budget enforces $\beta_n \le b_n$. Lemma 4.1's argument gives
$\|d_{\pi_n} - d_{\pi_{\theta_n}}\|_1 \le 2T\beta_n$, so Theorem 4.1
applies unchanged: training on the aggregated (state, teacher-logit) data
with any no-regret learner yields a policy in the sequence with surrogate
loss $\le \epsilon_N + \gamma_N + \frac{2\ell_{\max}}{N}[n_\beta +
T\sum_{i>n_\beta} b_i]$ under its own state distribution, and the last term
vanishes under the anneal. The trigger, anchor, and handback never enter
the bound — they decide only where the allowed $\beta$ mass is spent. That
is the license to spend it greedily where the measured value says the
student is stuck.

**Where the $\beta$ mass goes matters for the constant.** DAgger's cost
bound (Theorem 2.2) is $J(\pi) \le J(\pi^*) + uT\epsilon$, where $u$ bounds
the expert's cost-to-go disadvantage at visited states; it is small only
where the expert can recover. The anchor rule targets exactly this:
interventions are placed at the latest state with $\hat V_T \ge \delta_T$,
i.e. where teacher recovery — and hence teacher supervision — is
meaningful, and branches with no such state are dropped rather than
labeled. The failed-branch data that is kept (states the student reached,
labels from a recoverable region's teacher) is the recovery data DAgger's
Mario experiment identifies as the reason iterative methods beat behavior
cloning: the supervised policy never learned to get unstuck because it
never trained on stuck states.

**TCOD's findings, at state granularity.** TCOD's diagnosis is that
compounding errors push the student off the teacher's support, per-turn KL
grows with turn index, and supervision degrades. Its remedy bounds the
student's exposure by a clock. VTR-OPD bounds the same quantity by
measurement: the trigger removes low-value absorbing segments from the
rollout distribution, the anchor restarts from teacher-supported states,
and the $\kappa$ filter drops individual off-support turns instead of whole
horizon ranges. Both TCOD variants are special cases: F2B is the trigger
"fire at fixed turn $k(n)$" with truncation instead of takeover; B2F is
the trigger "always fire at $t=0$" with the anchor drawn from a
pre-collected success trajectory and budget $L - k(n)$.

## 4. Estimators and assumptions

$\hat V$ and $\hat V_T$: with a forkable environment, probe rollouts ($k$
forks from $h$, count successes — `teacher_intervention_blueprint.md` §1);
otherwise a value head regressed on rollout outcomes, with the teacher's
prefix perplexity as a cheap proxy for $\hat V_T$. The trigger's visit
counting can also be maintained across the batch (near-duplicate low-value
states seen in many trajectories raise intervention priority), but the
within-trajectory $m$-window is the base rule.

Rewind requires resetting the environment to an intermediate state: either
a snapshot or deterministic replay of the action prefix (the TCOD
benchmarks — ALFWorld, WebShop, ScienceWorld — support replay).

Estimator error degrades compute, not the gradient. A false trigger spends
budgeted teacher turns that carry no gradient; a missed trigger leaves that
trajectory as vanilla OPD; a wrong anchor is caught by the $M$-turn
handback cap. In every case the loss remains the on-student-distribution
KL, so value-estimation mistakes cost coverage and teacher compute but do
not bias the training signal.
