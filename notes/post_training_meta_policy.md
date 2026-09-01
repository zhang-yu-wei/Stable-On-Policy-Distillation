# A Meta-Policy for Selecting Post-Training

**Status:** design note  
**Date:** 2026-08-05

## Executive decision

The first version should be a **contextual algorithm selector**, not a language
model that invents an unrestricted training recipe.

Given a new task, the frozen meta-policy observes the problem definition,
summary statistics of the available data, and a small standardized probe of the
initial student and teacher. It then selects one of four fully specified
operators:

1. no training;
2. filtered SFT;
3. on-policy RL;
4. on-policy RL plus OPD.

Everything else—student, teacher, optimizer, compute budget, data budget, and
the internal hyperparameters of each operator—should initially be fixed. This
turns the problem into supervised algorithm selection with four actions. It is
much easier to train, audit, and falsify than a free-form research agent.

The best immediate **systems testbed** is `/mnt/officeqa-rl`, because it already
contains SFT, GRPO, GRPO+OPD, asynchronous agent rollouts, remote teacher
scoring, held-out evaluation, and pass@k. It is not a sufficient **scientific
testbed** by itself: a router trained and tested on one OfficeQA distribution
can memorize domain-specific rules. The scientific testbed must contain many
task distributions and use held-out task-family evaluation.

## 1. Problem statement

For a task \(\tau\), let:

- \(\theta_0\) be the initial student;
- \(D_\tau\) be the task's training distribution and resources;
- \(z_\tau\) be the observable task card and initial probe results;
- \(a\) be a post-training recipe;
- \(U_a(\theta_0,D_\tau)\) be the resulting trained model;
- \(J_\tau\) be performance on a hidden evaluation distribution.

Train a meta-policy to predict the improvement from each recipe:

\[
Q_\phi(z_\tau,a) \approx
\mathbb{E}\left[J_\tau(U_a(\theta_0,D_\tau))-J_\tau(\theta_0)\mid z_\tau\right].
\]

At deployment, the meta-policy is frozen and selects

\[
a^*(z_\tau)=\arg\max_a
\left(Q_\phi(z_\tau,a)-\lambda C_\tau(a)\right),
\]

where \(C_\tau(a)\) is training and teacher-inference cost. A safety term can
also penalize predicted loss on an out-of-domain retention set.

This is a meta-learning problem only at the outer level. The inner learner is
ordinary SFT, RL, or OPD. On a target task, a bounded initial probe is allowed,
but the meta-policy itself is not updated.

## 2. What should the first action space contain?

### Version 0: one categorical decision

| Action | Fixed operator | When it should tend to help |
|---|---|---|
| `NOOP` | Keep \(\theta_0\) | Training signal is weak, incompatible, or likely to damage transfer |
| `SFT` | Filtered teacher demonstrations, fixed epochs and LR | Good demonstrations exist; student success is too low for stable RL |
| `RL` | Cold-start on-policy GRPO, fixed rollout and update budget | Verifier is reliable and student rollouts have nonzero outcome variance |
| `RL_OPD` | The same GRPO recipe plus fixed OPD coefficient | Student visits useful states and the teacher gives compatible token-level guidance |

`NOOP` is necessary. The OfficeQA results already show that a recipe can improve
in-domain accuracy while severely harming spreadsheet transfer. Without a
no-training arm, the router is forced to choose a harmful intervention.

The current OfficeQA implementation's “OPD” action is **RL+OPD**, not a pure
standalone OPD trainer: teacher/student log-probability differences modify the
token advantages in the GRPO update. The first benchmark should name this
action accurately.

`SFT -> RL` is an important later action, but adding it immediately confounds
method selection with a two-stage budget allocation. It should enter only after
the four-way selector beats the best fixed recipe on held-out tasks.

### Version 1: two restricted knobs

After version 0 succeeds, add only:

- off-policy/on-policy budget split, from a small grid such as
  \(b_{off}\in\{0,0.25,0.5,0.75,1\}\);
- OPD strength, from a small calibrated grid.

These correspond directly to the rollout controller and trust weight developed
in the [mixed-policy survey](mixed_policy_rollouts_survey.md) and
[principled selector](principled_selector.md). A state-triggered mixture is
preferable to a raw training-step schedule because its meaning transfers more
cleanly across tasks.

### Defer these choices

Do not initially let the meta-policy simultaneously select:

- the student model;
- the teacher model;
- arbitrary datasets or generated examples;
- optimizer, LR, batch size, clipping, or rollout temperature;
- prompts, reward functions, or free-form training code.

Each creates counterfactual branches and makes it unclear whether the router
learned a transferable principle or exploited a benchmark-specific
correlation. Add one factor group at a time in the order: route, mixture/loss
strength, data subset, teacher, student, then free-form recipe construction.

## 3. What should the router observe?

Use a fixed probe budget and expose measured state rather than only raw task
text. A compact task card should contain:

### Task and resource features

- problem and output type;
- whether evaluation is executable, exact-match, model-graded, or human-graded;
- train-set size and demonstration availability;
- outcome sparsity and estimated verifier noise;
- context, response, trajectory-length, and compute constraints.

### Initial student probe

- pass@1 and pass@k;
- group success fraction \(\hat p\) and reward variance;
- pass@k minus pass@1 as a crude exploration headroom measure;
- entropy, response length, truncation, tool errors, and step-limit rate;
- failure clusters from a fixed sample of initial outputs.

### Demonstration and teacher probe

- teacher pass rate and student-teacher performance gap;
- student NLL/surprisal on demonstrations;
- teacher/student token log-ratio statistics on student rollouts;
- top-k overlap or ranking margin when available;
- a prefix-compatibility certificate measuring teacher drift from student-visited
  states;
- teacher cost and latency.

These features encode useful routing priors:

- **SFT:** demonstrations are good and learnable, while reward is sparse or the
  student has nearly zero success;
- **RL:** a trusted verifier exists and on-policy samples contain outcome
  variance;
- **RL+OPD:** RL has useful state visitation and a compatible teacher supplies
  additional token-level preference;
- **NOOP:** probes predict little gain, high teacher mismatch, or retention
  damage.

OPD supplies a dense **teacher preference signal**, not a dense estimate of
task value. It can shape every scored token even when terminal rewards are
sparse, but it does not by itself establish that a trajectory solves the task.
For that reason the default OPD arm should retain the outcome-RL anchor.

## 4. Testbed recommendation

No single existing benchmark provides enough tasks, matched SFT/RL/OPD
operators, and counterfactual runs. Use a two-layer testbed.

### Layer A: OfficeQA as the integration test

`/mnt/officeqa-rl` already provides most of the inner-loop machinery:

- Qwen3.5-4B student and fixed agent interface;
- filtered SFT;
- cold-start GRPO;
- GRPO+OPD with per-turn or per-sample remote teacher scoring;
- synchronous and asynchronous rollouts;
- exact hidden graders, pass@k, trajectory logs, and W&B instrumentation;
- OfficeQA Pro plus SpreadsheetBench 1 and 2 transfer evaluation.

It also supplies a useful motivating counterexample. On the August 5 results,
cold-start GRPO improves OfficeQA pass@1 from 0.095 to 0.295 and improves
SpreadsheetBench 1 strict accuracy from 28.5% to 33.5%. SFT+GRPO reaches the
best 50-step OfficeQA pass@1, 0.343, but falls to 14.0% on SpreadsheetBench 1
and 0% strict accuracy on SpreadsheetBench 2. The in-domain winner is therefore
not necessarily the best post-training decision.

OfficeQA should validate the harness, logging, equal-budget execution, feature
extraction, and action semantics. It cannot on its own demonstrate adaptation
to new task families.

### Layer B: a PostTrain-Router Gym for the scientific claim

Construct 20–50 meta-task distributions, all using the same initial student,
training budget, and operator implementations. They should cover at least three
signal regimes:

1. high-quality demonstrations but sparse/noisy reward;
2. reliable verifier and intermediate student success;
3. sparse terminal success but a strong compatible teacher.

Every meta-task must have:

- a train distribution for the inner update;
- a probe split visible to the meta-policy;
- a hidden in-distribution evaluation split;
- preferably an out-of-domain retention or transfer split;
- matched runs of every action with multiple seeds.

Split train and test by **task family**, not by prompt. Otherwise the selector
can memorize dataset identity. Task names should be hidden or randomized in the
statistics-only ablation.

The [autoresearch benchmark survey](autoresearch_benchmarks_survey.md) suggests
two useful later integrations:

- [Agent² RL-Bench](https://arxiv.org/abs/2604.10547) is the closest external
  match because each task packages a base model, task data, grader, workspace,
  and fixed compute budget;
- [ResearchGym](https://arxiv.org/abs/2602.15112) gives more heterogeneous
  containerized research tasks, but its larger design space makes it a better
  transfer benchmark than a version-0 training environment.

RE-Bench and MLRC-Bench are valuable final external-validity tests, but their
small number of expensive, open-ended environments makes them poor sources of
the dense counterfactual matrix needed to train the first router.

## 5. How to train the meta-policy

For every meta-training task, run all four actions under matched budgets and
seeds. This produces a full-information table rather than an RL problem:

| Task features | NOOP score | SFT score | RL score | RL+OPD score | Cost/retention |
|---|---:|---:|---:|---:|---:|
| \(z_\tau\) | ... | ... | ... | ... | ... |

Fit one outcome model \(Q_\phi(z,a)\) per action or a shared model with an
action embedding. Start with a regularized linear model or gradient-boosted
trees; then compare a small MLP. Supervised regression is preferable to a
contextual bandit while all counterfactual outcomes are available.

Use normalized held-out progress per unit cost as the target, with an explicit
retention penalty:

\[
y_{\tau,a}=\frac{J^{ID}_{\tau,a}-J^{ID}_{\tau,0}}
{C_{\tau,a}}
-\gamma\max(0,J^{OOD}_{\tau,0}-J^{OOD}_{\tau,a}).
\]

If exhaustive runs become too expensive, use successive halving to eliminate
clearly poor actions, but preserve some randomized complete task-action blocks
so the estimator is not trained only on selection-biased observations.

## 6. Role of a large outer policy such as Qwen-Max

Use the large policy as a **task analyst behind a strict schema**, not as the
sole decision rule.

Recommended harness:

1. deterministic code computes numerical probe features;
2. Qwen-Max reads the problem definition and representative initial outputs and
   emits a structured task card plus failure taxonomy;
3. a small calibrated router consumes both and predicts each allowed action's
   gain, cost, and retention risk;
4. a validator chooses only from registered recipes and launches the existing
   OfficeQA scripts;
5. hidden evaluation results are appended to the outer-training table.

A useful ablation is Qwen-Max zero-shot selection versus the numeric router
versus Qwen-Max features plus the numeric router. This reveals whether semantic
problem understanding adds value beyond success-rate and compatibility
statistics.

The remote teacher used for OPD must expose aligned log-probabilities for the
student's generated tokens. If a Qwen-Max endpoint cannot do that, it can still
serve as the outer task analyst or demonstration generator, while the existing
SGLang-compatible teacher remains the OPD scorer.

Example version-0 output schema:

```json
{
  "action": "RL_OPD",
  "predicted_delta": 0.08,
  "predicted_retention_delta": -0.01,
  "confidence": 0.71,
  "evidence": {
    "student_pass_at_1": 0.12,
    "reward_variance": 0.09,
    "teacher_pass_at_1": 0.61,
    "prefix_compatibility": 0.84
  },
  "recipe_id": "officeqa_grpo_opd_v0"
}
```

The schema contains a recipe ID, not arbitrary shell arguments.

## 7. Evaluation and falsification

Report:

- regret to the per-task oracle;
- improvement over the best single global recipe;
- action-selection accuracy only as a secondary metric;
- in-domain score, pass@k, and diversity;
- retention/transfer loss;
- training and teacher cost;
- calibration of predicted improvements;
- variance and catastrophic-run rate.

Required baselines:

1. best fixed global action;
2. random action;
3. the hand-derived selector from `principled_selector.md`;
4. Qwen-Max direct zero-shot selection;
5. learned statistics-only router;
6. Qwen-Max task card plus learned router;
7. per-task oracle, as an upper bound.

The main success criterion is not that the router predicts the winning label on
OfficeQA slices. It must beat the best fixed recipe in normalized improvement
on held-out task families while avoiding excess transfer damage.

## 8. Staged implementation

### Stage 0: make actions comparable

- Register `NOOP`, `SFT`, `RL`, and `RL_OPD` as immutable recipe IDs.
- Equalize GPU/token budget and evaluation protocol.
- Add a shared manifest recording task, seed, starting checkpoint, data hash,
  recipe, cost, and all hidden scores.
- Extract the numerical probe card from existing trajectory and grader logs.

### Stage 1: OfficeQA retrospective and prospective harness

- Import existing matched runs where possible, but flag unmatched historical
  runs as observational evidence.
- Run missing action/seed cells prospectively.
- Use OfficeQA Pro as ID evaluation and both SpreadsheetBench suites as
  retention tests.
- Verify that the router can at least learn sensible signal-regime boundaries;
  do not claim cross-task meta-generalization.

### Stage 2: held-out task families

- Add adapters for a controlled multi-task suite and then Agent² RL-Bench.
- Collect the task-by-action counterfactual matrix.
- Train the simple router and evaluate frozen on held-out families.

### Stage 3: cautiously increase freedom

- Add off-policy/on-policy budget split.
- Add low/high OPD trust.
- Add `SFT -> RL` and later `SFT -> RL+OPD`.
- Only then study data selection, teacher choice, and student-model choice.

## 9. Main risks

- **Too few independent tasks:** many prompts from one benchmark do not create
  many meta-learning examples.
- **Leakage:** dataset name, public validation results, or post-training outputs
  can reveal the answer to the router.
- **Unmatched budgets:** SFT, RL, and OPD comparisons are meaningless if one arm
  receives substantially more tokens or teacher compute.
- **OPD mistaken for value:** dense token preference can confidently reinforce
  a teacher's artifacts or an incompatible trajectory distribution.
- **Historical-run confounding:** existing checkpoints differ in seeds,
  schedules, step caps, and sometimes evaluation temperature.
- **Optimizing only ID score:** this selects recipes such as the current
  SFT+GRPO run despite severe transfer loss.
- **Outer-model overreach:** a persuasive free-form explanation from a large
  model is not a calibrated prediction of post-training gain.

## Bottom line

Build the first meta-policy as a four-action, cost-aware supervised router. Use
OfficeQA to make the machinery real, then require held-out task-family success
in a controlled PostTrain-Router Gym before adding more degrees of freedom.
Qwen-Max is most useful initially for producing a semantic task card and failure
analysis; the final recipe choice should remain constrained, calibrated, and
grounded in measured student/teacher probes.
