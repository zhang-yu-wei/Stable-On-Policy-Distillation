# Autoresearch Benchmark Survey

**Last updated:** 2026-08-05

## Scope

"Autoresearch" is used for several different capabilities that should not be
collapsed into one score:

1. **Research retrieval:** find the right papers or evidence.
2. **Research judgment:** choose a promising direction from incomplete evidence.
3. **Research implementation:** translate an idea or paper into working code.
4. **Closed-loop experimentation:** form hypotheses, edit code, run experiments,
   interpret results, and allocate the remaining budget.
5. **Replication or re-discovery:** reconstruct an existing result with some or
   all of the original solution hidden.
6. **Scientific communication:** write a correct, well-supported report.
7. **Scientific integrity:** reject bad premises, avoid fabricated results, and
   expose uncertainty and failed experiments.

For a Karpathy-style loop that repeatedly modifies a training program against a
fixed validation metric, category 4 is the central capability. Retrieval and
report-writing benchmarks are useful diagnostics but weak primary measures.

## Bottom line

There is no single mature benchmark for the complete autoresearch lifecycle.
The best current suites cover different slices:

- **Best direct fit for this repository:**
  [Agent² RL-Bench](https://arxiv.org/abs/2604.10547), because it measures an
  agent closing an actual post-training loop under a fixed budget.
- **Best clean end-to-end research environment:**
  [ResearchGym](https://arxiv.org/abs/2602.15112), because it withholds the
  published method but preserves datasets, baselines, and executable graders.
- **Best time-budgeted human comparison:**
  [RE-Bench](https://proceedings.mlr.press/v267/wijk25a.html).
- **Best objective test of novel ML improvement:**
  [MLRC-Bench](https://arxiv.org/abs/2504.09702).
- **Best broad scientific suite:**
  [AstaBench](https://arxiv.org/abs/2510.21652).
- **Best literature-retrieval diagnostic:**
  [AutoResearchBench](https://arxiv.org/abs/2604.25256).
- **Best integrity stress test:**
  [PseudoBench](https://arxiv.org/abs/2606.18060).

For OPD research, the recommended evaluation is therefore a **small custom,
execution-grounded OPD suite**, cross-checked with Agent² RL-Bench and one more
general environment such as ResearchGym or RE-Bench. A report benchmark should
not be the headline score.

## 1. Closed-loop and end-to-end research benchmarks

| Benchmark | Scale and setting | Evaluation signal | What it measures well | Main limitation |
|---|---|---|---|---|
| [ResearchGym](https://arxiv.org/abs/2602.15112) (2026) | 5 containerized environments and 39 subtasks derived from oral/spotlight papers; the proposed method is withheld while data, baselines, and graders remain | Executed task metrics and subtask completion | Hypothesis-to-experiment loops in real repositories; long-horizon resource management | Small number of source papers; re-discovery can reward reconstructing a known answer rather than genuinely new science |
| [AIRS-Bench](https://arxiv.org/abs/2602.06855) (2026) | 20 tasks from recent ML papers across language modeling, math, bioinformatics, and forecasting; no baseline code | Domain-specific task performance against human results and ceilings | Broad, from-scratch research lifecycle and sequential versus parallel scaffolds | Setup burden and cross-task compute comparability; only 20 tasks |
| [ResearchClawBench](https://arxiv.org/abs/2606.07591) (2026) | 40 tasks in 10 scientific domains; target paper hidden, related literature and raw data provided | Expert-curated, weighted multimodal rubrics for target-level re-discovery | End-to-end workflow coverage beyond ML and diagnosis of protocol/evidence mismatch | Rubric score is less objective than a hidden executable metric; target-paper re-discovery is not open-ended novelty |
| [RE-Bench](https://proceedings.mlr.press/v267/wijk25a.html) (2025) | 7 open-ended ML research-engineering environments; 71 eight-hour attempts from 61 human experts | Continuous environment score under matched time budgets | Score-versus-time curves and direct human comparison | Only seven environments; more research engineering than literature, theory, or writing |
| [MLRC-Bench](https://arxiv.org/abs/2504.09702) (2025) | 7 ML research-competition tasks | Objective competition metric and fraction of the gap from baseline to top human solution closed | Whether proposed and implemented ideas actually improve a strong baseline | Narrowly ML and competition-shaped; small task count; public competitions raise contamination risk |
| [MLR-Bench](https://arxiv.org/abs/2505.19955) (2025) | 201 open-ended workshop research tasks; stages include ideation, proposal, experiments, and paper writing | Rubric-based multi-LLM reviewer, validated against human review | Broad open-ended research outputs and stagewise diagnosis | LLM-judge sensitivity; the paper reports fabricated or invalid experiments in roughly 80% of coding-agent cases |
| [AstaBench](https://arxiv.org/abs/2510.21652) (2025/2026) | 2,400+ problems spanning literature, data, planning, tool use, coding, search, and end-to-end research | Suite-specific executable and structured measures with standardized scientific tools; cost tracked | Breadth, controlled tools, agent/model/cost comparisons, and reproducibility | An aggregate can hide weak end-to-end closure; many subtasks measure components rather than autonomous research as a whole |
| [Agent² RL-Bench](https://arxiv.org/abs/2604.10547) (2026) | 6 tasks across 3 levels, from static rule-based training to judge-based optimization and online RL with trajectory collection | Improvement of a base model through a grading API under a fixed budget | Autonomous design, implementation, debugging, and execution of post-training pipelines | Compact diagnostic suite; large run-to-run variance and substantial compute/setup requirements |
| [AARRI-Bench](https://arxiv.org/abs/2606.07462) (2026) | Granular "research intern" scenarios | Task success on diligence, methodology, ethics, and nuanced research behavior | Research professionalism and small but consequential mistakes | Does not directly test whether a novel experiment improves a scientific result |

### Interpretation

The strongest design shared by ResearchGym, RE-Bench, MLRC-Bench, and Agent²
RL-Bench is an **external, executable objective inside a budgeted environment**.
This is harder to game and easier to reproduce than asking a judge model whether
an idea or paper appears novel. Their common weakness is coverage: high-fidelity
environments are expensive to construct, so task counts are small.

## 2. Replication, implementation, and data-analysis benchmarks

These are valuable component tests. They are not complete autoresearch
benchmarks because the research question or target contribution is mostly given.

| Benchmark | Scale and task | Evaluation | Best use | Main limitation |
|---|---|---|---|---|
| [PaperBench](https://arxiv.org/abs/2504.01848) (2025) | Replicate 20 ICML 2024 oral/spotlight papers from scratch; 8,316 rubric leaves | Author-developed hierarchical rubrics graded by an LLM judge | Long-horizon paper understanding, codebase creation, and experiment execution | Replication, not novelty; costly long rollouts; grading is partly model-mediated |
| [CORE-Bench](https://arxiv.org/abs/2409.11363) (2024) | 270 tasks from 90 papers in computer science, social science, and medicine, at three difficulty levels | Reproduce reported results using supplied code/data | Reproducibility, environment repair, and artifact verification | Original repository substantially narrows the solution; does not test hypothesis generation |
| [ScienceAgentBench](https://arxiv.org/abs/2410.05080) (2024) | 102 expert-validated tasks from 44 papers in four disciplines | Execution and result metrics for a self-contained Python program, plus cost | Real data-driven scientific analysis and coding | Reformulates discovery as a bounded program-generation task |
| [ResearchCodeBench](https://arxiv.org/abs/2506.02314) (2025) | 212 coding challenges based on novel contributions from 2024–2025 ML papers | Hidden executable tests | Fast test of faithful paper-to-code translation and contamination-resistant recent ideas | No iterative experiment loop, resource allocation, or hypothesis choice |
| [MLE-bench](https://openai.com/index/mle-bench/) (2024) | 75 Kaggle competitions | Medal thresholds and public competition scores | End-to-end ML engineering, data preparation, training, and scaling studies | Mostly established predictive modeling; leaderboard improvement is not equivalent to scientific contribution |

These suites make good **gates**. An agent that cannot implement a method, run an
analysis, or reproduce a result is unlikely to conduct reliable end-to-end
research. Passing them, however, does not establish the ability to choose a new
research direction.

## 3. Literature, web research, and report benchmarks

| Benchmark | Scale and task | Evaluation | Best use | Why it is not a primary autoresearch score |
|---|---|---|---|---|
| [AutoResearchBench](https://arxiv.org/abs/2604.25256) (2026) | Deep search for one target paper and wide search for an unknown set of qualifying papers | Target accuracy for deep search; set IoU for wide search | Scientific literature discovery with fine-grained full-text clues | Stops before experiments or scientific claims are tested |
| [DeepResearch Bench](https://arxiv.org/abs/2506.11763) (2025) | 100 expert-written tasks across 22 domains, split between English and Chinese | RACE report-quality dimensions and FACT citation metrics | Long-form synthesis, instruction following, citation accuracy, and coverage | Open-web drift and judge dependence; report quality can be high without new evidence or experiments |
| [DeepResearch Bench II](https://arxiv.org/abs/2601.08536) (2026) | Reports evaluated against expert-report-derived rubrics | Information recall, analysis, and presentation | More interpretable report diagnosis than coarse holistic judging | Still evaluates a research report rather than a closed scientific loop |
| [DR³-Eval](https://arxiv.org/abs/2604.14683) (2026) | Multimodal, multi-file report tasks in a static sandbox containing evidence, distractors, and noise | Recall, factual accuracy, citation coverage, instruction following, and depth | Reproducible retrieval and hallucination testing without live-web drift | Sandbox synthesis rather than hypothesis and experiment execution |
| [IDRBench](https://arxiv.org/abs/2601.06676) (2026) | Underspecified research goals with simulated clarification and feedback | Quality/alignment gains versus interaction turns and token cost | Whether an agent asks useful questions and adapts to evolving intent | Measures user alignment around report generation, not scientific discovery |
| [BrowseComp](https://openai.com/index/browsecomp/) (2025) | 1,266 hard, entangled web fact-finding questions | Exact-answer accuracy | Persistent browsing and obscure multi-hop retrieval | Closed-answer web QA; no report, experiment, or novelty |

The central lesson is that **retrieval, synthesis, and discovery are separate**.
BrowseComp can be solved without producing a defensible report; a report suite
can be scored without executing an experiment; and neither establishes that the
agent can improve a model under a fixed compute budget.

## 4. Judgment and scientific-integrity benchmarks

| Benchmark | Task | Evaluation | Role in a complete evaluation |
|---|---|---|---|
| [ForeSci](https://arxiv.org/abs/2606.00644) (2026) | 500 temporally controlled tasks requiring forward-looking research decisions from a cutoff-aligned knowledge base | Post-cutoff outcomes validate decisions; evidence quality and decision correctness are separated | Tests direction selection and detects the failure mode "right evidence, wrong decision" |
| [PseudoBench](https://arxiv.org/abs/2606.18060) (2026) | 200 pseudoscientific claim-evidence pairs in five domains, run through experiments-to-writing pipelines | Resistance to the false premise and quality of the resulting research behavior | Safety/integrity stress test; catches systems that optimize a supplied objective while laundering a bad premise |

These should be treated as constraints, not optional polish. An optimizer that
achieves a high benchmark score by exploiting leakage, invalid statistics, or a
misleading premise is not a successful research agent.

## 5. Cross-benchmark failure modes

### 5.1 Proxy completion is not scientific success

Executable metrics are stronger than prose judges, but an agent may still overfit
one public validation set, exploit an evaluator, or trade away unmeasured
properties. For OPD, a gain in average accuracy can conceal repetition, length
inflation, calibration failure, OOD regression, or prohibitive inference cost.

### 5.2 Replication is not novelty

PaperBench, CORE-Bench, ResearchCodeBench, ResearchGym, and ResearchClawBench all
derive tasks from completed research. Hiding the solution reduces leakage, but a
model may reconstruct a known method from pretraining or nearby literature. These
benchmarks measure useful research skills, not clean-room scientific novelty.

### 5.3 Best-of-*k* can hide unreliability

Autoresearch systems are stochastic at the level of hypotheses, code edits,
training runs, and judge calls. Reporting only the best run rewards brute-force
sampling and hides catastrophic failures. At minimum report median, interquartile
range, valid-run rate, and the compute spent on discarded branches.

### 5.4 Wall time is not a portable budget

Two agents with the same wall-clock limit may receive very different GPU counts,
API parallelism, search access, and cached artifacts. Report wall time, accelerator
hours, model tokens, API cost, number of environment submissions, and peak
parallelism separately.

### 5.5 Public tasks become contaminated

Published task prompts, target papers, reference repositories, and winning
solutions can enter training data. Temporal cutoffs, recently published source
material, encrypted or private test cases, and hidden held-out datasets are all
needed. No single technique is sufficient.

### 5.6 LLM judges favor plausible research-shaped text

Rubric-based judges are scalable and useful for decomposition, but polished prose
can mask invalid experiments. Claims about experimental results should be checked
against generated files, logs, seeds, and evaluator outputs. An LLM judge should
never be the only verifier of an executable claim.

### 5.7 Infrastructure failures can dominate the score

Container setup, package availability, broken downloads, GPU OOMs, and evaluator
latency are part of real research, but they can also swamp the capability of
interest. Benchmarks should distinguish scientific, coding, resource-management,
and infrastructure failures.

## 6. Recommended benchmark stack for OPD autoresearch

### A. Fast gate: implementation and experiment hygiene

- A small recent subset of ResearchCodeBench-like paper-to-code tasks.
- A supplied OPD repository with deliberate bugs in masks, top-*k*
  renormalization, stop-gradient placement, padding, and sequence truncation.
- Score with hidden unit tests, a short training smoke test, and exact artifact
  checks.

This tests whether the agent can safely modify a training stack before expensive
runs are allowed.

### B. Primary score: closed-loop OPD research

Give the agent:

- a fixed base checkpoint and training stack;
- a problem statement, not a target paper or reference solution;
- train and public validation data;
- a fixed budget of GPU-hours, model/API tokens, and evaluator submissions;
- access to code editing, training, analysis, and experiment logs; and
- no access to the hidden evaluation split or private robustness tests.

Candidate task families drawn from this repository:

1. Prevent length inflation and repetition while preserving task accuracy.
2. Improve mixed math/agentic training under support-set drift.
3. Select or combine teachers under a fixed teacher-inference budget.
4. Improve learning when pass@1 is low and successful rollouts are sparse.
5. Reduce negative transfer across model scales or task mixtures.
6. Improve calibration without sacrificing correctness.

Each task should have multiple hidden instances: different seeds, datasets,
student scales, and at least one distribution shift. One visible instance is not
enough because the agent can simply optimize the validation set.

### C. External validity

Run the same agent scaffold, without benchmark-specific prompting, on:

- Agent² RL-Bench for direct post-training transfer;
- one of ResearchGym, RE-Bench, or MLRC-Bench for broader research behavior; and
- AutoResearchBench only if literature search is part of the intended agent.

### D. Integrity stress test

Include tasks with:

- a false premise;
- a metric that can be improved through a known degenerate behavior;
- a leaked-looking but invalid prior result;
- inconsistent logs and prose claims; and
- an underpowered experiment where the correct action is to report uncertainty.

The agent must be rewarded for refusing the premise, repairing the evaluator, or
reporting that the evidence is insufficient.

## 7. Suggested metrics and reporting protocol

Do not compress the evaluation to one number until the following components have
been reported separately.

### Outcome

- **Held-out normalized improvement:** improvement over the supplied baseline,
  normalized by a strong human/reference improvement where available.
- **Constraint pass rate:** accuracy gains count only if length, repetition,
  calibration, format validity, and compute constraints pass.
- **Generalization gap:** public-validation improvement minus private-test
  improvement.

### Efficiency

- **Anytime performance:** area under the best valid held-out score versus
  GPU-hour curve.
- Total GPU-hours, wall time, API tokens/cost, search calls, and grader
  submissions.
- Fraction of budget spent on failed or abandoned branches.

### Reliability

- Median and interquartile range across at least three independent agent runs.
- Valid-run rate and catastrophic-regression rate.
- Success on hidden seeds and at least one hidden environment variant.

### Scientific process

- Fraction of reported claims traceable to an actual logged experiment.
- Reproducibility from a clean container using the submitted artifact.
- Correct use of controls, ablations, and multiple training seeds where the
  budget permits.
- Number and severity of evaluator exploits, leaked-test accesses, fabricated
  results, or unsupported claims.
- Human interventions and information supplied after the run began.

### Minimum run record

Every result should publish or retain:

- agent/model/scaffold versions and prompts;
- environment and dependency lockfiles;
- allowed tools and network policy;
- hardware, concurrency, and exact budgets;
- all experiment commands, diffs, configs, seeds, metrics, and failures;
- the full trajectory or a privacy-preserving audit log; and
- aggregation rules fixed before seeing the runs.

## 8. Practical selection guide

| If the question is... | Use... |
|---|---|
| Can the agent autonomously improve an RL/post-training pipeline? | Agent² RL-Bench plus a hidden OPD-specific suite |
| Can it conduct a real code-and-experiment research loop? | ResearchGym or RE-Bench |
| Can its ideas beat strong ML baselines objectively? | MLRC-Bench or AIRS-Bench |
| Can it reproduce published research from scratch? | PaperBench |
| Can it repair and rerun supplied research artifacts? | CORE-Bench |
| Can it implement a new paper contribution? | ResearchCodeBench |
| Can it perform broad scientific assistant tasks? | AstaBench |
| Can it find the right scientific literature? | AutoResearchBench |
| Can it write a sourced analyst-style report? | DeepResearch Bench II or DR³-Eval |
| Can it choose promising directions without future leakage? | ForeSci |
| Will it resist invalid or pseudoscientific premises? | PseudoBench |

## 9. Overall assessment

The field is moving from **artifact judging** toward **environment-grounded,
budgeted evaluation**, which is the right direction. The main unresolved problem
is validity: a closed loop can optimize the benchmark without performing good
science. The most credible autoresearch evaluation therefore combines:

1. an executable hidden objective;
2. multiple held-out environments and seeds;
3. anytime score-versus-compute measurement;
4. auditable experiment provenance;
5. explicit robustness and integrity constraints; and
6. a small amount of expert review for claims that cannot be mechanically
   verified.

For this repository, the next useful step is not adopting one public leaderboard.
It is packaging several existing OPD hypotheses as hidden, containerized research
problems and using Agent² RL-Bench or ResearchGym as the external generalization
check.

## Related survey

- [AI for Auto-Research: Roadmap & User Guide](https://arxiv.org/abs/2605.18661)
  organizes systems and benchmarks across creation, writing, validation, and
  dissemination. The present note narrows that broader lifecycle view to
  benchmark design and selection for closed-loop experimental research.
