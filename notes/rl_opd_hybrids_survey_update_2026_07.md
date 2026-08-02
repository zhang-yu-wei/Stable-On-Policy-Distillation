# RL + OPD Hybrid Losses: Survey Update (July 2026)

Companion to `opd_rl_connection_and_reward_losses.md` (the original 16-paper survey)
and `advantage_tilted_opd.md` (the proposal). Compiled 2026-07-22 from a three-angle
literature sweep (broad hybrids / tilted-posterior line / exploration-preservation).
Roughly 30 new qualifying papers since the original survey. Papers already covered
there are not repeated; two search hits turned out to be duplicates under other names
(arXiv 2605.15155 = SDAR; arXiv 2602.12125 "G-OPD" = ExOPD).

---

## TL;DR: Six Design Lessons From the New Literature

1. **The additive family survived only by becoming a curriculum.** Every 2026
   additive paper anneals or gates its distillation coefficient to zero (TGPO linear
   decay, SDPG trapezoid, CEPO λ→0, AMR-SD annealed, KDRL annealed, SSOPD frontier
   weight). Nobody found a *static* coefficient that balances the two gradients —
   the field has implicitly conceded objection (a): a fixed additive combination has
   no good equilibrium, so it is used as a warm-up schedule, not an objective.

2. **The one working fix for the units problem inside the additive family is
   whiten-after-mixing.** CoDistill-GRPO folds the teacher log-ratio into the reward
   *before* group normalization, so the combined signal is standardized jointly.
   Crude, but it is the honest additive-family baseline any new method should beat.

3. **The "teacher as weight" family keeps growing and keeps its ceiling.** CEPO,
   RLCSD, AMR-SD, PBSD, GEAR — and now Distilled RL (2607.17247, Jul 19) — all
   preserve "reward controls sign, teacher controls magnitude" (objection (c) intact). Their genuinely new trick is **contrastive
   teachers** — condition the privileged teacher on a correct peer vs. wrong peers
   and use the *difference* as the signal, cancelling privileged-context artifacts.
   That trick is orthogonal and portable: it can debias $q$ before any downstream
   use, including a tilt.

4. **The field is converging on "edit the target, not the loss" — but heuristically.**
   SDPO, SGSD, Uni-OPD, OGLS-SD, AD-OPSD, DASD, AntiSD all inject the verifier or an
   entropy signal into the *distillation target* (conditioning, sign flips, margin
   shifts, logit steering, prior mixing). None derives the edited target from an
   objective; each is a patch with its own gate hyperparameters. The advantage-tilted
   posterior is the closed-form version of what this family approximates.

5. **Forward projection onto student-visited states is accumulating theory.**
   Distributional DAgger (2606.05152) proves monotonic-improvement/regret guarantees
   for forward CE on student states that reverse KL and JSD provably lack; LPO
   (2605.06139) finds forward-KL projection onto a tilted target preserves diversity
   where reverse-KL/GRPO collapse it; Decoupling-KL (2605.16826) finds long-sequence
   distillation needs substantial forward-KL weight to avoid entropy collapse.
   Caveat from OPD+ (2606.01039): naive stop-gradient *surrogates* of forward KL are
   biased (the correction $w_f(u) = -f(u)+u f'(u)$ is constant only for reverse KL).
   A constant-target CE — as in the tilted proposal — sidesteps that specific bias
   because the target carries no $\theta$-dependence.

6. **The reward-tilted-teacher posterior is now "in the air," but the token-level
   instantiation is unclaimed.** PSD (2605.05040) states $\pi^*\!\propto\pi_T e^{r/\beta}$
   verbatim (sequence-level, preference reward, DPO-ified projection); RTDMD/DMDR do
   the same math for diffusion models; AlignDistil distills $\pi_{\text{ref}}e^{r/\beta}$
   token-wise but with a DPO implicit reward and a synthesized (not external) teacher.
   As of 2026-07-22, no paper tilts an external/privileged teacher's *next-token*
   distribution by a *group-whitened verifier advantage* and distills forward-KL.
   This will not stay unclaimed long — PSD and RTDMD are both May 2026.

---

## Family A: Additive Term, Annealed or Gated

The family of dGRPO/KDRL descendants. All entries add $\lambda\cdot L_{\text{distill}}$ to a
GRPO-style loss; all schedule $\lambda$ toward zero or gate it.

| Paper | ID | Loss core | Balance mechanism |
|---|---|---|---|
| KDRL (precursor, Jun 2025) | 2506.02208 | $J_{\text{GRPO}} - \beta D^{k2}_{\mathrm{KL}}(\pi_\theta\|\pi_T)$, $D^{k2}=\tfrac12 R_{i,t}^2$, $R=\log\frac{\pi_T}{\pi_\theta}$ | β annealed 5e-3→1e-3; KD masked off on already-correct rollouts |
| TGPO (May 2026) | 2605.13230 | $J_{\text{GRPO}} + w\cdot$ CE on *teacher-proposed* next tokens under student prefixes | $w$ linear decay to 0 by step 200; imitation early, pure RL late |
| SDPG (Jun 2026) | 2606.04036 | GRPO + full-vocab reverse KL to privileged self-teacher, applied only on $A>0$ rollouts | trapezoid $\beta(k)$ warmup→decay→0; keeps entropy where RLSD collapses |
| SEED (Jul 2026) | 2607.14777 | GRPO + gated CE toward self-teacher conditioned on self-extracted "hindsight skills"; gate $g=\sigma(\beta_{\text{opd}}\Delta)$ | sigmoid confidence gate per token |
| SSOPD (May 2026) | 2605.17497 | GRPO + distill shortest-correct-conditioned self-teacher into prefixes of longest-wrong rollout | frontier weight $\lambda_x = \lambda_0\, 4\hat p(1-\hat p)$ — vanishes on all-correct/all-wrong groups |
| EOPD (Mar 2026) | 2603.07079 | clipped reverse-KL OPD $+\ \mathbb 1[H^{te}_t>\tau]\cdot$ forward KL (top-16 truncated) | hard teacher-entropy gate τ=0.8; adds mode-covering FKL only where teacher is uncertain |

**Design insights.** (i) TGPO's "teacher proposes actions on student states" (instead
of scoring student actions) stays informative under large teacher–student divergence,
where reverse-KL collapses — relevant when the prior is far from the student.
(ii) SSOPD's frontier weight $4\hat p(1-\hat p)$ is the same quantity that controls
GRPO's advantage variance — an elegant *automatic* localization of distillation to
prompts where the verifier signal is informative. Portable to any hybrid, including
as a per-prompt modulation of β in the tilted design. (iii) SDPG's positive-advantage
gate is a discrete, sample-level shadow of the tilt: it uses the verifier to decide
*whether* to distill; the tilt uses it to decide *toward what*.

## Family B: Distillation Folded Into the Reward Channel

| Paper | ID | Loss core |
|---|---|---|
| CoDistill-GRPO (May 2026) | 2605.08873 | $\tilde r_i = r(q,o_i) + \alpha\cdot\frac1N\sum_t\log\frac{\pi_\theta}{\pi_\phi}$, then standard GRPO whitening on $\tilde r$; teacher simultaneously updates via importance-reweighted GRPO on student rollouts |
| Direct-OPD (Jul 2026) | 2607.05394 | weak-to-strong: dense reward $r_t(v)=\log\pi_T(v|s_t)-\log\pi_{T_{\text{ref}}}(v|s_t)$ — the *RL-induced shift* of a small teacher — distilled over the student's top-k support; adaptive KL coefficient by batch-mean drift sign |
| LUFFY (Apr 2025, NeurIPS) | 2504.14945 | mixed-policy GRPO: off-policy teacher traces inserted into the rollout *group* (receiving max reward), sharing group-relative advantage computation; policy shaping via regularized importance sampling on off-policy tokens |

**Design insights.** (i) CoDistill's joint whitening is the cleanest scale fix
available to additive-style combination: standardize *after* mixing and the units
problem disappears (at the cost of collapsing the dense vocab-wide signal to a scalar
per sequence — objection (b) returns). (ii) Direct-OPD's reward is exactly ExOPD's
implicit-reward reading run in the weak-to-strong direction: the teacher's *RL delta*,
not the teacher itself, carries the verifier information — supporting Sang et al.'s
condition C1 (a raw non-RL teacher underperforms direct GRPO). (iii) LUFFY shows the
teacher can enter through the *baseline* (group composition) with no KL term at all.

## Family C: Teacher Ratios Reshape the Advantage (RLSD Descendants)

All keep the verifier's sign and use teacher ratios only as magnitude — objection (c)
persists throughout this family.

| Paper | ID | Advantage modulation |
|---|---|---|
| CEPO (May 2026) | 2605.19436 | $\hat A_t = A\,[(1-\lambda)+\lambda\,\mathrm{clip}(e^{\mathrm{sign}(A)\Delta^{CE}_t})]$, $\Delta^{CE}_t=\mathrm{sg}[\log P_{T^+}(y_t)-\log P_{T^-}(y_t)]$ with correct- vs *wrong*-answer-conditioned teachers; λ: 0.5→0 in 25 steps |
| RLCSD (Jun 2026) | 2606.11709 | contrast correct-peer-conditioned vs mean of K wrong-peer-conditioned self-teachers; modulates ~20–30% of tokens, sign-preserving clamp, two token-paths normalized independently |
| AMR-SD (May 2026) | 2605.18529 | $\hat A_{i,t}=A_i(1+\Delta_{i,t})$, thresholded contextual-information-gain applied asymmetrically ($\lambda=0.2$ pos / $\gamma=0.1$ neg), annealed to zero |
| PBSD (Jun 2026) | 2606.09348 | Bayesian support score $s_{i,t}=\log\frac{p(a_{i,t}|\cdot,y^*)}{p(a_{i,t}|\cdot)}$ (= per-turn evidence for the verified answer); $w=1+\mathrm{sign}(A)\,\mathrm{clip}(\tanh(s/\delta))$; explicitly *never* used as a target |
| GEAR (May 2026) | 2605.11853 | student–teacher divergence decides both granularity (token vs adaptive segment) and modulation strength of trajectory advantages (PDF-only; abstract-level) |
| Distilled RL (Jul 2026) | 2607.17247 | GRPO surrogate × token weight $w_{i,t}=\widetilde\rho_{i,t}$ if $A_i>0$, else $1$; $\rho_{i,t}=\pi_T(y_t)/\pi_{\mathrm{old}}(y_t)$ clipped to $[1/3,3]$, then divided by its per-sequence **geometric mean** so $\prod_t\widetilde\rho_{i,t}=1$ ("redistribute credit within the response, never amplify it"); no KL/CE term anywhere; needs only sampled-token teacher log-probs; gains in both pass@1 and pass@k (Qwen3-8B-GRPO → Qwen3-4B and → R1-Distill-1.5B, DAPO-17K) |

**Design insights.** (i) The **contrastive-teacher** construction (CEPO, RLCSD; also
Purified OPSD and RLCSD's ancestor CREDIT) is the family's real contribution:
$\log P_{T^+} - \log P_{T^-}$ cancels privileged-context style artifacts by
construction, isolating outcome-relevant credit. This is portable to the tilted
design as a debiasing step on $q$ (or as the tilt direction in a dense-tilt variant).
(ii) PBSD's identity — trajectory quality = posterior-to-prior ratio of the verified
answer, tractable as a student/privileged-teacher likelihood ratio — is a clean
Bayesian account of *where* credit lives, complementary to the banked-credit analysis
in the proposal note. (iii) The family's uniform sign-preservation rule ("teacher may
never flip the verifier") is a design axiom worth challenging: rule (5) of the tilted
proposal prices the flip instead of forbidding it. Distilled RL enforces the axiom in
its strongest form — the teacher is switched *off* entirely on $A_i\le 0$, not merely
clamped. (iv) **Distilled RL is the closest published cousin of the PPO–OPD bridge
note, on the opposite side of its Step 7 fork.** Its centered log-weight
$\log\widetilde\rho_{i,t}=\log\frac{\pi_T}{\pi_{\mathrm{old}}}(y_t)-\text{seq. mean}$
is (up to $\beta$ and the ref-vs-old baseline) exactly the centered increment of the
teacher potential $\widehat V_T$ — the *same* teacher quantity, spent in the
**credit channel** (multiplicative on advantage: permanent, uncharacterized bias, no
weaning schedule, only the $[1/3,3]$ clip bounds it; geometric normalization fixes the
weights' product, not their mean, so per-token credit is genuinely distorted, and
asymmetrically since positives only) rather than the bridge's **potential/init
channel** (transient by the shaping theorem) or **target shape** (deliberate,
closed-form fixed point $\propto\pi_T e^{R/\beta}$). Operational corollaries: on
uniform groups ($A_i=0$) its update vanishes exactly — no OPD floor, dead prompts stay
dead; gradient support is the sampled coordinate only; but with no distillation pull
at all it exerts zero mode-collapse pressure toward the teacher, consistent with its
pass@k gains.

## Family D: Verifier Reshapes the Distillation Target

The closest family to the proposal — all inject outcome information into *what* is
distilled rather than *how much*.

| Paper | ID | Target edit |
|---|---|---|
| SDPO (Jan 2026, ETH) | 2601.20802 | $\sum_t \mathrm{KL}(\pi_\theta(\cdot|x,y_{<t})\,\|\,\mathrm{sg}[\pi_\theta(\cdot|x,f,y_{<t})])$ — verifier feedback $f$ enters through the *conditioning* of a self-teacher; implicit advantage $=\log\frac{\pi_\theta(y_t|x,f,y_{<t})}{\pi_\theta(y_t|x,y_{<t})}$ |
| SGSD (May 2026) | 2605.28791 | bank of K skill-conditioned self-teachers; binary outcome multiplies each teacher's *sign* (reinforce/reverse/drop) via polarity $\rho_k=\mathrm{sgn}(r)\cdot\mathrm{sgn}(\tilde a_k)$ plus a saturating robust gate |
| Uni-OPD (May 2026) | 2605.03677 | margin calibration: add uniform shift $\lambda(q)=\delta-m(q)$ to distillation returns of correct trajectories so teacher scores become order-consistent with the verifier — "smallest additive correction" |
| RG-OPD (Jul 2026) | 2607.04037 | distill only when advantage sign and teacher–student likelihood gap agree directionally: $g_i=\mathbb 1[(A>0\wedge L_T>L_S+\delta)\vee(A\le0\wedge L_T<L_S-\delta)]$ |
| SG-OPD (Jun 2026) | 2606.09304 | token routing by consensus: verifier+teacher agree → extrapolated update (λ=1.8, ExOPD-style); conflict → interpolated (β=1); +7.50 pass@32 over OPD |
| OGLS-SD (May 2026) | 2605.12400 | outcome-guided **logit steering** of the privileged self-teacher: additive logit edit along a contrastive success-vs-fail direction before distilling |
| AOPD (May 2026) | 2605.06387 | diagnoses advantage-weighted CE breaking on negative-advantage sampled tokens (unbounded gradients); switches discretely: PG where $A>0$, teacher alignment where $A\le 0$ |

**Design insights.** (i) This family is the empirical forerunner of "argue on the page":
every entry edits the target distribution or its ordering with verifier information.
But each edit is a *heuristic with a gate* — thresholds δ, polarity rules, routing
constants. The tilted posterior is the same move derived once from
$J=\mathbb E[R]-\beta\,\mathrm{KL}$, with β as the single parameter. (ii) OGLS-SD is
mechanically the nearest neighbor: an additive logit edit *is* a multiplicative tilt;
the delta is the direction (learned contrastive vector vs. $\hat A/\beta$ on the
sampled coordinate) and the absence of an objective it optimizes. (iii) AOPD is
direct evidence for the tilt's handling of the negative-advantage regime: the place
where naive advantage-weighted CE explodes is exactly where the renormalized tilt
stays bounded (mass flows to the teacher's alternatives; support floor
$\tilde q_v \ge (1-\kappa)q_v$). (iv) SG-OPD's pass@32 gain shows consensus-routing
preserves exploration — but rule (5) achieves the same arbitration continuously,
without the two routing constants.

## Family E: Exploration/Entropy Preservation (Fork-Suppression Responses)

Context: the Princeton paper (2607.05184) showed privileged self-teachers suppress
forking/self-correction tokens in thinking models. Two papers already cite it
(AD-OPSD 2607.10805; Purified OPSD 2607.02234). The fixes cluster into four
mechanisms:

- **Entropy-routed direction flips.** DASD (2605.22263): low-entropy tokens pulled
  toward the teacher, high-entropy tokens *pushed away* (sign-flipped KL). AntiSD
  (2605.11609): identifies the OPSD per-token reward as
  $\mathrm{PMI}(y_t; c\,|\,x,y_{<t})$ with the privileged context $c$ — negative on
  deliberation tokens — and *inverts* it through a bounded transform
  $\varphi(u)=\tfrac12(\mathrm{softplus}(u)-\log 2)$, added to the GRPO advantage
  with a Schmitt-trigger entropy gate (on above $H_{\text{warm}}$, off below
  $0.93\,H_{\text{warm}}$). Reaches GRPO accuracy in 2–10× fewer steps.
- **Prior anchoring at fragile positions.** AD-OPSD (2607.10805): inside a top-20%
  student-entropy "sandbox," blend the target with the *frozen base* prior,
  $\pi^*_i = U_i\,\pi_{\text{base}} + (1-U_i)\,\pi_T$, with unreliability
  $U_i=\sigma(P_\theta(y_i)(\log P_\theta-\log P_T)/\tau)$ — the first published fix
  citing the Princeton finding. Structurally, this is the geometric-mixture-prior
  mitigation of the proposal note, in arithmetic-mixture form and entropy-gated.
- **Teacher purification.** Purified OPSD (2607.02234): build a reference-only
  teacher $\pi(y_t|\text{reference},y_{<t})$ (no question) to isolate the
  non-transferable reference-copying component; distill only the residual via a PMI
  transform. Same debiasing spirit as CREDIT/CEPO, applied before any loss.
- **Entropy restoration.** TS-OPSD (2606.00755): distill from a temperature-scaled
  copy of the *collapsed policy itself* to reheat entropy post-collapse; CurioSFT
  (2602.02244): entropy-guided per-token temperature on a self-teacher during SFT,
  paying off (+5.0 avg) in the subsequent RL stage.

**Diagnosis papers** (no fix): Kim et al. (2603.24472) — the SD trade-off is governed
by task coverage vs conditioning richness; Nicolicioiu et al. (2606.26091) — proves
the privileged teacher tilts the base by pointwise *conditional mutual information*
with the demonstration, amplifying existing gaps (unlike optimal RL, which preserves
ratios among correct outputs) — the sharpest formal account of why OPSD collapses
diversity.

**Design insights.** (i) All fixes detect fragile positions by *entropy* and
special-case them; none uses verifier evidence to decide *which* high-entropy forks
deserve to live. Rule (5) of the tilted proposal is exactly that missing arbitration.
(ii) The PMI identifications (AntiSD, Nicolicioiu) put math under the "biased prior"
diagnosis: the OPSD reward literally equals mutual information with the privileged
context, which is negative on hedging tokens. A principled combination should
therefore *separate* the prior (teacher) from the evidence (verifier) rather than
inherit the teacher's PMI as an implicit reward — which is what the posterior
factorization $\pi_T\cdot e^{R/\beta}$ does.

## Family F: The Tilted-Target Line (Novelty-Relevant)

| Paper | ID | Relation to the proposal |
|---|---|---|
| PSD (May 2026) | 2605.05040 | **Closest theory match.** Prop. 1: $\pi^*=\pi_{\text{teach}}e^{r/\beta}/Z$, "reward-tilted teacher distribution," optimizer of $\mathbb E[r]-\beta\mathrm{KL}(\pi\|\pi_{\text{teach}})$. Deltas: self-teacher, *sequence-level*, latent preference reward, and a DPO-ified Bradley–Terry projection that sidesteps $Z$ instead of renormalizing |
| AlignDistil (Mar 2025, ACL) | 2503.02832 | token-level RLHF with DPO reward ≡ distilling a synthesized teacher with extrapolated DPO/ref logits — the foundational token-level $\pi_{\text{ref}}e^{r/\beta}$ distillation; no verifier, no external teacher |
| DMDR / RTDMD (Nov 2025 / May 2026) | 2511.13649 / 2605.26108 | same math for diffusion: distill toward $p_\psi(x)e^{\beta r(x)}/Z$ (RTDMD even uses GRPO for the reward term); sample-level, DMD projection. Finding: the tilted-teacher term *mitigates reward hacking* during RL |
| LPO (May 2026) | 2605.06139 | group RLVR = projection onto $\mathrm{softmax}(R/\tau+\text{logits})$ over the K sampled responses; tilts the *current policy*, no teacher. Finding: forward-KL projection preserves diversity where reverse/GRPO collapse |
| Constrained-MDP distillation (Sep 2025) | 2509.22921 | $\max \mathbb E[R]$ s.t. $\mathrm{KL}(\pi\|\pi_T)\le\varepsilon$ — same objective in constrained form, solved by state-augmented constrained RL rather than closed-form posterior |
| Sparse-to-Dense (May 2026) | 2605.12483 | states OPD "substitutes the teacher's policy for the reward-tilted target"; conditions C1 (teacher must be reward-shaped) & C2 (teacher near student in KL); solves *sequentially* (RL the teacher, then distill) what the tilt fuses into one target |
| BOND (2024) | 2407.14622 | distills the Best-of-N policy (monotone reward transform of $\pi_{\text{ref}}$) via Jeffreys divergence — ancestral precedent for "distill a reward-improved transform of a policy" |
| RWOPD (May 2026) | 2605.13501 | closest *practical* setup (external teacher + verifier + forward KL on student rollouts) but the reward is a scalar weight **on the loss**, not a tilt **of the target** — changes gradient magnitude, never redirects mass |

**Novelty status (as of 2026-07-22).** No found paper performs: one-coordinate
exponential tilt of an external/privileged teacher's next-token distribution by a
group-whitened verifier advantage, renormalized (Z tractable exactly because only one
coordinate is tilted), trained by forward KL/CE on student-sampled prefixes. The
*framing* (teacher=prior, verifier=likelihood, OPD ≈ KL-regularized RL) is now
explicit in PSD, ExOPD, Sang, and Zimmer — any writeup must position against those
four plus AlignDistil and RTDMD. Recommended before drafting: a citation-graph pass
over PSD (2605.05040) and RTDMD (2605.26108), the two most likely to spawn the LLM
token-level variant.

## Theory Corner: Results That Bear Directly on the Proposal

- **OPD+ (2606.01039).** Stop-gradient OPD surrogates drop the term
  $u f'(u)\nabla\log p_\theta$; the corrected weight is $w_f(u)=-f(u)+uf'(u)$.
  Constant for reverse KL (why reverse-KL OPD "just works"), material for forward
  KL/JSD. *Implication:* the tilted loss avoids this bias class because its target is
  a constant (no $\theta$ inside $\tilde q$) — worth one line in the note's Step 3.
- **vOPD (2605.07865).** Closed-form per-token value
  $V(c_t) = -\mathrm{KL}(\pi_\theta(\cdot|c_t)\,\|\,\pi_T(\cdot|c_t))$ as an analytic
  control-variate baseline (+6.2 on MATH500). *Implication:* for the value-increment
  upgrade in Step 6, the *distillation component* of the value has a closed form;
  only the verifier component needs a learned head.
- **Distributional DAgger (2606.05152).** Forward CE on student-visited states admits
  monotonic policy improvement and regret guarantees; reverse KL and JSD provably do
  not, even with a strictly better expert. *Implication:* theoretical backing for the
  forward-projection choice (Estimator B) beyond the boundedness argument.
- **Decoupling KL (2605.16826).** 2×2 unification (prefix source × KL direction)
  recovering SFT/DAgger/offline-RL/OPD; empirically, long-sequence distillation needs
  substantial forward-KL weight to avoid entropy collapse.
- **Rethinking OPD (2604.13016).** OPD success tracks thinking-pattern consistency
  (top-k overlap), not teacher benchmark strength; distilling toward a *weaker but
  mismatched* checkpoint reproduces the failure. *Implication:* supports the
  geometric-mixture prior $\pi_T^\alpha\pi_{\text{old}}^{1-\alpha}$ when the teacher
  is far from the student.
- **Nicolicioiu et al. (2606.26091).** Privileged self-teacher = PCMI tilt of the
  base; amplifies probability gaps rather than preserving ratios among correct
  outputs. *Implication:* formalizes why the *prior alone* must not carry the
  evidence — separation of prior and likelihood is essential.

## Positioning Checklist for `advantage_tilted_opd.md`

Must-cite / must-differentiate: **PSD** (same posterior, sequence-level, BT
projection), **AlignDistil** (token-level closed-form distillation, DPO reward),
**RTDMD/DMDR** (same math, diffusion), **ExOPD + Sang 2605.12483 + Zimmer 2509.22921**
(same objective, different solution route), **OGLS-SD** (logit steering ≈ tilt,
heuristic direction), **RWOPD** (weight-on-loss vs tilt-of-target contrast),
**AOPD** (negative-advantage failure mode the tilt fixes continuously),
**LPO** (forward-projection-preserves-diversity evidence), **AntiSD/Nicolicioiu**
(PMI theory motivating prior/evidence separation), **AD-OPSD** (published
arithmetic-mixture cousin of the mixture-prior mitigation).

Portable upgrades surfaced by this sweep, in priority order:
1. **Contrastive debiasing of $q$** before tilting (CEPO/RLCSD/Purified OPSD trick)
   — composes with the tilt, targets privileged-context artifacts the tilt ignores.
2. **Analytic KL value + learned verifier head** for value-increment tilts (vOPD).
3. **Per-prompt β modulation** by group success variance, à la SSOPD's
   $4\hat p(1-\hat p)$ frontier weight — concentrates the verifier's influence where
   the group signal is informative, decays to pure OPD elsewhere (consistent with the
   $\hat A=0$ limit).

## Verification Caveats

Loss formulas above were extracted from arXiv HTML by search subagents; spot-check
before citing in a paper. Known-lossy extractions: GEAR (2605.11853) and CoPD
(2604.27083) are PDF-only (abstract-level descriptions); LUFFY's exact shaping
function and AOPD/ROAD-VLA details came from lossy PDF text; DASD's exact equation
was not exposed in the fetched version. Citation counts for the Princeton paper
(2 citing works) are from Semantic Scholar on 2026-07-22 and will be stale quickly.
Useful trackers: `github.com/chrisliu298/awesome-on-policy-distillation`,
`github.com/nick7nlp/Awesome-LLM-On-Policy-Distillation`, surveys 2604.00626
(§4.3 "RL-Augmented Objectives") and 2606.22793.
