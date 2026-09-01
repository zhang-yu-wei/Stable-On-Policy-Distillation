# Teacher Intervention: Trigger, Handback, and Span Granularity

Setting: multi-turn agent; the student rolls out; the teacher takes over
mid-trajectory when the student is stuck and hands back after repairing the
state. Two questions: when to switch control, and at what granularity spans
should be defined.

## 1. When to trigger, when to hand back

Both decisions are the same measurement applied to different models:
**success-to-go at a state**, estimated by forking $k$ probe continuations
from that state and counting successes.

**Trigger** — take over at $s_h$ when both hold:

1. *Student stuck:* student success-to-go at $s_h$ is ≈ 0. Cheap online
   signals raise suspicion (teacher potential not rising for $n$ turns,
   repeated near-identical actions, sustained entropy spike); $k$ failed
   student probes from $s_h$ confirm it.
2. *Teacher can recover:* teacher perplexity of the prefix is acceptable
   and one teacher probe from $s_h$ succeeds. If not, test earlier states
   until one passes — the latest state the teacher can recover from is the
   takeover point, and the turns just after it are where the fatal error
   lies.

**Handback** — return control at the first teacher-reached state $s'$ where
the student is competent again: student probes from $s'$ succeed at rate
$\ge\tau$. Handing back earlier just re-triggers the intervention.

**Without a forkable environment** (no probes possible): trigger on the
online signals alone, and hand back once they have stayed clear for a full
turn (potential rising, no loops).

## 2. Granularity of spans

**Rule: place span boundaries where the measured value changes.** A span is
the coarsest unit within which success-to-go (equivalently, the teacher
potential) is roughly constant; a boundary belongs wherever the value
drops. A boundary across which the value does not change carries no credit
information, so nothing is gained by cutting there.

Consequences:

- **The turn is the base unit** in agent setups, not by convention but
  because states are only well-defined at turn boundaries — observations
  arrive between turns, and probes can only fork from a state the
  environment can be reset to. Trigger, handback, and credit all live at
  turn resolution or coarser.
- **Coarser when turns are redundant:** merge adjacent turns whose
  environment state is essentially unchanged (same observation, no value
  movement) into one span — SERL's anchor grouping. Long trajectories then
  have few spans, each ending at a real state change.
- **Finer than a turn only inside the loss, never for control:** within a
  turn, token-level choices (action-token masks, entropy-gated tokens)
  decide *which tokens* of a span receive logit supervision, but the
  intervention and credit unit remains the turn — you cannot fork the
  environment mid-turn, so sub-turn "spans" have no measurable value of
  their own.
- **Stop refining at the probe noise floor:** with $k$ probes, success-rate
  differences below $\sqrt{p(1-p)/k}$ are indistinguishable, so bisecting
  the fatal-error span further than that resolution (usually: down to one
  or two turns) only measures noise.
