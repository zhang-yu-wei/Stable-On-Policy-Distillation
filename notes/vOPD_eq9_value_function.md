# vOPD (KL for a KL) — Understanding Eq. 9: The OPD Value Function

## Setup

In OPD, the per-token reward at context `c_t` for sampled token `y_t` is:

```
r_t(c_t, y_t) = log π_T(y_t | c_t) - log π_θ(y_t | c_t)
```

## Eq. 9: Value function has a closed form

The value function is the expected reward under the student's own distribution:

```
V^{π_θ}(c_t) = E_{y_t ~ π_θ(·|c_t)} [r_t(c_t, y_t)]
             = Σ_v π_θ(v|c_t) · (log π_T(v|c_t) - log π_θ(v|c_t))
             = - Σ_v π_θ(v|c_t) · log(π_θ(v|c_t) / π_T(v|c_t))
             = - D_KL(π_θ(·|c_t) || π_T(·|c_t))
```

**The OPD value function is exactly the negative per-step reverse KL.**

## Why this matters

1. **Free to compute**: The forward pass already produces both `π_θ` and `π_T` at each prefix. Computing `D_KL(π_θ || π_T)` requires no additional critic network, no extra rollouts — just a sum over the vocabulary (or top-k approximation).

2. **Standard RL variance reduction**: In policy-gradient RL, subtracting a baseline `b(c_t)` from the reward reduces variance without introducing bias (because `E[b(c_t) · ∇ log π_θ(y_t|c_t)] = 0` for any baseline independent of `y_t`). The canonical choice is the value function.

3. **vOPD advantage**: The resulting advantage is:
```
a_t(c_t, y_t) = r_t(c_t, y_t) - V^{π_θ}(c_t)
              = r_t(c_t, y_t) + D_KL(π_θ(·|c_t) || π_T(·|c_t))
```

## Intuition for variance reduction

- When `r_t` is very negative (student sampled a token teacher dislikes), the raw gradient is large and destabilizing.
- But `V = -D_KL` is also negative when student and teacher diverge, so the advantage `r_t - V = r_t + D_KL` stays bounded.
- The variance reduction is largest precisely at high-mismatch tokens (where `D_KL` is large), which are the main source of gradient instability.

## Top-k approximation (Eq. 14-15)

The baseline can be approximated using only the student's top-k tokens:
```
b̂_t(c_t) = - D_KL(π̄_θ(·|c_t) || π̄_T(·|c_t))
```
where `π̄` is renormalized over student top-k support `S_t`. This preserves unbiasedness because the baseline does not depend on the sampled token `y_t`, regardless of how it is computed. The key distinction from OPD_top-k: top-k is used only in the *baseline* (detached), not in the *loss* itself, so no gradient bias is introduced.
