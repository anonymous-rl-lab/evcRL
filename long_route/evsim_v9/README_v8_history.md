# EV eco-driving testbed — 20 km arterial, **v8 (MK)**

```
python3 check_env.py     # cores, RAM, ETA measured on this machine
python3 verify.py        # 7 pre-flight checks; run_all.sh refuses to start if any fails
bash run_all.sh          # 12 processes, plain configuration, 12 seeds, ~1.5-2.5 h
python3 analyze.py       # validation-selected results on held-out conditions
tar czf results_v8.tgz out/*.json logs/ out/ckpt_*_5000k.pt
```

References, objective `R_L = -E_batt/100kJ - 0.08 t`:

| full DP | info-matched DP | attentive | normal | distracted |
|---|---|---|---|---|
| −168.19 | −169.79 | **−176.20** | −178.04 | −179.38 |

---

## The one change in this version, and why

**The observation was not Markov with respect to the episode deadline.**

An episode ends at `t_end`. For a cold start that is `1.25 * T_budget = 1312.5 s`; for an
exploring start (85 % of training episodes) it is `t0 + 1.25 (L - x0)/pace`, i.e. it depends on
where and when that episode began. The observation's clock channel carried
`(T_budget - t)/T_budget` with a **fixed** `T_budget`, so the deadline was invisible to the agent.

Measured directly:

```
episode A (cold start,   x0=0,    t0=0  ) : t_end = 1312.5 s
episode B (exploring,    x0=8 km, t0=550) : t_end = 1337.5 s
both at (x=10 km, v=18 m/s, t=600 s)      : observations identical, max|diff| = 0.00e+00
```

Across the training distribution `t_end` ranges **927–1644 s** while evaluation always uses
1312.5 s; **57 % of training episodes have a deadline more than 60 s away from evaluation's.**
Because an unfinished trip is charged up to **−300** units of cost-to-go, the same observation
carried wildly different true values.

A critic is a *function*: one input, one output. Given contradictory labels it regresses to their
average, which is wrong for every episode. This is not estimation noise — the information is
absent from the input, so more data cannot fix it. That is exactly what the probes showed:

| observed | explanation |
|---|---|
| `Q(s0)` frozen at −264 for 4 M steps while the true return moved −178 to −191 | it had converged — to the average |
| twin-critic disagreement decaying 0.83 → 0.36 | both critics agreeing on the same wrong number |
| bias −80, never shrinking | irreducible, not statistical |

−264 is, to within a unit, the return of a trip that quits at 5.5 km. The actor maximises the
critic, so it was being pulled toward quitting. The probes caught the direction: when the policy
was at its best (−178) the gap to `Q(s0)` was largest (−86); as the policy degraded to −198 the
gap shrank to −61. **The policy was moving toward the critic, not the critic toward the policy.**

**The fix is one line** — report the episode's real remaining budget:

```python
(self.t_end - self.t) / self.t_budget      # was (self.t_budget - self.t) / self.t_budget
```

`verify.py` check 7 pins it so it cannot regress. This is the classic RL time-limit trap: with a
time-limited episode you must either put the remaining time in the state or bootstrap through the
truncation. The code did neither. *(There is an ICML 2018 paper on exactly this,
"Time Limits in Reinforcement Learning" — **author list and venue not yet verified**, check before
citing.)*

## What the fix did and did not do

Same seeds, 20 km, 5 M steps, all numbers on the 12-condition grid:

| | seed 0 before | **seed 0 after** | seed 1 before | **seed 1 after** |
|---|---|---|---|---|
| checkpoints where arrival < 12/12 | 11/20 | **2/11** | **19/20** | **3/11** |
| `Q(s0)` bias | −73 … −97 | **−44 … −50** | −73 … −97 | **−44 … −54** |
| validation-selected score | −172.13 | −177.53 | −177.48 | −179.77 |
| last-checkpoint score | −172.13 | −180.94 | **−190.77** | **−183.11** |

**Fixed:** the arrival collapse. Seed 1 used to lose the route in 19 of 20 checkpoints; now 3 of
11. The critic bias halved. The last checkpoint no longer falls off a cliff.

**Not fixed:** the peak level is *lower* (−177.5 against the old best of −172.1), the return still
oscillates in a ±4-unit band, and the residual −45 bias remains. The likely remaining cause is
off-policy staleness — the buffer holds the whole 5 M-step run and n-step returns carry no
importance correction — but that is **untested**; the experiment was stopped before it ran.

## Everything that was tried and failed

Recorded so the next person does not repeat it. Validation-selected means where available.

| attempt | outcome |
|---|---|
| progress-potential shaping, as originally shipped | −177.51 (n=8); the implementation was missing `Phi(absorbing)=0`, so the total shaping was policy-dependent and up to 184 units |
| the same shaping, corrected (`Phi(absorbing)=0`, `c = c* = 0.00844`) | −177.52 (n=2). Still worse than adding nothing |
| priming the warmup with a scripted 16–20 m/s cruise | fixes the early bootstrap hole (both seeds reach −178 at 750 k) but degrades to −187/−194 by 5 M. Lower spread, worse level |
| cost-to-go 15 → 9 per km | arrival collapses to 0.00, buffer 99 % truncated |
| γ = 1 → 0.995 | full braking everywhere, cruise 8 m/s, **R −391** |
| relaxing the training cut-off to 12 m/s | escapes the trap at 150 k, then settles at 14 m/s and fails the strict criterion — one trap traded for another |
| **removing the 1050 s deadline entirely** (bootstrap through truncation, no cost-to-go, no clock) | **200–340 units worse at every checkpoint and not converging.** The policy runs to the speed clamp (25–27 m/s), the action goes dead (96 % at 250 k) and the return scale collapses |

**Every one of these fails for the same reason: `γ = 1` and the cost-to-go cliff are not bugs, they
are what makes this task well-posed.** Anything that makes quitting or over-speeding cheaper
breaks it. The plain configuration — no shaping, no priming, cliff intact — remains the best.

## A second, independent problem, diagnosed but not fixed

The episode cut-off `t_end = t + 1.25 (L-x)/pace` is **scale-free**: arriving requires an average
of **15.24 m/s from any start**, so exploring starts supply no easy completions. The first ~100 k
steps decide by exploration noise alone which side of that a seed lands on; a seed below it
completes nothing and its buffer holds 3 % completed trips against a good seed's 60 %. Seed 1
needed 1.35 M steps to climb out. `verify.py` check 5 demonstrates the threshold (14 m/s does not
arrive, 18 m/s does). **No fix for this is included** — the two that were tried (relaxing the
cut-off, priming) both traded it for something worse.

## Reporting protocol

Training peaks and then degrades; the peak sits anywhere from 250 k to 5 M steps. `analyze.py`
splits the 12 conditions into 6 **validation** (offsets 0 and 45 s) and 6 **reporting** (22.5 and
67.5 s), balanced across pack states, selects the checkpoint on the validation half and reports
the held-out half. On every run on file this recovers 5.4 units and nearly doubles the completion
rate (11/14 against 6/14) relative to last-checkpoint reporting. **It is not optional on this task
and it must be declared in the paper.**

Every result is reported as `last / last-3`. The last-3 mean is the number that goes in the paper:
an earlier headline was built on a single peak checkpoint (−174.57) whose three-point mean was
−178.32, below the normal driver.

## What TD3 achieves

With validation-selected checkpoints, across all 14 runs that reached 5 M steps: best **−172.13**
(12/12 completed, 4.1 units better than the attentive driver), 5/14 beat the attentive driver,
9/14 beat the normal driver, mean −177.32. **Achievable but not reliable** is the honest summary.

## Open questions for whoever picks this up

1. **Off-policy staleness.** The buffer never wraps and n-step returns are uncorrected. Does a
   FIFO buffer (500 k) or an importance-corrected target (Retrace) close the residual −45 bias?
   Launched, never completed.
2. **Why is the peak level lower after the Markov fix** (−177.5 vs −172.1)? The fix is provably
   correct, so this needs explaining rather than accepting.
3. **The 15.24 m/s bootstrap hole.** Both attempted fixes failed. A curriculum on the cut-off
   (relaxed early, tightened later) was designed but never run.
4. **Training instability.** Nothing tried removed the peak-then-degrade pattern; only the
   validation-selection protocol works around it.
