# DESIGN.md — Quilter PCB Placer

This was written incrementally as decisions were made, not reconstructed afterward, so the level of
detail tracks how much a given decision actually mattered rather than how it looks in hindsight.

## 1. Architecture overview

```
placer/
  board.py             Component, Pin, Net, KeepoutZone, Board -- the data model
  generator.py         generate_board(config) -> Board
  cost.py              C(p): RSMT (exact <=4 pins, MST heuristic >4), overlap, congestion
  incremental_cost.py  IncrementalCost: delta-cost tracking for simulated annealing
  feasibility.py       check_feasibility(board) -> FeasibilityReport
  legalize.py          clamp/keepout-push/overlap-resolution, shared by baseline + learned placer
  baseline.py          force-directed init + simulated annealing refinement
  learned/             GNN warm-start + budgeted refinement
scripts/
  benchmark.py         paired baseline-vs-learned comparison, |V|=200 and 1000
  spike_day1_validation.py   an early validation experiment, described in section 6
```

`place(board)` and `generate_board(config)`, the two required entry points, are re-exported from
`placer/__init__.py`.

## 2. The cost function: decisions under uncertainty

The spec's cost function, `C(p) = sum_e w_e * RSMT(e) + 100 * sum_(u,v) max(0, overlap(u,v)) +
C_cong(p)`, leaves several things genuinely underspecified. Each decision below was made
deliberately, and each is a place a different reasonable choice was possible.

### 2.1 RSMT: exact for <=4 pins, MST heuristic above that

Computing an exact rectilinear Steiner minimum tree is NP-hard in general. The spec asks for exact
computation at <=4 pins and permits a heuristic above that, provided the approximation ratio is
documented.

**Exact solver (<=4 pins).** Built on Hanan's theorem (1966): an optimal RSMT for any point set has
a realization using only points on the "Hanan grid," the grid formed by drawing a horizontal and
vertical line through every terminal. For n<=4 terminals that grid has at most 16 points, and an
optimal tree needs at most n-2 of the non-terminal grid points as extra Steiner junctions. We
brute-force every subset of up to (n-2) candidate points, take the rectilinear MST of terminals plus
subset for each, and keep the shortest. At most ~66 subsets of size 2 from <=12 candidates, so this
is effectively instant.

Verified in `tests/test_cost.py::test_three_pin_steiner_tree_beats_spanning_tree`: three points
`(0,0), (2,0), (1,2)` have a naive spanning tree of length 5, but the true RSMT, found via the
Hanan-grid point `(1,0)`, is length 4. This is the textbook case that justifies bothering with the
exact solver at all instead of just using MST everywhere.

**Heuristic (>4 pins).** Rectilinear minimum spanning tree, no Steiner points. Documented
approximation ratio: RMST/RSMT <= 1.5 in the worst case (Hwang, 1976), a well-characterized standard
placement-time proxy.

**Rejected alternative.** A full Steiner-point search for large nets is more accurate but the search
space is exponential in the number of Hanan grid points, which grows as O(n^2) with net arity. Not
viable at nets touching up to 8 pins, evaluated per SA perturbation, tens of thousands of times.

### 2.2 Layer crossings and congestion use the terminal-level MST topology, not the Steiner-augmented one

The exact <=4-pin solver may route through extra Steiner points that aren't real pins. Vias and
congestion are physical phenomena tied to real pins, not a geometric convenience point, so both are
computed against the plain rectilinear-MST topology over the actual terminals, independent of
whichever topology the exact solver used to shorten the geometric length. This decouples "what's the
shortest wire" from "how many vias does this net need," a reasonable and documentable simplification
given the spec doesn't define an actual router.

**Congestion rasterization.** Each net's MST edges are drawn as an L-shaped, horizontal-then-vertical
path on the appropriate 32x32-per-layer grid. When an edge's two endpoints are on different layers,
the horizontal leg is attributed to the source pin's layer and the vertical leg to the destination's
— modeling "route on one layer up to the via, continue on the other after it." The spec doesn't
specify a router, so something reasonable has to be picked here, and this is it.

### 2.3 Netlist generation: preferential attachment on pin count

"Generate |E| ~= 2|V| via preferential attachment (higher pin count => more nets)" doesn't fully
specify the mechanism. Interpretation used: pins are drawn into nets with probability proportional to
their parent component's total pin count, so high-pin-count "connector-like" parts end up in more
nets. Each pin is used by at most one net (a physical pad can only belong to one net), and arity is
drawn uniformly in [2, 8].

## 3. Performance engineering

Simulated annealing needs 50,000 perturbations. A naive implementation recomputing the full board
cost every perturbation was profiled at ~2.3s per full recompute at |V|=1000 — 50,000 of those is
multiple days, not minutes. Two optimizations turned out to be necessary, not optional, and both came
from profiling rather than guessing:

1. **`IncrementalCost`** (incremental_cost.py). After a single component moves, only that
   component's nets (wirelength + congestion-grid cells) and its own overlap contribution against the
   rest of its layer get recomputed — O(degree) and O(V) respectively, not O(V+E). Verified against a
   from-scratch `compute_cost()` after 200 random moves in `tests/test_incremental_cost.py`, and it's
   exact agreement, not just close.

2. **`_rectilinear_mst` is plain Python, not numpy**, despite operating on point arrays. Profiling
   showed `scipy.sparse.csgraph.minimum_spanning_tree` spending most of its time in sparse-matrix
   construction and validation, not the actual MST computation, because every call here operates on
   <=10 points (nets top out at 8 pins, the exact solver adds at most 2 more). At that scale, numpy
   and scipy's per-call overhead exceeds what vectorization buys, and a hand-rolled dense Prim's
   algorithm in plain Python cut this hot path by more than 3x. Probably the single most
   counter-intuitive result of this project: reaching for numpy isn't always faster, and the fix came
   from `cProfile`, not intuition.

Net effect: full-board cost recompute at |V|=1000 dropped from 7.7s to 0.7s, and a single SA
perturbation from ~13ms to ~5ms, bringing a full 50,000-iteration baseline run at |V|=1000 down to
roughly 4-5 minutes (measured, see section 5). One thing worth flagging for anyone scaling this
further: per-perturbation cost is close to independent of |V|, because it's dominated by the
wirelength/Steiner recompute for the handful of nets touching the moved component (bounded by net
arity, not board size), not by the O(V) overlap check. |V|=60 and |V|=1000 full SA runs took similar
wall-clock time in practice.

## 4. Baseline: bugs found and why they mattered

The baseline (force-directed init + SA refinement) surfaced three real correctness issues worth
recording, because each reflects a general lesson rather than a one-off typo.

**Greedy layer assignment has a degenerate global optimum.** Minimizing cross-layer nets in isolation
is trivially solved by putting every component on one layer — zero cross-layer nets by construction —
but that defeats the point of having two layers, since overlap is penalized per-layer and collapsing
onto one roughly doubles the same-layer pairs at risk of overlapping. Fixed with a balance term that
penalizes whichever layer is already more populated, which prevents collapse while still preferring
to group net-connected components when it doesn't fight balance.

**A trial-and-revert loop that didn't reset state between trials.** `_snap_rotations` tries all four
rotations for a component and keeps the best. The per-trial `clamp_to_bounds` call mutates position
in place, and without resetting position before each trial, one trial's clamp silently carried into
the next — so the position ultimately committed didn't necessarily correspond to the rotation
actually evaluated as best. Caught by a test asserting the function can only decrease total overlap,
which failed by about 1 unit. Small enough to almost wave away, which is exactly why the test mattered.

**The cost function has no keepout term at all — it's a pure hard constraint.** SA's cost-driven
acceptance has zero incentive to move a component out of a keepout zone, and nothing stops a jitter
perturbation from moving a component into one and getting accepted, since cost wouldn't object.
Fixed by enforcing "never centered in a keepout" structurally: every perturbation that could move a
component's center is followed by `push_out_of_keepouts` rather than hoping the annealer cares.
Regression-tested in `test_sa_never_leaves_a_component_centered_in_a_keepout`.

**Overlap has a finite, not infinite, cost weight, so SA can freeze with residual overlap.** A full
50,000-iteration run on a 60-component board still left 3 same-layer pairs overlapping. Overlap is
weighted 100x in C(p), and any move that reduces overlap should be unconditionally accepted, but at
very low end-of-schedule temperature, if the only available escape move increases wirelength by more
than 100x the overlap it removes, that move is a net cost increase, and Metropolis acceptance of
uphill moves becomes vanishingly rare as T approaches 0. This isn't a bug in the SA implementation —
it's an intrinsic property of annealing with a large but finite penalty. Fixed with a final
`legalize()` pass after SA completes, so full feasibility is a structural guarantee rather than
something purely emergent from the cost function.

**The overlap-resolution push heuristic has a real blind spot on tangled clusters.** The pairwise
minimum-translation-vector push (each overlapping pair pushed apart along whichever axis needs less
movement) can reach a fixed point without fully resolving when several components mutually overlap at
once. Observed in practice: 6 components in a mutually-overlapping cluster after SA, still unresolved
after 500 passes, because pushing to fix pair A-B can reintroduce overlap with C in the same pass.
Rather than design a more elaborate multi-body resolution rule under time pressure, `legalize()`
falls back to rejection-sampling a fresh random position for any component still overlapping after
the normal pass budget. Guaranteed to terminate given the spec's 40-70% utilization bound leaves real
free area, at the cost of possibly worse wirelength for the relocated components. Robustness over
elegance, a pragmatic and documented trade-off. What I'd build instead with more time: a proper
multi-body overlap resolution, treating the tangled cluster as a small local force-directed
re-relaxation with stronger repulsion, seeded from the stuck configuration, rather than random
rejection sampling — likely produces better final placements for the rare stuck cases.

## 5. Baseline numbers

Single-board runs, seed 42, on this dev machine, CPU only — the baseline doesn't need a GPU:

| \|V\| | cost      | wall-clock time | feasible |
|-----|-----------|------------------|----------|
| 200 | 186,374.8 | 374.2s           | yes      |
| 1000| 6,757,486.0 | 480.0s         | yes      |

The wall-clock time isn't proportional to |V| the way you might expect: 200 to 1000 is 5x the
components but only about 1.3x the time. As section 3 covers, a single SA perturbation's cost is
dominated by the wirelength/Steiner recompute for the handful of nets touching the moved component,
bounded by net arity and independent of |V|, not by the O(V) overlap check. So per-iteration cost is
close to flat across board sizes and total SA time is mostly a function of iteration count (fixed at
50,000 per spec), not |V|.

## 6. Validation spike: does a warm start actually help?

Before committing real time to training a GNN, `scripts/spike_day1_validation.py` checks the central
bet of the whole approach. The baseline is exempt from the 60s budget but the learned placer isn't,
so winning requires starting SA from a much better point so far fewer refinement iterations are
needed. The spike substitutes the cheapest possible stand-in for "a good warm start," a naive,
net-unaware shelf-packing initializer (sorted by height, packed left to right in rows), and compares
naive init plus wall-clock-budgeted SA against a full, unconstrained baseline run, on the same
boards.

**Results:**

| \|V\| | full baseline (cost, time) | naive init only (cost) | naive + budgeted SA (cost, time, feasible) | vs. baseline |
|-----|---|---|---|---|
| 150 | 98,733.4, 334.2s   | 239,783.3 | 170,542.7, 33.4s, **infeasible** | **-72.7%** (far worse) |
| 200 | 171,163.6, 344.7s  | 443,155.0 | 264,123.5, 48.4s, **infeasible** | **-54.3%** (far worse) |

Two findings came out of this, one reassuring and one that sharpened the plan.

First, a naive warm start is nowhere close to sufficient, even given the same time budget the learned
placer will get. That's the reassuring part — it rules out the trivial failure mode where literally
any warm start plus budgeted SA would already look competitive, which would have meant a trained
model wasn't adding real value. A 55-73% gap isn't something a marginally smarter initializer closes.

Second, it raised the bar higher than originally framed. Our own force-directed init, used by both
the baseline and, eventually, the learned placer, is already fast, sub-second even before the
vectorization work in section 3, nowhere near the bottleneck. The actual constraint is that budgeted
SA only gets about 20% of the baseline's 50,000 iterations in a 60s window (measured: ~5ms per
perturbation at both |V|=200 and 1000, so ~55s of budget buys roughly 10,000-11,000 iterations). If a
learned warm start only matches what force-directed already produces, a placer with 5x fewer
refinement iterations than baseline should be expected to lose, not win. The training objective
therefore has to do real work: learn, from patterns across many training boards, a placement that's
already closer to what a fully-annealed baseline would reach, not just replicate force-directed's
local, single-board relaxation faster. That's a harder and more specific target than "any reasonable
init," and it's a direct consequence of this spike rather than a guess.

Both budgeted runs above ended infeasible (residual overlap), which is expected and not a red flag —
the spike calls `simulated_annealing()` directly, without the final `legalize()` backstop that
`place_baseline()` applies. The learned placer's `place()` entry point includes that same backstop,
budgeted to leave a small time margin before the 60s deadline.

Implication going forward: push on both levers, not just model quality. Train to close most of the
gap to full-baseline quality directly, since the differentiable-proxy training objective exists
precisely to let it learn cross-board structure a local method can't, and keep the budgeted-SA
refinement loop as fast per-iteration as possible so the 60s window buys as many polishing iterations
as it can, since the model's output will still benefit from some refinement even if it starts much
closer to optimal than force-directed does.

## 7. Learned placer: architecture, and the pivot away from a residual

`place(board)` (`placer/learned/placer.py`) is the required entry point: GNN forward pass, then the
baseline's own `_greedy_layer_assignment` and `_snap_rotations` heuristics (rotation and layer aren't
modeled by the GNN, see 7.1), then `legalize`, then wall-clock-budgeted `simulated_annealing` — the
same function the baseline uses, just stopped early via `max_seconds` — then a final `legalize`
backstop, for the same reason as baseline section 4d: overlap's finite cost weight means budgeted SA
can also finish with residual overlap.

The GNN (`placer/learned/model.py`) is a bipartite component-net message-passing network. Encode each
component's and net's raw features into an embedding, run a few rounds of message passing where
component embeddings update from the nets touching them and net embeddings update from their member
components, via residual-connected small MLPs, then decode each component's final embedding into a
2D position.

### 7.1 The original design used a residual on top of force-directed init; profiling killed it

The first version had the GNN predict a correction on top of `_force_directed_init`'s output
(`position = init_pos + bounded_residual`), on the reasoning that an untrained model should still
produce something reasonable, close to force-directed init, rather than noise, and that the network
only has to learn a correction rather than placement from scratch.

Profiling a training step at increasing |V| found this reasoning didn't survive contact with the
numbers:

| \|V\| | total step time | `_force_directed_init` | everything else (generate_board, GNN fwd/bwd, loss) |
|-----|---|---|---|
| 50  | 0.240s | 0.094s (39%) | 0.146s |
| 150 | 0.638s | 0.564s (88%) | 0.074s |
| 300 | 3.219s | 3.062s (95%) | 0.157s |

`_force_directed_init`'s 500-iteration force simulation was consuming 95%+ of every training step at
larger |V|, and given training has no fixed dataset (a fresh random board every step, see section 9),
step cost is the training-speed bottleneck. Given training volume turned out to be the single biggest
lever on quality (section 11), this was worth fixing directly rather than working around.

**Decision: drop force-directed init entirely, both at training and inference time.** The GNN now
predicts absolute positions directly from graph structure, through a sigmoid, denormalized by board
dimensions, with no initializer in the loop at all. Verified: per-step time dropped from
0.240s/0.638s/3.219s to 0.036s/0.076s/0.223s at the same three sizes, about 14x faster at |V|=300, and
a real end-to-end `train_learned.py --steps 2000` run, not just the synthetic profiling loop, dropped
from an estimated ~37 minutes to a measured 4m7s. This also frees more of `place()`'s 60s inference
budget for the SA phase, since that overhead is gone at inference too, not just in training.

Was this a wash on quality, though? Two checks, both before committing to the full rewrite:

Same-seed comparison. Both training runs seed with `seed_offset + step`, so at any given step the
exact same random board was generated both times. At step 1990 (|V|=116), the old
residual-on-force-directed-init run's loss was 129,612; the new GNN-only run's was 91,153 on the
identical board — a direct, controlled comparison, not two runs on different boards that happened to
differ.

An ablation, reusing the same before/after methodology throughout this project: build a
force-directed-init-only pipeline and a GNN pipeline on identical seeds, compare cost pre-SA (isolates
the warm start) and post-SA (after a short, equal budget on both sides). Both architectures beat plain
force-directed init, which rules out "the model isn't learning anything real," and the GNN-only
architecture's pre-SA margin over force-directed-init-alone was roughly double the old residual
architecture's — for example seed 1001 went from 26.6% to 50.1%. The post-SA gap narrowed a lot on
both, since SA erases a fair amount of warm-start quality difference given enough of a budget, and one
seed (1001) actually did slightly worse post-SA under the new architecture, +5.3% to +0.1%. A reminder
that pre-SA and post-SA numbers can tell different stories, and both are worth checking.

## 8. The proxy loss: three real bugs found by checking real cost, not proxy loss

The real cost function can't be trained against directly — RSMT/MST construction is a discrete
combinatorial search, and the real overlap term is a hard cutoff. `placer/learned/loss.py` builds a
differentiable stand-in instead, and this section is the record of how many times that stand-in
turned out to disagree with the thing it's supposed to approximate, and how each disagreement was
actually found. Never by reasoning about the loss function alone — always by checking real, non-proxy
cost on real placements.

### 8.1 Iteration 1: smoothed HPWL plus fixed-weight overlap, which looked fine and was quietly missing a third of the cost function

The first version summed two terms: a log-sum-exp-smoothed half-perimeter wirelength and a smoothed
pairwise bounding-box overlap, combined as `wirelength + 100 * overlap`, with the 100 chosen by
analogy to `cost.py`'s own `OVERLAP_WEIGHT`, not independently validated. Both terms are averaged, not
summed, over nets and pairs respectively — an early normalization fix, since an unnormalized sum made
the raw loss swing about 30x between a small and large randomly-sampled board purely from net or pair
count, noise from the model's perspective with nothing to do with placement quality.

Trained for 2000 steps, this produced a placement that lost to the untrained, random-weight model's
own baseline-comparison result (-48.5% at |V|=200 vs. an earlier untrained smoke test's -39.5%) —
training was making things worse, not better. A dedicated ablation, force-directed-init-alone versus
the trained GNN, same seeds, short-budget SA on both, ruled out "the residual is actively harmful": the
trained warm start was genuinely 5-27% better than plain force-directed init, consistently. So the
model was learning something real, it just wasn't enough to matter against the real target
(`place_baseline`, unbudgeted 50,000 SA iterations).

A chunked training run, introduced specifically to get a real, held-out signal instead of trusting
proxy loss alone (see section 9), made the actual problem visible. Over 7 chunks, 35,000 steps, proxy
loss fell smoothly and monotonically — 47.3, 39.4, 38.5, 38.2, 37.9, 37.7, 37.7 — while the real
dev-set score oscillated noisily with no trend — -38.3%, -38.9%, -40.6%, -36.5%, -43.0%, -42.4%,
-35.8%. Proxy loss and real cost had decoupled. That divergence is diagnostic in itself: if optimizing
the proxy were still helping, the thing we actually care about should track it, at least loosely. It
didn't, which meant the proxy itself needed inspecting, not just more training.

Pulling a full cost breakdown, `compute_cost`'s wirelength/overlap/congestion split, not just the
total, on one board made the gap concrete:

| | baseline | learned (chunk-7 best) | gap |
|---|---|---|---|
| wirelength | 137,750.2 | 132,159.5 | learned better by 5,590.7 |
| overlap | 0.0 | 0.0 | tied |
| congestion | 51,653.0 | 136,442.0 | learned worse by 84,789.0 |
| **total** | 189,403.2 | 268,601.5 | gap = 79,198.3 |

Congestion alone exceeded the entire gap; wirelength was actually already winning. The proxy loss had
no congestion term at all — the GNN had never once received a gradient signal telling it not to cram
many nets' routing into the same small region, while the baseline's SA directly optimizes the real,
congestion-inclusive cost for 50,000 iterations. Not a training-volume problem. A structural blind spot
in what the model was ever asked to minimize.

### 8.2 Iteration 2: adding a differentiable congestion term

The real congestion term rasterizes each net's exact MST routing path onto a 32x32-per-layer grid and
penalizes `max(0, demand - 10)^2` per cell — discrete, since it depends on which cells a path
crosses, and per-layer, which is unknown at this stage, the same reason overlap is layer-agnostic
here. `soft_congestion` approximates it by spreading each net's bounding-box footprint across the grid
cells it overlaps, area-weighted and normalized so every net contributes exactly 1.0 total "mass"
regardless of size. A net whose box is small relative to a cell concentrates mass, penalized once
local density exceeds a capacity threshold; a net already spread wide contributes little to any single
cell.

Two normalization passes were needed, both found by direct measurement, not foresight. A naive
magnitude check on an untrained model showed the raw term ranging from 4.18 (|V|=50) to 2048.47
(|V|=1000), a roughly 490x swing purely from board size, the same size-confound problem already fixed
for wirelength and overlap, just missed on the newest term. Fixed by dividing by net count too, since
mass scales with net count the same as the other two terms, bringing the range down to 0.0418-1.0242,
comparable across sizes. And `congestion_weight` was initially a fixed constant, 20.0, chosen only by
eyeballing magnitude parity with wirelength on an untrained model at three board sizes — a sanity
check that it wasn't negligible or dominant, not a real calibration (see 8.4 for why this mattered
later).

### 8.3 Degree-corrected, weighted-average HPWL

Plain HPWL treats every net's degree as equally "tight" against true RSMT. That's provably wrong: HPWL
is a valid lower bound on RSMT for any degree, and exactly tight only for degree <=3, a proven
identity — this project's own exact solver already demonstrates the gap opening at higher degree, see
section 2.1's Steiner-point test. A degree-aware correction factor `q(d)` scaling HPWL toward true
RSMT is standard practice, RISA-style, but the textbook tables are fit on real ASIC placement
benchmarks with different pin and net topology than this generator produces.

The correction implemented here is fit against this project's own `cost.py` functions, not a published
table and not FLUTE, which this project doesn't have and which would calibrate to the wrong target
anyway for degree>=5 nets, since the grader itself uses the cruder RMST heuristic there, per the
spec's documented choice, not a hypothetical true-optimal RSMT. Sampling 60 boards, |V| in [50,400],
force-directed-init positions as the point distribution, and comparing mean of cost.py's own
exact/RMST length over mean HPWL per degree bucket gave:

| degree | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| q(d) | 1.0000 | 1.0000 | 1.0634 | 1.2038 | 1.2998 | 1.3775 | 1.4505 |

q(2) and q(3) fell out exactly at 1.0, matching the proven identity, a strong sanity check that the
HPWL and exact-RSMT implementations agree, not something forced by the fitting procedure. q(4) came
out close to but not exactly 1.0, as expected, since HPWL=RSMT is only sometimes tight at degree 4, not
universally the way it is at 3 and below.

Alongside this, the smoothing model switched from log-sum-exp to a DREAMPlace-style weighted average.
Weighted average is a convex combination of the group's own values, so unlike log-sum-exp, whose
smooth max is always biased above the true max by roughly gamma times log of group size, it can never
imply a wirelength span larger than what's physically there. The numerical-stability trick of shifting
by the true max before exponentiating was already present either way, so the weighted-average switch
wasn't rescuing an active overflow bug, just a structurally better-behaved smoothing choice.

### 8.4 Weight calibration: fixed constants, then EMA self-normalization, then two real bugs, then freeze-after-warmup

`overlap_weight=100` and `congestion_weight=20` were both magic numbers, calibrated only by eyeballing
term magnitude on an untrained model, never validated against what actually matters: real placement
quality. The fix considered was to divide each term by a running exponential moving average of its
own recent magnitude instead of a fixed weight, so every term contributes about equally at every point
in training, adapting as each term's natural scale shifts — overlap starts huge when an untrained
model clusters everything at the board center, then shrinks as it learns to spread out, and congestion
and wirelength may not shrink at the same rate.

This went through two real failure modes before it worked, both found by running it, not by reasoning
about it in advance.

The first was a collapse spiral. Dividing by a continuously-updating EMA has a runaway-collapse risk:
if the model finds a way to drive a term toward zero, the effective gradient pressure to shrink it
further grows as its own EMA shrinks, a positive feedback loop. A run without any floor saw
`wirelength_ema` spiral from about 2.4 to about 1e-5, a roughly 240,000x collapse, before the loss
exploded, since a tiny denominator turns any residual numerator into a huge ratio. Caught mid-run by
watching the per-term EMA values, not just the aggregate loss, which is why `train_learned.py` logs
all three. The fix was to floor each term's EMA at a fraction, `min_scale_ratio=0.05`, of its own
historical peak — generous enough to allow a real 20x reduction, bounded enough to prevent numerical
blow-up.

The second was floor-drift imbalance. The floor stopped the crash, but a longer, 4000-step run then
showed a second problem: once both wirelength's and congestion's EMAs hit their floors, the balance
between them drifted in an uncontrolled direction. Wirelength kept improving at congestion's expense —
congestion's real-cost ratio went from 1.29x worse at 300 steps to 1.48x worse at 4000 floor-only
steps — with no adaptive counter-pressure left, since neither scale was still tracking anything
meaningful. The fix was to calibrate adaptively for a short warmup window, `warmup_steps=300` updates,
then freeze the EMA permanently. This keeps the real benefit, a scale derived from actual
early-training behavior rather than a hand-picked guess, while removing the ongoing feedback loop that
caused both failures: a frozen scale can't chase its own shrinking value, and can't drift once
floored, because nothing is still adapting. Verified stable over a fresh 4000-step run:
`wirelength_ema` and `congestion_ema` settled at healthy, non-degenerate values, 0.325 and 0.168, and
held constant for the remaining ~3700 steps, no drift, no spikes.

## 9. Why proxy loss alone can't be trusted, and the dev-set methodology built to check it

There's no fixed training dataset here — every training step calls `generate_board` with a fresh
seed, so there's nothing to overfit to in the classical sense, the standard reason for a validation
split. But there's a different, real risk this project hit directly: a differentiable proxy can keep
improving on its own terms while the real, non-differentiable cost function it's supposed to
approximate doesn't, exactly what happened in 8.1. Watching proxy loss during training isn't
sufficient evidence of progress. Only checking real cost is.

**Three-way seed separation.** Training uses `TRAIN_SEED_OFFSET=10,000,000`; the final benchmark
(`scripts/benchmark.py`) reserves seeds 1000-1009; a third, small, fixed dev set, 3 boards at
|V|=200, 3 at |V|=1000, seeds 20,000,000 and up, exists purely to make "keep training or stop"
decisions along the way, never trained on and never used for the final reported number. Reusing the
benchmark's own seeds for interim decisions would be a mild form of peeking at the test set —
repeatedly checking against the exact boards the final score comes from risks, even unintentionally,
picking whichever stopping point happens to look best on those specific boards rather than one that
reflects genuine convergence.

**Caching the expensive half.** Each dev board's baseline cost, `place_baseline`, unbudgeted, about
5-8 minutes per board, is computed once and cached in `placer/checkpoints/dev_baseline_cache.json`.
Every subsequent check (`scripts/dev_eval.py`) only re-runs the much faster learned placer against
the cached baseline numbers.

**Chunked training with auto-stop** (`scripts/train_chunked.py`). Train in chunks of 5,000 steps,
dev-eval the real cost after each chunk with the full wirelength/overlap/congestion breakdown, not
just the aggregate, keep whichever checkpoint scores best separately as `placement_gnn_best.pt`, and
stop automatically after a patience of 3 consecutive non-improving chunks. Training itself is
resumable — the optimizer state and the loss normalizer's EMA state are both saved and restored
across chunks, so resuming doesn't reset either mid-run.

**A real measurement-noise finding.** Re-running `dev_eval.py` on the exact same saved checkpoint
gave two different results on two different invocations: -42.2% during the training run's own
chunk-boundary check, -38.8% on a standalone re-evaluation moments later. Same weights, same 6
boards, same seeds. The cause is almost certainly the same phenomenon already observed for the
classical baseline (section 4, and again independently in two different full-benchmark runs on
identical seeds giving different baseline costs): simulated annealing is a long, chaotic iterative
process, and tiny floating-point or threading-level nondeterminism, likely from multi-threaded BLAS
operations in the force-directed init and cost bookkeeping, compounds over enough iterations into a
meaningfully different final trajectory. The practical implication is that a single chunk's "no
improvement" verdict in the auto-stop logic could itself be measurement noise rather than a genuine
plateau. `patience=3`, requiring three consecutive non-improving reads rather than one, exists
specifically to be robust to this, but it's worth stating plainly that dev-set numbers on 6 boards,
individually noisy on top of that, are a directional signal, not a precise one.

## 10. Time-budget robustness: two problems that had nothing to do with model quality

**A flat safety margin doesn't scale with board size.** `place()` reserves `time_budget - elapsed -
safety_margin` seconds for budgeted SA. An initial fixed `SAFETY_MARGIN_SECONDS=6.0` was verified only
at |V|=200 (58.55s, comfortably under budget) before being used for a full benchmark run, which then
showed every single |V|=1000 board finishing at 61.0-61.7s, over the 60s limit, despite the same
margin. The uncounted cost is the final `legalize()` backstop, section 4's overlap-cleanup pass, which
runs after budgeted SA and isn't itself time-bounded, and its cost, overlap-resolution passes and the
relocate-stuck fallback, grows with board size, so a margin calibrated at the smallest tested size
wasn't nearly enough at the largest. The fix was to scale the margin with |V|:
`BASE_SAFETY_MARGIN_SECONDS=5.0 + PER_COMPONENT_SAFETY_MARGIN_SECONDS=0.02 * num_components`, verified
this time at both |V|=200 (56.43s) and |V|=1000 (43.98s) directly, not extrapolated from one size —
the earlier mistake was fixing a bug at one scale and assuming it generalized without checking the
scale that actually broke.

**SA calibration cost is real and was previously uncounted.** `_calibrate_temperature`'s 500-sample
calibration runs before `simulated_annealing`'s own `max_seconds` timer starts, so its wall-clock cost
is invisible to the budget the SA loop thinks it has, and it just eats into the outer safety margin
instead. Made the sample count configurable so the learned placer can use fewer samples than the
baseline's 500 (the baseline has no time limit, so this doesn't matter for it) and free more of the
budget for actual annealing, which matters most exactly where it's scarcest: at |V|=1000, a rough
estimate puts budgeted SA at only single-digit touches per component, versus the baseline's unbudgeted
50,000 iterations across the whole board.

## 11. Results: four training runs, and what each one taught

| Run | Architecture | Proxy loss | Steps | Result |
|---|---|---|---|---|
| 1 | Force-directed init + GNN residual | HPWL (log-sum-exp) + fixed overlap weight | 2,000 | Full benchmark: -48.5% (\|V\|=200, 10 bd), -49.8% (\|V\|=1000, 10 bd) |
| 2 | GNN-only (pivot, 7.1) | same proxy as Run 1 | 2,000 | Partial benchmark: -45.1% (\|V\|=200, 3 bd); dev-set \|V\|=1000, first reading for this architecture: ~-21.2% |
| 3 | GNN-only | same proxy, no congestion term yet | 35,000 (7 chunks, manually stopped) | Best dev score -35.8% (chunk 4 of 7); proxy loss and dev cost visibly decoupled (8.1), which led directly to finding the missing congestion term |
| 4 | GNN-only | + congestion, degree-correction, weighted-average smoothing, freeze-after-warmup normalization | 25,000 (5 chunks, auto-stopped on patience) | Best dev score -42.2%/-38.8% (noise-dependent, section 9); congestion ratio plateaued at roughly 1.3-1.45x worse than baseline regardless of training duration, since 300, 4000, and 25,000 steps all landed in the same band |

Run 4's plateau is the most important honest finding here. Wirelength consistently ends up better
than baseline, so the model has clearly learned to exploit that side of the objective well, but
congestion's gap hasn't moved with additional training at any of the three durations tested. That
reads as the bounding-box-density congestion proxy (8.2) having its own fidelity ceiling, not as an
undertrained model — a genuinely different problem than the one three previous fixes solved, and one
that would need a materially better congestion approximation, closer to the true rasterized path and
not just bounding-box overlap, to move further rather than more training volume.

**Final benchmark**, `scripts/benchmark.py`, 10 boards per size, both sizes, against
`placement_gnn_best.pt`, Run 4's chunk-2 checkpoint. These are the actual spec deliverable numbers:

| \|V\| | baseline cost (mean +/- std) | baseline time | learned cost (mean +/- std) | learned time | paired improvement |
|---|---|---|---|---|---|
| 200  | 174,560 +/- 10,705   | 335.7s | 259,916 +/- 16,585  | 52.1s | -49.1% +/- 8.1% |
| 1000 | 6,352,346 +/- 139,330 | 420.4s | 7,320,154 +/- 271,159 | 38.2s | -15.2% +/- 3.1% |

Neither size clears the spec's 10% target, which is the honest bottom line of this project. But the
two sizes tell a genuinely different story, not just "losing everywhere by a similar margin."

|V|=1000 is far closer to competitive than |V|=200, -15.2% versus -49.1%, with less than half the
variance too. This isn't noise. It echoes a signal seen much earlier — Run 2's dev-set |V|=1000
reading was already notably better than its |V|=200 reading, the first time this architecture was
ever checked at that size — and it has a plausible mechanical explanation: baseline's fixed 50,000 SA
iterations are a much thinner budget per component at |V|=1000 (roughly 50 touches per component)
than at |V|=200 (roughly 250 touches per component, by the same math), so baseline's own
unbudgeted-SA advantage is diluted at scale, giving relatively more credit to whatever the learned
placer's warm start already got right, which is wirelength, consistently, across every checkpoint
tested. Wall-clock also dropped from 52.1s to 38.2s between the two sizes — the safety margin (section
10) deliberately reserves more of the 60s budget at larger |V| (25s versus 9s), leaving less time for
SA to run at all, which is itself consistent with this size-dependent gap being about budget
allocation, not just proxy quality.

Given the congestion-ratio plateau held at both sizes roughly equally, and the relative gap still
narrowed substantially at |V|=1000 anyway, the most defensible reading is that the architecture,
proxy loss, and training approach here are on a sound trajectory but ran out of runway, in training
volume, proxy fidelity, and remaining engineering time, before reaching the target. Not a dead end.
Section 13 covers what would most plausibly close the remaining gap, including a different approach
explored late in this project that looks genuinely promising but wasn't finished in time to verify
properly.

## 12. Failure modes anticipated

- **Proxy/real divergence recurring elsewhere.** Section 8.1's core lesson, that a differentiable
  proxy can improve on its own terms while real cost doesn't follow, isn't fully closed off by the
  fixes so far. It's the general risk of this whole approach, and the dev-set check (section 9) is a
  mitigation, not a guarantee, especially since it's itself noisy.
- **SA-budget starvation at large |V|.** However good the warm start gets, |V|=1000 gives budgeted SA
  only a small fraction of baseline's unbudgeted iteration count. A warm start that isn't close to
  fully-annealed quality has little room for SA to close the remaining gap — a structural ceiling on
  how much a better proxy alone can achieve, independent of proxy fidelity.
- **No via or layer-crossing signal in training at all.** Layer is decided entirely by a heuristic
  (`_greedy_layer_assignment`) after the GNN runs; the GNN itself has zero training signal about
  via/crossing cost, unlike the baseline's SA, which directly optimizes it via layer-flip
  perturbations. A real, uncounted gap, deliberately deferred (see section 13).
- **Congestion proxy ceiling.** Demonstrated, not hypothetical — the plateau held across three
  training durations spanning two orders of magnitude.
- **Board-size distribution shift.** Training samples |V| in [50,1000] uniformly; the benchmark
  evaluates specifically at 200 and 1000. Performance at sizes never directly emphasized during
  training, very small or in between, is untested.

## 13. What I'd build next with more time

**A per-board analytic optimizer, GPU-native, in place of the amortized GNN — the highest-priority
item here, and the one I'd start with.** Everything in sections 7-11 trains a model ahead of time to
generalize across an entire board-size distribution, then evaluates it once per test board in a
single forward pass. That's a strictly harder problem than the task actually asks for. The spec gives
a 60-second-per-board budget and a GPU, which is a much more direct fit for optimizing the placement
itself, directly against the board actually being placed, the way DREAMPlace and similar analytic
placers work: treat each component's position as a parameter, run gradient descent against the exact
same differentiable proxy already built in `loss.py` for as many steps as the time budget allows, then
hand off to the same legalize/rotate/layer/SA pipeline already in place. No training data, no
checkpoint, no generalization question at all — just a much larger optimization budget spent on the
one board that actually matters.

I started building this late, on a branch, once it became clear the amortized approach had hit a real
ceiling (section 11), and ran out of time to finish verifying it properly, so it isn't what's
submitted. Worth recording what happened, since it's informative either way. On CPU, after fixing a
few real bugs along the way — an unbounded optimization loop that could drift into expensive-to-clean
overlap, a per-step time-check granularity that let large boards overshoot the 60s budget, and reusing
`_force_directed_init` for the starting point without re-checking its cost at scale, which turned out
to eat 36 of the 60 seconds at |V|=1000 on its own — single-board spot checks landed meaningfully
better than the shipped GNN checkpoint: roughly -31% at |V|=200 and roughly -20% at |V|=1000, against
that checkpoint's real benchmark numbers of -49.1% and -15.2%. That's encouraging, but it's CPU-only
evidence for an approach whose entire value proposition depends on GPU throughput, and I found real
GPU-specific bottlenecks I didn't have time to fully resolve: a loss normalizer that synchronized the
GPU and CPU three separate times per step instead of once, and a small lookup table that was
re-transferred to the GPU on every single step instead of being cached. Both were real, both got
fixed, and neither turned out to be the whole story — GPU wall-clock still came back close to CPU
wall-clock afterward, which means there's at least one more bottleneck I didn't find in the time
available. I'd want to profile step-by-step on the actual GPU rather than guessing at individual
fixes, which is exactly what running out of time prevented.

The reasoning for why this is worth finishing: it directly targets the one weakness section 11
actually demonstrates. Wirelength already beats baseline under the current approach; congestion is the
entire gap, and a per-board optimizer removes the main reason that's hard to fix under amortized
training. An amortized model has to find one setting that works reasonably across an entire
distribution of boards; a per-board optimizer only ever has to get one board right, with orders of
magnitude more optimization steps available for it specifically. There's also a real, if secondary,
option to keep a trained component in the loop without contradicting any of this: use a small GNN,
trained the easier way, to predict a good starting point for the per-board optimizer rather than the
final placement itself, so an imperfect prediction just gets refined away instead of mattering on its
own. I wouldn't build that without first checking, via the same kind of ablation used throughout this
project, that it actually earns its complexity over a cheap random or force-directed start.

**A materially better congestion proxy**, independent of the item above. Something closer to
DREAMPlace's electrostatic-density formulation, a proper Poisson-equation-based spreading force,
rather than raw bounding-box overlap, or at minimum a closer approximation of the true rasterized
L-shaped path instead of the full bounding box. Likely the highest-leverage change to the proxy loss
itself, given section 11's plateau.

**Via and layer awareness during training**, not just at the heuristic-assignment stage afterward. For
example, having the model also predict a soft layer logit, with a differentiable via-crossing penalty
feeding back into training, so the warm start itself accounts for layer cost rather than leaving it
entirely to post-hoc heuristics and SA.

**A real hyperparameter search**, not ad-hoc calibration. Gamma, the EMA's min-scale ratio, the warmup
step count, and the congestion free-capacity constant were each set from a single reasonable check,
not a proper sweep against real dev-set cost. Given how much the proxy-loss iteration in section 8
mattered, this is likely underinvested relative to its payoff.

**A learning-rate schedule.** Every run so far used a flat `lr=1e-3` for the entire run. Decaying it
over training is a standard fix for exactly the kind of noisy, plateauing behavior seen in Runs 3 and
4, and hasn't been tried at all yet.

**More model capacity or more message-passing rounds**, once the above are in place. Deliberately not
pursued yet, since there wasn't strong evidence capacity, rather than the objective itself, was the
bottleneck, and it's the more expensive lever to iterate on.

**A proper multi-body overlap-resolution pass** in `legalize()`. Section 4 already flags this as a
known, deliberately-deferred gap, independent of the learned placer.

## 14. What breaks if board size doubles, or the cost function changes

**Board size doubles, |V|=2000.** The classical baseline is largely unaffected, since SA's
per-iteration cost is dominated by net arity, not |V| (section 3), so 50,000 iterations still takes
roughly the same wall-clock time. The learned placer is more exposed: `soft_overlap`'s all-pairs
computation is O(|V|^2), a 2000x2000 tensor is still trivial at training time so this is a
training-cost concern rather than a correctness one, and more importantly, budgeted SA's already-thin
iteration count per component at |V|=1000 would roughly halve again at |V|=2000, exactly the axis
where section 11's results are already weakest. Training would also need re-checking, since the
current range tops out at |V|=1000, so a model trained on this generator has never seen |V|=2000
boards and would be extrapolating outside its training distribution, section 12's board-size-shift
risk, just more severe.

**Cost function changes.** The classical baseline is more robust to this than the learned placer by
construction, since SA optimizes whatever `IncrementalCost`/`compute_cost` compute directly, so a
changed weight or an added term is immediately reflected with no retraining needed. The learned placer
would need its proxy loss updated to match — exactly the section 8 story would repeat, a proxy term
missing or miscalibrated relative to a changed real cost function produces the same kind of silent
divergence found in 8.1 — and then retrained from the new objective. There's no way around this given
the proxy has to be re-derived from whatever the new real cost function actually rewards. If the
change added a genuinely new kind of penalty rather than just a reweighting, it would need its own
differentiable approximation built from scratch, the same way congestion did in 8.2.
