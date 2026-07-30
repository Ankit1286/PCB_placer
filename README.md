# Quilter PCB Placer

A system that takes an unplaced PCB board and returns a placed one: a classical baseline
(force-directed init + simulated annealing) and a learned placer (GNN warm start + budgeted SA
refinement) that's meant to beat it under a 60-second time budget.

See [DESIGN.md](DESIGN.md) for the full design rationale, decisions made and rejected, what worked,
what didn't, and why. This file is just "how do I run it."

## Setup

```
pip install -r requirements.txt
pip install -e .
```

Requires Python 3.10+. No GPU required (this was developed and benchmarked CPU-only).

## The two required entry points

```python
from placer import generate_board, place, GeneratorConfig

board = generate_board(GeneratorConfig(num_components=200, seed=0))
placed_board = place(board)  # GNN warm start + budgeted SA, returns within 60s
```

The classical baseline (exempt from the 60s budget, used as the thing to beat) is a separate entry
point:

```python
from placer.baseline import place_baseline

placed_board, cost_state, _ = place_baseline(board.copy_unplaced(), seed=0)
print(cost_state.total)  # C(p)
```

`placer.feasibility.check_feasibility(board)` and `placer.cost.compute_cost(board)` check/score any
placed board (both used throughout the codebase and its tests).

## Running the tests

```
pytest tests/ -v
```

49 tests covering the data model, cost function, incremental cost tracking, legalization, the
baseline algorithm, the generator, and the learned placer's model/loss/pipeline wiring.

## Benchmarking baseline vs. learned

```
python scripts/benchmark.py                 # both sizes (|V|=200, 1000), 10 boards each -- slow (hours)
python scripts/benchmark.py --num-boards 3   # quicker smoke run
python scripts/benchmark.py --sizes 200      # just one size
```

Reports mean/std cost and wall-clock time per placer per size, plus the paired % improvement, and
logs everything to a local MLflow run (`mlflow ui` to view). Note: `place_baseline` is *not*
time-budgeted (per spec) and takes several minutes per board at |V|=1000 -- the full default run is
genuinely slow, budget hours for it, not minutes.

## Training the learned placer

The GNN has no fixed training set -- every step generates a fresh random board via `generate_board`
and trains directly against a differentiable proxy of the real cost function (see
`placer/learned/loss.py` and DESIGN.md for why). Three related scripts:

**`train_learned.py`** -- a single training run, resumable:
```
python scripts/train_learned.py --steps 5000 --min-components 50 --max-components 1000
python scripts/train_learned.py --steps 5000 --resume-from placer/checkpoints/placement_gnn.pt
```

**`train_chunked.py`** -- the recommended way to train: runs `train_learned.py` in chunks, evaluating
real (non-proxy) cost against a held-out dev set after each chunk, auto-stopping once the dev score
stops improving for `--patience` consecutive chunks:
```
python scripts/train_chunked.py --chunk-size 5000 --max-chunks 10 --patience 3 \
    --min-components 50 --max-components 1000
```
Produces two checkpoints: `placement_gnn.pt` (the last chunk) and `placement_gnn_best.pt` (whichever
chunk actually scored best on the dev set -- use this one for `place()`/benchmarking).

**`dev_eval.py`** -- run the dev-set check standalone against any checkpoint (also prints the real
cost *breakdown* -- wirelength/overlap/congestion separately -- not just the aggregate):
```
python scripts/dev_eval.py --checkpoint placer/checkpoints/placement_gnn_best.pt
```
The dev set's baseline costs are computed once and cached to
`placer/checkpoints/dev_baseline_cache.json`, so repeat evaluations only need to run the (much
faster) learned placer, not recompute baseline every time.

## Project layout

```
placer/
  board.py, generator.py, cost.py, incremental_cost.py, feasibility.py, legalize.py, baseline.py
  learned/
    model.py      GNN warm-start model (board_to_tensors + PlacementGNN)
    loss.py        differentiable proxy loss the GNN trains against
    placer.py      place() -- the learned pipeline's entry point
scripts/
  benchmark.py            paired baseline-vs-learned comparison (spec deliverable #3)
  train_learned.py         single training run
  train_chunked.py         chunked training with dev-set-driven auto-stop
  dev_eval.py              standalone dev-set check against any checkpoint
  spike_day1_validation.py the Day-1 warm-start validation experiment (see DESIGN.md sec. 6)
tests/                     pytest suite
```

DESIGN.md walks through every file's role and the reasoning behind it in much more depth.
