"""Held-out dev-set evaluation, for tracking real (non-proxy) progress during a long training run.

This is a *third*, separate range of seeds -- distinct from both scripts/train_learned.py's training
seeds (10,000,000+) and scripts/benchmark.py's reserved seeds (1000-1009, used only for the final
reported number). The dev set exists purely to decide, mid-run, whether a checkpoint is actually
getting better at the real objective, without ever touching the seeds the final deliverable number
comes from -- repeatedly checking progress against those and using it to decide when to stop would
be a mild form of peeking at the number that's ultimately reported.

A falling proxy training loss doesn't guarantee the real placement is improving (this project's own
history: an earlier checkpoint's proxy loss fell while its real benchmark cost got worse than an
untrained model's). This module is the check that would catch that divergence early, on a handful of
boards, instead of finding out only after a multi-hour run and a multi-hour benchmark.

Baseline cost per dev board doesn't depend on which checkpoint is being evaluated and is expensive
(full unbudgeted SA -- several minutes at |V|=1000), so it's computed once and cached to disk rather
than recomputed on every check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from placer.baseline import place_baseline
from placer.cost import compute_cost
from placer.feasibility import check_feasibility
from placer.generator import GeneratorConfig, generate_board

DEV_SEED_OFFSET = 20_000_000  # clear of both train_learned.py's (10,000,000+) and benchmark.py's (1000-1009)
DEV_BOARDS: list[tuple[int, int]] = [
    (200, DEV_SEED_OFFSET + 0),
    (200, DEV_SEED_OFFSET + 1),
    (200, DEV_SEED_OFFSET + 2),
    (1000, DEV_SEED_OFFSET + 10),
    (1000, DEV_SEED_OFFSET + 11),
    (1000, DEV_SEED_OFFSET + 12),
]
BASELINE_CACHE = Path(__file__).resolve().parent.parent / "placer" / "checkpoints" / "dev_baseline_cache.json"


def _board_key(num_components: int, seed: int) -> str:
    return f"{num_components}_{seed}"


def _breakdown_dict(breakdown) -> dict:
    return {
        "total": breakdown.total,
        "wirelength": breakdown.wirelength,
        "overlap": breakdown.overlap,
        "congestion": breakdown.congestion,
    }


def _baseline_costs() -> dict[str, dict]:
    """Return cached baseline cost breakdowns for every dev board, computing (and caching) any missing."""
    costs: dict[str, dict] = {}
    if BASELINE_CACHE.exists():
        costs = json.loads(BASELINE_CACHE.read_text())

    changed = False
    for num_components, seed in DEV_BOARDS:
        key = _board_key(num_components, seed)
        # Old cache format stored a plain float (total only) -- treat as missing and recompute
        # with the full breakdown, since that's what evaluate_dev_set now needs.
        if key in costs and isinstance(costs[key], dict):
            continue
        board = generate_board(GeneratorConfig(num_components=num_components, seed=seed))
        placed, cost_state, _ = place_baseline(board.copy_unplaced(), seed=seed)
        report = check_feasibility(placed)
        if not report.is_feasible:
            raise RuntimeError(f"Dev baseline infeasible (seed={seed}): {report.summary()}")
        costs[key] = _breakdown_dict(compute_cost(placed))
        changed = True

    if changed:
        BASELINE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_CACHE.write_text(json.dumps(costs, indent=2))
    return costs


def evaluate_dev_set(checkpoint_path: Path) -> dict:
    """Run the full learned-placer pipeline on the fixed dev set; return paired improvement vs. baseline,
    plus each side's real cost breakdown (wirelength/overlap/congestion), not just the aggregate total --
    the congestion gap that motivated adding a congestion-aware proxy term would be invisible in the
    aggregate alone.
    """
    from placer.learned.placer import place as learned_place

    baseline_costs = _baseline_costs()
    improvements, per_board = [], []
    for num_components, seed in DEV_BOARDS:
        board = generate_board(GeneratorConfig(num_components=num_components, seed=seed))
        placed = learned_place(board, seed=seed, checkpoint_path=checkpoint_path)
        report = check_feasibility(placed)
        if not report.is_feasible:
            raise RuntimeError(f"Dev eval infeasible (seed={seed}): {report.summary()}")

        learned = _breakdown_dict(compute_cost(placed))
        baseline = baseline_costs[_board_key(num_components, seed)]
        pct = (baseline["total"] - learned["total"]) / baseline["total"] * 100
        improvements.append(pct)
        per_board.append(
            {
                "num_components": num_components,
                "seed": seed,
                "baseline": baseline,
                "learned": learned,
                "improvement_pct": pct,
            }
        )

    mean_breakdown = {
        side: {
            term: float(np.mean([b[side][term] for b in per_board]))
            for term in ("total", "wirelength", "overlap", "congestion")
        }
        for side in ("baseline", "learned")
    }

    return {
        "mean_improvement_pct": float(np.mean(improvements)),
        "std_improvement_pct": float(np.std(improvements)),
        "per_board": per_board,
        "mean_breakdown": mean_breakdown,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_ckpt = Path(__file__).resolve().parent.parent / "placer" / "checkpoints" / "placement_gnn.pt"
    parser.add_argument("--checkpoint", type=Path, default=default_ckpt)
    args = parser.parse_args()

    result = evaluate_dev_set(args.checkpoint)
    for b in result["per_board"]:
        print(
            f"  |V|={b['num_components']:4d} seed={b['seed']}: "
            f"baseline={b['baseline']['total']:,.0f}  learned={b['learned']['total']:,.0f}  {b['improvement_pct']:+.1f}%"
        )
    print(f"\n  dev paired improvement: {result['mean_improvement_pct']:+.1f}% +/- {result['std_improvement_pct']:.1f}%")
    mb = result["mean_breakdown"]
    print(f"\n  mean breakdown (baseline vs learned):")
    for term in ("wirelength", "overlap", "congestion", "total"):
        print(f"    {term:>10}: {mb['baseline'][term]:>12,.0f}   vs   {mb['learned'][term]:>12,.0f}")


if __name__ == "__main__":
    main()
