"""Paired baseline-vs-learned cost comparison at |V|=200 and |V|=1000, per the spec's deliverable #3.

Usage:
    python scripts/benchmark.py                 # both sizes, 10 boards each
    python scripts/benchmark.py --sizes 200      # just one size
    python scripts/benchmark.py --num-boards 3   # quicker smoke run

For each size, the same `num_boards` random boards (fixed seeds, so the
comparison is paired -- baseline and learned placer see identical boards)
are placed by both the baseline and the learned placer. Reports mean/std
cost and wall-clock time per placer per size, plus the paired % improvement,
and logs everything to a local MLflow run.
"""

from __future__ import annotations

import argparse
import time

import mlflow
import numpy as np

from placer.baseline import place_baseline
from placer.feasibility import check_feasibility
from placer.generator import GeneratorConfig, generate_board

try:
    from placer.learned.placer import place as learned_place

    LEARNED_AVAILABLE = True
except ImportError:
    LEARNED_AVAILABLE = False


def _run_one_size(num_components: int, num_boards: int, seed_offset: int) -> dict:
    baseline_costs, baseline_times = [], []
    learned_costs, learned_times = [], []

    for i in range(num_boards):
        seed = seed_offset + i

        board = generate_board(GeneratorConfig(num_components=num_components, seed=seed))
        t0 = time.perf_counter()
        baseline_board, cost_state, _ = place_baseline(board.copy_unplaced(), seed=seed)
        baseline_time = time.perf_counter() - t0
        baseline_report = check_feasibility(baseline_board)
        if not baseline_report.is_feasible:
            raise RuntimeError(f"Baseline produced an infeasible board (seed={seed}): {baseline_report.summary()}")
        baseline_costs.append(cost_state.total)
        baseline_times.append(baseline_time)

        if LEARNED_AVAILABLE:
            learned_board = board.copy_unplaced()
            t0 = time.perf_counter()
            placed = learned_place(learned_board)
            learned_time = time.perf_counter() - t0
            report = check_feasibility(placed)
            if not report.is_feasible:
                raise RuntimeError(f"Learned placer produced an infeasible board (seed={seed}): {report.summary()}")
            from placer.cost import compute_cost

            learned_costs.append(compute_cost(placed).total)
            learned_times.append(learned_time)

        print(
            f"  |V|={num_components} board {i+1}/{num_boards} (seed={seed}): "
            f"baseline cost={baseline_costs[-1]:,.0f} ({baseline_time:.1f}s)"
            + (f", learned cost={learned_costs[-1]:,.0f} ({learned_time:.1f}s)" if LEARNED_AVAILABLE else "")
        )

    result = {
        "baseline_cost_mean": float(np.mean(baseline_costs)),
        "baseline_cost_std": float(np.std(baseline_costs)),
        "baseline_time_mean": float(np.mean(baseline_times)),
    }
    if LEARNED_AVAILABLE:
        result["learned_cost_mean"] = float(np.mean(learned_costs))
        result["learned_cost_std"] = float(np.std(learned_costs))
        result["learned_time_mean"] = float(np.mean(learned_times))
        paired_improvement = [(b - l) / b * 100 for b, l in zip(baseline_costs, learned_costs)]
        result["paired_improvement_mean_pct"] = float(np.mean(paired_improvement))
        result["paired_improvement_std_pct"] = float(np.std(paired_improvement))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 1000])
    parser.add_argument("--num-boards", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=1000)
    args = parser.parse_args()

    if not LEARNED_AVAILABLE:
        print("NOTE: placer.learned.placer.place not found yet -- reporting baseline-only numbers.\n")

    mlflow.set_experiment("quilter-placer-benchmark")
    with mlflow.start_run():
        for num_components in args.sizes:
            print(f"\n=== |V|={num_components} ({args.num_boards} paired boards) ===")
            result = _run_one_size(num_components, args.num_boards, args.seed_offset)
            for key, value in result.items():
                mlflow.log_metric(f"V{num_components}_{key}", value)

            print(f"\n  baseline: {result['baseline_cost_mean']:,.0f} +/- {result['baseline_cost_std']:,.0f}"
                  f"  (mean time {result['baseline_time_mean']:.1f}s)")
            if LEARNED_AVAILABLE:
                print(f"  learned:  {result['learned_cost_mean']:,.0f} +/- {result['learned_cost_std']:,.0f}"
                      f"  (mean time {result['learned_time_mean']:.1f}s)")
                print(f"  paired improvement: {result['paired_improvement_mean_pct']:+.1f}%"
                      f" +/- {result['paired_improvement_std_pct']:.1f}%"
                      f"  (target: >= 10%)")


if __name__ == "__main__":
    main()
