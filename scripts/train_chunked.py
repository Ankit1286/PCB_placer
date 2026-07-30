"""Chunked training with periodic dev-set evaluation and automatic early stopping.

Rather than committing to one large, blind step count upfront, this trains in fixed-size chunks and,
after each one, checkpoints and evaluates *real* cost (not the proxy loss) against `place_baseline`
on a small fixed dev set (scripts/dev_eval.py -- seeds distinct from both training and the final
benchmark). This is the check that would catch proxy-vs-real divergence early (a falling proxy loss
without the real placement actually improving), rather than only finding out after the full run and
a multi-hour benchmark.

Stops automatically once the dev-set paired improvement hasn't improved for `--patience` consecutive
chunks, or after `--max-chunks` chunks, whichever comes first. The checkpoint with the best dev score
seen so far is kept separately (`--best-checkpoint-out`), since the *last* chunk isn't necessarily
the *best* one once a plateau/regression triggers the stop.

Usage:
    python scripts/train_chunked.py
    python scripts/train_chunked.py --chunk-size 2000 --max-chunks 3  # quick smoke test
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dev_eval import evaluate_dev_set
from train_learned import CHECKPOINT_DIR, TRAIN_SEED_OFFSET, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--max-chunks", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3, help="consecutive non-improving chunks before stopping")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--min-components", type=int, default=50)
    parser.add_argument("--max-components", type=int, default=1000)
    parser.add_argument("--checkpoint-out", type=Path, default=CHECKPOINT_DIR / "placement_gnn.pt")
    parser.add_argument("--best-checkpoint-out", type=Path, default=CHECKPOINT_DIR / "placement_gnn_best.pt")
    args = parser.parse_args()

    best_score = float("-inf")
    chunks_since_improvement = 0
    resume_from: Path | None = None

    for chunk_idx in range(args.max_chunks):
        step_lo, step_hi = chunk_idx * args.chunk_size, (chunk_idx + 1) * args.chunk_size
        print(f"\n=== Chunk {chunk_idx + 1}/{args.max_chunks}  (global steps {step_lo}-{step_hi}) ===")

        mean_loss = train(
            steps=args.chunk_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            min_components=args.min_components,
            max_components=args.max_components,
            seed_offset=TRAIN_SEED_OFFSET + step_lo,
            checkpoint_out=args.checkpoint_out,
            resume_from=resume_from,
        )
        resume_from = args.checkpoint_out
        print(f"  chunk mean proxy loss: {mean_loss:,.1f}")

        print("  evaluating dev set (real cost vs. baseline, 6 fixed held-out boards)...")
        dev_result = evaluate_dev_set(args.checkpoint_out)
        for b in dev_result["per_board"]:
            print(
                f"    |V|={b['num_components']:4d} seed={b['seed']}: "
                f"baseline={b['baseline']['total']:,.0f}  learned={b['learned']['total']:,.0f}  {b['improvement_pct']:+.1f}%"
            )
        mb = dev_result["mean_breakdown"]
        for term in ("wirelength", "overlap", "congestion", "total"):
            print(f"    mean {term:>10}: baseline={mb['baseline'][term]:>12,.0f}   learned={mb['learned'][term]:>12,.0f}")
        score = dev_result["mean_improvement_pct"]
        print(f"  dev paired improvement: {score:+.1f}% +/- {dev_result['std_improvement_pct']:.1f}%  (best so far: {best_score:+.1f}%)")

        if score > best_score:
            best_score = score
            chunks_since_improvement = 0
            args.best_checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(args.checkpoint_out, args.best_checkpoint_out)
            print(f"  new best -- saved to {args.best_checkpoint_out}")
        else:
            chunks_since_improvement += 1
            print(f"  no improvement over best ({chunks_since_improvement}/{args.patience} chunks without improvement)")

        if chunks_since_improvement >= args.patience:
            print(f"\nStopping early after chunk {chunk_idx + 1}: no dev-set improvement for {args.patience} consecutive chunks.")
            break
    else:
        print(f"\nReached max-chunks ({args.max_chunks}) without triggering early stop.")

    print(f"\nBest checkpoint: {args.best_checkpoint_out}  (dev paired improvement {best_score:+.1f}%)")


if __name__ == "__main__":
    main()
