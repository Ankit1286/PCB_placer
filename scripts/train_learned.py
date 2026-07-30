"""Training loop for the learned placer's GNN warm-start model (placer/learned/model.py).

Self-supervised / analytic training: no labels exist for this task (there's no "correct"
placement to imitate), so each step samples a fresh procedurally-generated board and backprops
directly through a differentiable proxy of the true placement cost (placer/learned/loss.py) --
the same "optimize directly against a differentiable relaxation of the objective" approach used in
analytic global placement.

Training seeds default to a range far above scripts/benchmark.py's paired-comparison seeds
(seed_offset=1000, 10 boards per size) so a long training run can never accidentally train on a
board the benchmark later scores against.

Usage:
    python scripts/train_learned.py --steps 2000
    python scripts/train_learned.py --steps 50 --min-components 50 --max-components 100  # smoke test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import torch

from placer.generator import GeneratorConfig, generate_board
from placer.learned.loss import ProxyLossNormalizer, proxy_cost
from placer.learned.model import PlacementGNN, board_to_tensors

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "placer" / "checkpoints"
TRAIN_SEED_OFFSET = 10_000_000  # far above benchmark.py's reserved [1000, 1000+num_boards) range


def train(
    steps: int,
    lr: float,
    hidden_dim: int,
    min_components: int,
    max_components: int,
    seed_offset: int,
    checkpoint_out: Path,
    log_every: int = 10,
    checkpoint_every: int = 200,
    resume_from: Path | None = None,
) -> float:
    """Train for `steps` steps, optionally resuming model+optimizer state from a prior checkpoint.

    Returns the mean proxy loss over this call's steps, so a chunked driver (scripts/train_chunked.py)
    can report per-chunk progress without needing to parse stdout.
    """
    model = PlacementGNN(hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    normalizer = ProxyLossNormalizer()

    if resume_from is not None and resume_from.exists():
        state = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        if "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if "normalizer_state_dict" in state:
            normalizer.load_state_dict(state["normalizer_state_dict"])

    loss_sum = 0.0

    mlflow.set_experiment("quilter-placer-learned-training")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "steps": steps,
                "lr": lr,
                "hidden_dim": hidden_dim,
                "min_components": min_components,
                "max_components": max_components,
                "seed_offset": seed_offset,
                "resumed_from": str(resume_from) if resume_from is not None else "",
            }
        )

        for step in range(steps):
            seed = seed_offset + step
            rng = np.random.default_rng(seed)
            num_components = int(rng.integers(min_components, max_components + 1))

            board = generate_board(GeneratorConfig(num_components=num_components, seed=seed))
            graph = board_to_tensors(board)

            board_dims = torch.tensor([board.width, board.height], dtype=torch.float32)
            pred_pos_norm = model(graph)
            positions = pred_pos_norm * board_dims

            loss = proxy_cost(
                positions,
                graph.widths,
                graph.heights,
                graph.edge_comp_idx,
                graph.edge_net_idx,
                graph.net_weights,
                num_nets=graph.net_feats.shape[0],
                board_width=board.width,
                board_height=board.height,
                normalizer=normalizer,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

            if step % log_every == 0:
                ema = normalizer._ema
                ema_str = "  ".join(f"{k}_ema={v:.3g}" for k, v in ema.items())
                print(f"step {step:5d}  |V|={num_components:4d}  loss={loss.item():.4f}  {ema_str}")
                mlflow.log_metric("proxy_loss", loss.item(), step=seed_offset + step)
                for k, v in ema.items():
                    mlflow.log_metric(f"{k}_ema", v, step=seed_offset + step)

            if (step + 1) % checkpoint_every == 0 or step == steps - 1:
                checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "normalizer_state_dict": normalizer.state_dict(),
                        "hidden_dim": hidden_dim,
                    },
                    checkpoint_out,
                )

    print(f"\nSaved final checkpoint to {checkpoint_out}")
    return loss_sum / steps if steps > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--min-components", type=int, default=50)
    parser.add_argument("--max-components", type=int, default=300)
    parser.add_argument("--seed-offset", type=int, default=TRAIN_SEED_OFFSET)
    parser.add_argument("--checkpoint-out", type=Path, default=CHECKPOINT_DIR / "placement_gnn.pt")
    parser.add_argument("--resume-from", type=Path, default=None)
    args = parser.parse_args()

    train(
        steps=args.steps,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        min_components=args.min_components,
        max_components=args.max_components,
        seed_offset=args.seed_offset,
        checkpoint_out=args.checkpoint_out,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
