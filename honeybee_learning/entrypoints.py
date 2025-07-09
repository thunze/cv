"""CLI entrypoints for the project."""

from __future__ import annotations

from argparse import ArgumentParser

from honeybee_learning import simclr, vicreg

__all__ = ["train"]


def train():
    """CLI for running deep learning experiments on the honeybee dataset."""
    parser = ArgumentParser(
        description="Run deep learning experiments on the honeybee dataset."
    )
    parser.add_argument(
        "--model",
        choices=["simclr", "vicreg"],
        required=True,
        help="model to train",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="whether to log the training run to Weights & Biases",
    )
    args = parser.parse_args()

    if args.model == "simclr":
        simclr.train_simclr(log_to_wandb=args.wandb)
    else:
        vicreg.train_vicreg(log_to_wandb=args.wandb)
