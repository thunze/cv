"""CLI entrypoints for the project."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from honeybee_learning import simclr, test_precalculate, vicreg
from honeybee_learning import test_linear as test_linear_
from honeybee_learning import test_visual as test_visual

__all__ = ["train"]


def train():
    """CLI for training a model on the honeybee dataset."""
    parser = ArgumentParser(
        description=(
            "Train a self-supervised representation learning model on the honeybee "
            "dataset."
        )
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


def precalculate_representations():
    """CLI for precalculating representations of honeybee images in the dataset."""
    parser = ArgumentParser(
        description=(
            "Precalculate and store representations of all honeybee images in the "
            "honeybee dataset using a checkpoint of a pretrained self-supervised "
            "representation learning model."
        )
    )
    parser.add_argument(
        "--model",
        choices=["simclr", "vicreg"],
        required=True,
        help="type of the model to use for precalculating representations",
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="path to the checkpoint file from which to load the pretrained model",
    )
    args = parser.parse_args()

    model_type = args.model
    checkpoint_path = Path(args.checkpoint)
    test_precalculate.precalculate_representations(model_type, checkpoint_path)


def test_linear():
    """CLI for training and testing linear predictors on top of a frozen
    self-supervised representation learning model.
    """
    parser = ArgumentParser(
        description=(
            "Train and test linear predictors on top of a frozen self-supervised "
            "representation learning model."
        )
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="whether to log the training and testing progress to Weights & Biases",
    )
    parser.add_argument(
        "representations",
        type=str,
        help=(
            "path to the file containing precalculated representations generated by "
            "the `precalculate-representations` command"
        ),
    )
    args = parser.parse_args()

    representations_path = Path(args.representations)
    test_linear_.train_and_test_linear_predictors(
        representations_path, log_to_wandb=args.wandb
    )


def visualize():
    parser = ArgumentParser(
        description=(
            "Visualize representations generated by a self-supervised representation "
            "learning model."
        )
    )
    parser.add_argument(
        "representations",
        type=str,
        help=(
            "path to the file containing precalculated representations generated by "
            "the `precalculate-representations` command"
        ),
    )
    args = parser.parse_args()

    representations_path = Path(args.representations)
    test_visual.evaluate(representations_path)
