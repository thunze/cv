"""Model testing functions.

Applied to only the final model to evaluate its performance.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

__all__ = ["test_something"]


# Hyperparameters for training linear predictors
LINEAR_PREDICTORS_TRAIN_EPOCHS = 10  # Number of epochs for which to train predictors
LINEAR_PREDICTORS_LEARNING_RATE = 1e-3  # Learning rate to use for predictors


def test_something(
    model: nn.Module,
    test_dataloader: DataLoader,
) -> None:
    """Placeholder test function."""
    assert not torch.is_grad_enabled()
