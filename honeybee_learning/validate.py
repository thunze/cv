"""Model validation functions."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import DEVICE

__all__ = ["validate_epoch_validation_loss"]


def validate_epoch_validation_loss(
    model,
    validate_dataloader: DataLoader,
    criterion: nn.Module,
) -> float:
    """Validate `model` for one epoch on data provided by `validate_dataloader` using
    `criterion` to compute the validation loss.

    Args:
        model: Model to validate.
        validate_dataloader: `DataLoader` providing the validation dataset.
        criterion: Loss function to compute the validation loss.

    Returns:
        Average validation loss for the epoch.
    """
    assert not torch.is_grad_enabled()

    validation_loss_epoch = 0  # Aggregate validation loss for the epoch

    # One pass through the validation dataset
    for batch in validate_dataloader:
        # `x0` and `x1` are two views of the same honeybee.
        x0, x1 = batch[0]

        # Move data to target device
        x0 = x0.to(DEVICE)
        x1 = x1.to(DEVICE)

        # Forward pass
        z0 = model(x0)
        z1 = model(x1)

        # Compute validation loss
        batch_loss = criterion(z0, z1)
        validation_loss_epoch += batch_loss.detach()

    # Compute average validation loss for the epoch
    avg_validation_loss_epoch = validation_loss_epoch / len(validate_dataloader)
    return avg_validation_loss_epoch
