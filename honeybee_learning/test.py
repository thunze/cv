"""Model testing functions.

Applied to only the final model to evaluate its performance.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

__all__ = ["test_something"]


def test_something(
    model: nn.Module,
    test_dataloader: DataLoader,
) -> None:
    """Placeholder test function."""
    assert not torch.is_grad_enabled()
