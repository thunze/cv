"""Dataset and data loading functions."""

from __future__ import annotations

from typing import Literal

from lightly.data import LightlyDataset
from torch.utils.data import DataLoader, Dataset

from .config import DATALOADER_NUM_WORKERS

__all__ = ["HoneybeeDataset", "get_dataset", "get_dataloader"]


class HoneybeeDataset(Dataset):  # Placeholder for now
    def __init__(self, *, mode: Literal["train", "validate", "test"]):
        super().__init__()


def get_dataset(*, mode: Literal["train", "validate", "test"]) -> LightlyDataset:
    """Get a `LightlyDataset` for the honeybee dataset.

    Args:
        mode: The dataset split to load. Can be 'train', 'validate', or 'test'.

    Returns:
        A `LightlyDataset` object for the specified dataset split.
    """
    torch_dataset = HoneybeeDataset(mode=mode)

    # Lightly models require a `LightlyDataset`
    return LightlyDataset.from_torch_dataset(torch_dataset)


def get_dataloader(
    *, mode: Literal["train", "validate", "test"], batch_size: int
) -> DataLoader:
    """Get a `DataLoader` for the honeybee dataset.

    Args:
        mode: The dataset split to load. Can be 'train', 'validate', or 'test'.
        batch_size: The batch size for the `DataLoader`.

    Returns:
        A `DataLoader` object for the specified dataset split.
    """
    dataset = get_dataset(mode=mode)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),  # Shuffle only for training
        drop_last=True,  # For stability, drop the last batch if it's < batch size
        num_workers=DATALOADER_NUM_WORKERS,
    )
