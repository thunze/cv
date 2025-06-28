"""Dataset and data loading functions."""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch
from lightly.data import LightlyDataset
from torch.utils.data import DataLoader, Dataset

from .config import DATALOADER_NUM_WORKERS

__all__ = [
    "HoneybeeSample",
    "HoneybeeImagePair",
    "HoneybeeDataset",
    "HoneybeeImagePairDataset",
    "get_dataset",
    "get_dataloader",
]


class HoneybeeSample(NamedTuple):
    """A sample from the honeybee dataset.

    Attributes:
        x: Cropped image of the honeybee at the center of the shot at a certain
            frame; floating-point tensor of shape (channels, height, width).
        id_: Bee ID, a unique identifier for the honeybee in the dataset.
        class_: Bee class, either 0 (not within a comb cell) or 1 (within a combcell).
        angle: Bee orientation angle in degrees (0-360).
    """

    x: torch.Tensor
    id_: int
    class_: int
    angle: int


class HoneybeeImagePair(NamedTuple):
    """A pair of cropped images of the same honeybee created from adjecent frames of
    the honeybee dataset video.

    Attributes:
        x1: First cropped image of the honeybee at the center of the shot at a
            certain frame; floating-point tensor of shape (channels, height, width).
        x2: Second cropped image of the honeybee at the center of the shot at an
            adjacent frame; floating-point tensor of shape (channels, height, width).
    """

    x1: torch.Tensor
    x2: torch.Tensor


class HoneybeeDataset(Dataset):
    """PyTorch `Dataset` implementation for the honeybee dataset.

    This dataset provides access to individual honeybee samples, each represented
    by a `HoneybeeSample` named tuple containing the cropped image, bee ID, class,
    and orientation angle.

    Args:
        mode: The dataset split to load. Can be 'train', 'validate', or 'test'.
            This determines which predefined subset of the dataset to use.
    """

    def __init__(self, *, mode: Literal["train", "validate", "test"]):
        super().__init__()  # Placeholder for now

    def __getitem__(self, index: int) -> HoneybeeSample:
        """Get a `HoneybeeSample` from the dataset by index."""


class HoneybeeImagePairDataset(Dataset):
    """PyTorch `Dataset` implementation for pairs of cropped images of the same
    honeybee created from adjacent frames of the honeybee dataset video.

    Args:
        mode: The dataset split to load. Can be 'train', 'validate', or 'test'.
            This determines which predefined subset of the dataset to use.
    """

    def __init__(self, *, mode: Literal["train", "validate", "test"]):
        super().__init__()  # Placeholder for now

    def __getitem__(self, index: int) -> HoneybeeImagePair:
        """Get a `HoneybeeImagePair` from the dataset by index."""


def get_dataset(
    *, pairs: bool, mode: Literal["train", "validate", "test"]
) -> LightlyDataset:
    """Get a `LightlyDataset` wrapper for the honeybee dataset.

    Args:
        pairs: Whether to return a dataset of image pairs instead of individual
            honeybee samples. If `True`, returns a wrapper for a new
            `HoneybeeImagePairDataset` instance; if `False`, returns a wrapper for a
            new `HoneybeeDataset` instance.
        mode: The dataset split to load. Can be 'train', 'validate', or 'test'.

    Returns:
        A `LightlyDataset` object for the specified dataset split.
    """
    if pairs:
        torch_dataset = HoneybeeImagePairDataset(mode=mode)
    else:
        torch_dataset = HoneybeeDataset(mode=mode)

    # Lightly models require a `LightlyDataset`
    return LightlyDataset.from_torch_dataset(torch_dataset)


def get_dataloader(
    *, pairs: bool, mode: Literal["train", "validate", "test"], batch_size: int
) -> DataLoader:
    """Get a `DataLoader` for the honeybee dataset.

    Args:
        pairs: Whether to return a dataloader for a dataset of image pairs instead of
            individual honeybee samples. If `True`, returns a dataloader for a new
            `HoneybeeImagePairDataset` instance; if `False`, returns a dataloader for
            a new `HoneybeeDataset` instance.
        mode: The dataset split to load. Can be 'train', 'validate', or 'test'.
        batch_size: The batch size for the `DataLoader`.

    Returns:
        A `DataLoader` object for the specified dataset split.
    """
    dataset = get_dataset(pairs=pairs, mode=mode)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),  # Shuffle only for training
        drop_last=True,  # For stability, drop the last batch if it's < batch size
        num_workers=DATALOADER_NUM_WORKERS,
    )
