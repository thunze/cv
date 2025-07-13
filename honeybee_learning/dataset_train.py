"""Dataset implementations and data loading functions for training self-supervised
representation learning models on the honeybee dataset.
"""

from __future__ import annotations

import random
from typing import Literal, NamedTuple

import torch
from numpy import load
from torch.utils.data import DataLoader, Dataset

from .config import (
    CROPS_PATH,
    DATALOADER_NUM_WORKERS,
    DATASET_CREATE_SHUFFLE_SEED,
    METADATA_PATH,
    RATIO_SAMPLE,
)
from .dataset_split import split_pairs

__all__ = [
    "HoneybeeImagePair",
    "HoneybeeImagePairDataset",
    "get_train_dataloader",
]


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


class HoneybeeImagePairDataset(Dataset):
    """PyTorch `Dataset` implementation for pairs of cropped images of the same
    honeybee created from adjacent frames of the honeybee dataset video.

    Args:
        mode: The dataset split to load. Can be 'train', 'validate', or 'test'.
            This determines which predefined subset of the dataset to use.
        transform: Optional transformation function to apply to the images.
    """

    def __init__(self, *, mode: Literal["train", "validate", "test"], transform=None):
        # Load images and metadata from file
        self.images = load(CROPS_PATH)
        self.metadata = load(METADATA_PATH)
        self.mode = mode
        self.transform = transform

        # Build set of pairs to use for selected mode and seed
        all_pairs = [p for p in split_pairs(self.metadata) if p["set"] == mode]

        # Sample a fraction of the pairs randomly if 0 < RATIO_SAMPLE < 1
        if RATIO_SAMPLE <= 0 or RATIO_SAMPLE >= 1:
            self.pairs = all_pairs
        else:
            total = len(all_pairs)
            sample_size = int(RATIO_SAMPLE * total)

            rng = random.Random(DATASET_CREATE_SHUFFLE_SEED)
            self.pairs = rng.sample(all_pairs, sample_size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int) -> HoneybeeImagePair:
        """Get a `HoneybeeImagePair` from the dataset by index."""

        # Get a pair and their indices for loading them
        pair = self.pairs[index]
        index_img1 = pair["first_index"]
        index_img2 = pair["second_index"]

        img1 = self.images[index_img1]
        img2 = self.images[index_img2]

        # Convert to tensor and float, then divide by 255.0
        img1 = torch.from_numpy(img1).float() / 255.0
        img2 = torch.from_numpy(img2).float() / 255.0

        # Convert to RGB
        img1 = img1.repeat(3, 1, 1)
        img2 = img2.repeat(3, 1, 1)

        # Apply transformations if specified
        if self.transform is not None:
            img1, img2 = self.transform((img1, img2))

        return HoneybeeImagePair(x1=img1, x2=img2)


def get_train_dataloader(
    *, mode: Literal["train", "validate", "test"], batch_size: int, transform=None
) -> DataLoader:
    """Get a `DataLoader` using `HoneybeeImagePairDataset` under the hood for the
    honeybee dataset.

    Args:
        mode: The dataset split to load. Can be 'train', 'validate', or 'test'.
        batch_size: The batch size for the `DataLoader`.
        transform: Optional transformation function to apply to images in the dataset.

    Returns:
        A `DataLoader` object for the specified dataset split.
    """
    dataset = HoneybeeImagePairDataset(mode=mode, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),  # Shuffle only for training
        drop_last=True,  # For stability, drop the last batch if it's < batch size
        num_workers=DATALOADER_NUM_WORKERS,
    )
