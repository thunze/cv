"""Dataset and data loading functions."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Literal, NamedTuple

import numpy as np
import torch
from numpy import load
from torch.utils.data import DataLoader, Dataset

from .config import (
    CROPS_PATH,
    DATALOADER_NUM_WORKERS,
    DATASET_CREATE_SHUFFLE,
    DATASET_CREATE_SHUFFLE_SEED,
    MAX_FRAME_DIFFERENCE,
    METADATA_PATH,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)

__all__ = [
    "HoneybeeSample",
    "HoneybeeImagePair",
    "HoneybeeDataset",
    "HoneybeeImagePairDataset",
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
        # Load images and metadata from file
        self.images = load(CROPS_PATH)
        self.metadata = load(METADATA_PATH)
        self.mode = mode

        np.random.seed(DATASET_CREATE_SHUFFLE_SEED)

        # Create a permutation of indices
        indices = np.arange(self.images.shape[0])
        np.random.seed(42)
        np.random.shuffle(indices)

        # Shuffle both arrays using the same indices
        self.images = self.images[indices]
        self.metadata = self.metadata[indices]

        # Get bounds for modes
        num = self.images.shape[0]
        n_train = int(TRAIN_RATIO * num)
        n_val = int(VALIDATION_RATIO * num)

        # Sample images for the correct mode
        if mode == "train":
            self.images = self.images[:n_train]
            self.metadata = self.metadata[:n_train]
        elif mode == "validate":
            self.images = self.images[n_train : n_train + n_val]
            self.metadata = self.metadata[n_train : n_train + n_val]
        else:
            self.images = self.images[n_train + n_val :]
            self.metadata = self.metadata[n_train + n_val :]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int) -> HoneybeeSample:
        """Get a `HoneybeeSample` from the dataset by index."""

        img = self.images[index]
        metadata = self.metadata[index]

        # Image operations: Convert to tensor and float, divide by 255.0.
        # Then convert to RGB.
        img = torch.from_numpy(img).float() / 255.0
        img = img.repeat(3, 1, 1)
        print(img.shape)

        # Get metadata
        rec_no = int(metadata[0])
        bee_no = int(metadata[2])
        class_id = int(metadata[3])
        angle = int(metadata[4])

        # Adjust bee number to be a continuous number instead of per recording
        if rec_no == 1:
            bee_id = bee_no
        else:
            bee_id = 361 + bee_no

        return HoneybeeSample(
            x=img,
            id_=bee_id,
            class_=class_id,
            angle=angle,
        )


class HoneybeeImagePairDataset(Dataset):
    """PyTorch `Dataset` implementation for pairs of cropped images of the same
    honeybee created from adjacent frames of the honeybee dataset video.

    Args:
        mode: The dataset split to load. Can be 'train', 'validate', or 'test'.
            This determines which predefined subset of the dataset to use.
    """

    def __init__(self, *, mode: Literal["train", "validate", "test"]):
        # Load images and metadata from file
        self.images = load(CROPS_PATH)
        self.metadata = load(METADATA_PATH)

        # Build set of pairs to use for selected mode and seed
        self.pairs = [p for p in split_pairs(self.metadata) if p["set"] == mode]

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

        return HoneybeeImagePair(x1=img1, x2=img2)


def get_dataloader(
    *, pairs: bool, mode: Literal["train", "validate", "test"], batch_size: int
) -> DataLoader:
    """Get a `DataLoader` for the honeybee dataset.

    Uses a `HoneybeeImagePairDataset` under the hood if `pairs` is `True` or a
    `HoneybeeDataset` if `pairs` is `False`.

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
    if pairs:
        dataset = HoneybeeImagePairDataset(mode=mode)
    else:
        dataset = HoneybeeDataset(mode=mode)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),  # Shuffle only for training
        drop_last=True,  # For stability, drop the last batch if it's < batch size
        num_workers=DATALOADER_NUM_WORKERS,
    )


def split_pairs(metadata_array):
    """Create and return a train-validate-test split of pairs of temporally adjacent
    honeybee images from the honeybee dataset.
    """
    # Validate input ratios
    if not (0 < TRAIN_RATIO < 1):
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    if not (0 <= VALIDATION_RATIO < 1):
        raise ValueError("val_ratio must be between 0 (inclusive) and 1 (exclusive).")

    if TRAIN_RATIO + VALIDATION_RATIO >= 1.0:
        raise ValueError(
            "The sum of train_ratio and val_ratio must be less than 1.0 to leave "
            "room for a test split."
        )

    # Build look up dictionary to find next frames fast
    index = defaultdict(list)
    for idx, entry in enumerate(metadata_array):
        # Key = (rec_no, bee_no)
        key = (entry[0], entry[2])
        # value = (index in metadata, entry)
        index[key].append((idx, entry))

    # Sort dict by frame_no
    for key in index:
        index[key].sort(key=lambda x: x[0])

    # Compute pairs by checking frame difference between temporally adjacent images
    pairs = []
    for _, frames in index.items():
        # If the bee has only one frame available, skip because we want pairs
        if len(frames) == 1:
            continue

        for i in range(len(frames) - 1):
            # Get indices and frame numbers for the current pair
            idx1, first_frame = frames[i]
            idx2, second_frame = frames[i + 1]
            first_frame_no = first_frame[1]
            second_frame_no = second_frame[1]

            if second_frame_no - first_frame_no <= MAX_FRAME_DIFFERENCE:
                pairs.append(
                    {
                        "frame": first_frame,
                        "paired_frame": second_frame,
                        "first_index": idx1,
                        "second_index": idx2,
                    }
                )

    # Shuffle if necessary
    if DATASET_CREATE_SHUFFLE:
        random.seed(DATASET_CREATE_SHUFFLE_SEED)
        random.shuffle(pairs)

    # Calculate boundaries
    no_pairs = len(pairs)
    n_train = int(TRAIN_RATIO * no_pairs)
    n_val = int(VALIDATION_RATIO * no_pairs)

    # Split pairs and assign sets
    for i, pair in enumerate(pairs):
        if i < n_train:
            pair["set"] = "train"
        elif i < n_train + n_val:
            pair["set"] = "validate"
        else:
            pair["set"] = "test"

    return pairs
