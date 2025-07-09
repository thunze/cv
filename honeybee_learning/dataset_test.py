"""Test dataset and data loading functions."""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch
from numpy import load
from torch.utils.data import DataLoader, Dataset

from .config import CROPS_PATH, DATALOADER_NUM_WORKERS, METADATA_PATH
from .dataset_split import split_pairs

__all__ = [
    "HoneybeeRepresentationSample",
    "HoneybeeDataset",
    "get_single_dataloader",
    "get_representation_dataloader",
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


class HoneybeeRepresentationSample(NamedTuple):
    """A vector representation of a sample from the honeybee dataset, plus its
    original metadata.

    Attributes:
        z: Vector representation of the cropped image of the honeybee; floating-point
            tensor of shape (representation_dim,).
        id_: Bee ID, a unique identifier for the honeybee in the dataset.
        class_: Bee class, either 0 (not within a comb cell) or 1 (within a combcell).
        angle: Bee orientation angle in degrees (0-360).
    """

    z: torch.Tensor
    id_: int
    class_: int
    angle: int


class HoneybeeDataset(Dataset):
    """PyTorch `Dataset` implementation for the honeybee dataset.

    This dataset provides access to individual honeybee samples, each represented
    by a `HoneybeeSample` named tuple containing the cropped image, bee ID, class,
    and orientation angle.
    """

    def __init__(self):
        # Load images and metadata from file
        self.images = load(CROPS_PATH)
        self.metadata = load(METADATA_PATH)

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


def get_single_dataloader(*, batch_size: int) -> DataLoader:
    """Get a `DataLoader` using `HoneybeeDataset` under the hood for the honeybee
    dataset.

    Args:
        batch_size: The batch size for the `DataLoader`.

    Returns:
        A `DataLoader` object for the specified dataset split.
    """
    dataset = HoneybeeDataset()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Do not shuffle for precalculation to maintain order
        drop_last=False,
        num_workers=DATALOADER_NUM_WORKERS,
    )


class HoneybeeRepresentationDataset(Dataset):
    """PyTorch `Dataset` implementation for the honeybee dataset.

    This dataset provides access to individual honeybee representation samples, each
    represented by a `HoneybeeRepresentationSample` named tuple containing the
    representation vector, bee ID, class, and orientation angle.

    Args:
        mode: The dataset split to load. Can be 'train_and_validate', or 'test'.
            In 'train_and_validate' mode, the dataset contains all representations
            that appear in the training and validation splits of the pair dataset.
            In 'test' mode, the dataset contains all representations that do not
            appear in the training and validation splits of the pair dataset.
    """

    def __init__(
        self, path_to_representations, *, mode: Literal["train_and_validate", "test"]
    ):
        # Load representation and metadata from file
        self.representations = load(path_to_representations)
        self.metadata = load(METADATA_PATH)

        self.mode = mode

        # Get all pairs that belong to training and validation split
        self.pairs_train = [
            p for p in split_pairs(self.metadata) if p["set"] == "train"
        ]
        self.pairs_val = [
            p for p in split_pairs(self.metadata) if p["set"] == "validate"
        ]

        # Collect all indices that appear in pairs_train and pairs_val
        all_pair_indices = {
            idx
            for p in self.pairs_train + self.pairs_val
            for idx in (p["first_index"], p["second_index"])
        }

        # If mode is train and validate, we use all representations that appear in the
        # training and validation splits of the pair dataset
        if mode == "train_and_validate":
            self.indices = sorted(all_pair_indices)
        # Otherwise we use all representations that do not appear in the train and
        # validation splits
        else:
            self.indices = [
                i for i in range(len(self.representations)) if i not in all_pair_indices
            ]

        # Get representations and metadata for those indices
        self.representations = [self.representations[i] for i in self.indices]
        self.metadata = [self.metadata[i] for i in self.indices]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int) -> HoneybeeRepresentationSample:
        """Get a HoneybeeRepresentationSample from the dataset by index."""
        representation = self.representations[index]
        metadata = self.metadata[index]

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

        return HoneybeeRepresentationSample(
            z=representation,
            id_=bee_id,
            class_=class_id,
            angle=angle,
        )


def get_representation_dataloader(
    path_to_representations,
    *,
    mode: Literal["train_and_validate", "test"],
    batch_size: int,
) -> DataLoader:
    """Get a DataLoader for all representations without splits or shuffling.

    Args:
        mode: The dataset split to load. Can be 'train_and_validate', or 'test'.
        batch_size: The batch size for the `DataLoader`.

    Returns:
        A `DataLoader` object for the specified dataset.
    """
    dataset = HoneybeeRepresentationDataset(path_to_representations, mode=mode)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode != "test"),  # Shuffle only if not testing
        drop_last=False,
        num_workers=DATALOADER_NUM_WORKERS,
    )
