"""Dataset and data loading functions."""

from __future__ import annotations

import glob
import os
import random
from collections import defaultdict
from typing import Literal, NamedTuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image
from torchvision.transforms.functional import resize

from .config import (
    CROPS_PATH,
    DATALOADER_NUM_WORKERS,
    DATASET_CREATE_SHUFFLE,
    DATASET_CREATE_SHUFFLE_SEED,
    MAX_FRAME_DIFFERENCE,
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
        self.samples = []
        self.mode = mode

        split_mapping = split_single()

        all_png_files = glob.glob(
            os.path.join(CROPS_PATH, "**", "*.png"), recursive=True
        )

        self.filepaths = [
            f
            for f in all_png_files
            if os.path.basename(f) in split_mapping
            and split_mapping[os.path.basename(f)] == mode
        ]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index: int) -> HoneybeeSample:
        """Get a `HoneybeeSample` from the dataset by index."""
        filepath = self.filepaths[index]
        filename = os.path.basename(filepath)
        info = parse_filename(filename)

        img = decode_image(filepath).float() / 255.0
        # img = torch.unsqueeze(img, 0)

        # Resize input image to 224x224, as expected by the ResNet backbone
        img = resize(img, (224, 224))

        # Convert grayscale to RGB, as the ResNet backbone expects 3-channel input
        img = img.repeat(3, 1, 1)

        if info["recording_no"] == "1":
            bee_id = int(info["bee_no"])
        else:
            bee_id = 361 + int(info["bee_no"])

        return HoneybeeSample(
            x=img,
            id_=bee_id,
            class_=int(info["class_no"]),
            angle=int(info["angle"]),
        )


class HoneybeeImagePairDataset(Dataset):
    """PyTorch `Dataset` implementation for pairs of cropped images of the same
    honeybee created from adjacent frames of the honeybee dataset video.

    Args:
        mode: The dataset split to load. Can be 'train', 'validate', or 'test'.
            This determines which predefined subset of the dataset to use.
    """

    def __init__(self, *, mode: Literal["train", "validate", "test"]):
        self.pairs = [p for p in split_pairs() if p["set"] == mode]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int) -> HoneybeeImagePair:
        """Get a `HoneybeeImagePair` from the dataset by index."""
        pair = self.pairs[index]
        info_img1 = parse_filename(pair["frame"])
        img1_path = os.path.join(
            CROPS_PATH, info_img1["recording_no"], info_img1["bee_no"], pair["frame"]
        )
        img1 = decode_image(img1_path).float() / 255.0

        info_img2 = parse_filename(pair["paired_frame"])
        img2_path = os.path.join(
            CROPS_PATH,
            info_img2["recording_no"],
            info_img2["bee_no"],
            pair["paired_frame"],
        )
        img2 = decode_image(img2_path).float() / 255.0

        # Resize input images to 224x224, as expected by the ResNet backbone
        img1 = resize(img1, (224, 224))
        img2 = resize(img2, (224, 224))

        # Convert grayscale to RGB, as the ResNet backbone expects 3-channel input
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


def split_pairs():
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

    # Get list of files

    all_png_files = glob.glob(os.path.join(CROPS_PATH, "**", "*.png"), recursive=True)

    files = [os.path.basename(f) for f in all_png_files]
    bee_to_files = defaultdict(list)

    for f in files:
        parts = f.replace(".png", "").split("_")
        rec = int(parts[0])
        bee = int(parts[2])
        bee_to_files[(rec, bee)].append(f)

    pairs = []

    for (rec, bee), frame_files in bee_to_files.items():
        # If the bee has only one frame available, skip because we want pairs
        if len(frame_files) == 1:
            continue
        frame_files.sort()
        for i in range(len(frame_files) - 1):
            filename1 = frame_files[i]
            filename2 = frame_files[i + 1]
            frame_no1 = int(parse_filename(filename1)["frame_no"])
            frame_no2 = int(parse_filename(filename2)["frame_no"])
            if frame_no2 - frame_no1 <= MAX_FRAME_DIFFERENCE:
                pairs.append({"frame": filename1, "paired_frame": filename2})

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


def split_single():
    """Create and return a train-validate-test split of single honeybee images from
    the honeybee dataset.
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

    all_png_files = glob.glob(os.path.join(CROPS_PATH, "**", "*.png"), recursive=True)

    files = [os.path.basename(f) for f in all_png_files]

    if DATASET_CREATE_SHUFFLE:
        random.seed(DATASET_CREATE_SHUFFLE_SEED)
        random.shuffle(files)

    no_crops = len(files)
    n_train = int(TRAIN_RATIO * no_crops)
    n_val = int(VALIDATION_RATIO * no_crops)

    split_map = {}
    for i, f in enumerate(files):
        if i < n_train:
            split_map[f] = "train"
        elif i < n_train + n_val:
            split_map[f] = "validate"
        else:
            split_map[f] = "test"

    return split_map


def parse_filename(filename: str):
    """Parse a filename of the format
    `recordingNo_frameNo_beeNo_posX_posY_classNo_angle.png` into a dict containing
    this information.

    Args:
        filename: The filename to parse.

    Returns:
        A dict containing the corresponding `recording_no`, `frame_no`, `bee_no`,
        `class_no` and `angle`.
    """
    parts = filename.replace(".png", "").split("_")
    return {
        "recording_no": parts[0],
        "frame_no": parts[1],
        "bee_no": parts[2],
        "class_no": parts[5],
        "angle": parts[6],
    }
