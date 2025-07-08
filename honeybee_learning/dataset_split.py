"""Functions for splitting the honeybee dataset into splits for training, validation,
and testing.
"""

from __future__ import annotations

import random
from collections import defaultdict

from .config import (
    DATASET_CREATE_SHUFFLE,
    DATASET_CREATE_SHUFFLE_SEED,
    MAX_FRAME_DIFFERENCE,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)

__all__ = ["split_pairs"]


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
