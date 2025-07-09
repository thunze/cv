"""Functions for precalculating the representations of honeybee images in the
honeybee dataset using a frozen self-supervised representation learning model.

This is useful for later evaluating the model's performance without having to
perform the forward pass through the model for each image again and again, which
would be very time-consuming.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import DEVICE, REPRESENTATIONS_PATH
from .dataset_test import get_single_dataloader

__all__ = ["precalculate_representations"]


# Batch size to use when precalculating representations of cropped honeybee images.
REPRESENTATION_PRECALCULATION_BATCH_SIZE = 64


def precalculate_representations(model_filepath: Path) -> None:
    """Using a frozen self-supervised representation learning model, precalculate
    and save representations of all cropped honeybee images in the honeybee dataset.

    This is done to speed up the evaluation of the self-supervised representation
    learning model.

    The representations are saved in a file named just like the model checkpoint,
    but ending with `_representations.npy` instead of `.pth`. The file will be saved
    in the configured `REPRESENTATIONS_PATH` directory.

    Args:
        model_filepath: Path to the self-supervised representation learning model
            checkpoint file.
    """
    # Make sure the representations directory exists
    REPRESENTATIONS_PATH.mkdir(parents=True, exist_ok=True)

    # Load the model checkpoint
    print(f"Loading model from {model_filepath!r}...")
    model = torch.load(model_filepath, map_location=DEVICE)

    # Get data loaders for both `train_and_validate` and `test` modes
    batch_size = REPRESENTATION_PRECALCULATION_BATCH_SIZE
    dataloader = get_single_dataloader(batch_size=batch_size)
    num_representations = len(dataloader.dataset)

    representations = np.empty(
        (num_representations, model.module.output_dim), dtype=np.float32
    )

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        print("\nCalculating representations...")
        for i, batch in enumerate(dataloader):
            print(f"\tProcessing batch {i + 1}/{len(dataloader)}")
            x, _, _, _ = batch
            x = x.to(DEVICE)  # Move data to the target device
            z = model(x)  # Get the representations
            start_index = i * batch_size
            end_index = start_index + x.shape[0]  # Actual batch size may vary
            representations[start_index:end_index] = z.cpu().numpy()

    # Save the representations to a file
    representations_filepath = (
        REPRESENTATIONS_PATH / f"{model_filepath.stem}_representations.npy"
    )
    print(f"\nSaving representations to {representations_filepath!r}...")
    np.save(representations_filepath, representations)
