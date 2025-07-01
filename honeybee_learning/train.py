"""Model training functions."""

from __future__ import annotations

import json
from datetime import datetime

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader

from .config import (
    CHECKPOINT_EVERY_N_EPOCHS,
    CHECKPOINTS_PATH,
    DEVICE,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from .dataset import HoneybeeImagePair
from .validate import validate_epoch_validation_loss

__all__ = ["train"]


def train(
    model: nn.DataParallel,
    train_pair_dataloader: DataLoader,
    validate_pair_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epochs: int,
    all_hyperparameters: dict[str, float | int | str],
    *,
    log_to_wandb: bool = False,
) -> None:
    """Shared training algorithm implementation.

    Args:
        model: The model to train.
        train_pair_dataloader: `DataLoader` providing the training
            `HoneybeeImagePairDataset`.
        validate_pair_dataloader: `DataLoader` providing the validation
            `HoneybeeImagePairDataset`.
        criterion: Loss function to use for training.
        optimizer: Optimizer to use for training.
        scheduler: Learning rate scheduler to use during training.
        epochs: Number of epochs to train `model` for.
        all_hyperparameters: Dictionary of all hyperparameters to log.
        log_to_wandb: Whether to log training progress to Weights & Biases (wandb).
    """
    # Prepare logging for the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model.module.__class__.__name__.lower()}_{timestamp}"
    print(f"Starting training run {run_name!r}...")

    # Initialize wandb run if enabled
    if log_to_wandb:
        wandb_run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            anonymous="must",  # Force anonymous login, needed to create anonymous key
            name=run_name,
            config=all_hyperparameters,
        )
    else:
        wandb_run = None

    # Save hyperparameters for the run
    CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)
    (CHECKPOINTS_PATH / f"{run_name}.json").write_text(
        json.dumps(all_hyperparameters, indent=4, ensure_ascii=False)
    )

    # Train the model
    model.train()  # Set the model to training mode
    model.zero_grad()  # Zero the gradients before training, just to be safe

    # Iterate over epochs
    for epoch in range(epochs):
        training_loss_epoch = 0  # Aggregate training loss for the epoch

        # --- Training ---
        print(f"\nEpoch {epoch + 1}/{epochs}: Training...")

        batch: HoneybeeImagePair

        # Train for one epoch
        # One pass through the training dataset
        for i, batch in enumerate(train_pair_dataloader):
            print(f"\tTraining on batch {i + 1}/{len(train_pair_dataloader)}...")

            # `x0` and `x1` are two views of the same honeybee.
            x0, x1 = batch

            # Move data to target device
            x0 = x0.to(DEVICE)
            x1 = x1.to(DEVICE)

            # Forward pass
            z0 = model(x0)
            z1 = model(x1)

            # Compute training loss
            batch_loss = criterion(z0, z1)
            training_loss_epoch += batch_loss.item()

            # Backpropagation and optimization
            batch_loss.backward()
            optimizer.step()  # Update model parameters
            optimizer.zero_grad()  # Zero the gradients for the next iteration
            scheduler.step()  # Update learning rate

        # Compute average training loss for the epoch
        avg_training_loss_epoch = training_loss_epoch / len(train_pair_dataloader)

        # --- Validation ---
        print(f"\nEpoch {epoch + 1}/{epochs}: Validating...")

        model.eval()  # Set the model to evaluation mode

        # Validate the model after one epoch of training
        with torch.no_grad():
            avg_validation_loss_epoch = validate_epoch_validation_loss(
                model, validate_pair_dataloader, criterion
            )

        model.train()  # Set the model back to training mode

        # --- Tracking ---

        print(f"\nEpoch {epoch + 1}/{epochs}: Tracking...")

        # Save model state after every Nth epoch and after the last epoch
        if (epoch + 1) % CHECKPOINT_EVERY_N_EPOCHS == 0 or epoch == epochs - 1:
            print(f"\tSaving model checkpoint for epoch {epoch + 1}...")
            torch.save(
                model.state_dict(),
                CHECKPOINTS_PATH / f"{run_name}_epoch_{epoch + 1}.pth",
            )

        # Gather data to log
        log_data = {
            "train/epoch": epoch,
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            "train/loss": avg_training_loss_epoch,
            "validate/loss": avg_validation_loss_epoch,
        }

        # Log to standard output
        for key, value in log_data.items():
            if isinstance(value, float):
                print(f"\t{key}: {value:.5f}")
            else:
                print(f"\t{key}: {value}")

        # Log to wandb if enabled
        if wandb_run is not None:
            wandb_run.log(log_data)

    # --- Finalization ---

    # Finish wandb run if enabled
    if wandb_run is not None:
        wandb_run.finish()
