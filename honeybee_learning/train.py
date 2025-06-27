"""Model training functions."""

from __future__ import annotations

import json
from datetime import datetime

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader

from .config import CHECKPOINTS_PATH, DEVICE, WANDB_ENTITY, WANDB_PROJECT
from .validate import validate_epoch_validation_loss

__all__ = ["train"]


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    validate_dataloader: DataLoader,
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
        train_dataloader: `DataLoader` for the training dataset.
        validate_dataloader: `DataLoader` for the validation dataset.
        criterion: Loss function to use for training.
        optimizer: Optimizer to use for training.
        scheduler: Learning rate scheduler to use during training.
        epochs: Number of epochs to train for.
        all_hyperparameters: Dictionary of all hyperparameters to log.
        log_to_wandb: Whether to log training progress to Weights & Biases (wandb).
    """
    # Prepare logging for the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"vicreg_{timestamp}"
    print(f"Starting training run {run_name!r}...\n")

    # Initialize wandb run if enabled
    if log_to_wandb:
        wandb_run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=run_name,
            config=all_hyperparameters,
        )
    else:
        wandb_run = None

    # Train the model
    model.train()  # Set the model to training mode
    model.zero_grad()  # Zero the gradients before training, just to be safe

    # Iterate over epochs
    for epoch in range(epochs):
        training_loss_epoch = 0  # Aggregate training loss for the epoch

        # --- Training ---

        # Train for one epoch
        # One pass through the training dataset
        for batch in train_dataloader:
            # `x0` and `x1` are two views of the same honeybee.
            x0, x1 = batch[0]  # TODO: This may need to be adjusted based on the dataset

            # Move data to target device
            x0 = x0.to(DEVICE)
            x1 = x1.to(DEVICE)

            # Forward pass
            z0 = model(x0)
            z1 = model(x1)

            # Compute training loss
            batch_loss = criterion(z0, z1)
            training_loss_epoch += batch_loss.detach()

            # Backpropagation and optimization
            batch_loss.backward()
            optimizer.step()  # Update model parameters
            optimizer.zero_grad()  # Zero the gradients for the next iteration
            scheduler.step()  # Update learning rate

        # Compute average training loss for the epoch
        avg_training_loss_epoch = training_loss_epoch / len(train_dataloader)

        # --- Validation ---

        model.eval()  # Set the model to evaluation mode

        # Validate the model after one epoch of training
        with torch.no_grad():
            avg_validation_loss_epoch = validate_epoch_validation_loss(
                model, validate_dataloader, criterion
            )

        model.train()  # Set the model back to training mode

        # --- Logging ---

        # Log to standard output
        print(
            f"epoch: {epoch:>02}, "
            f"training loss: {avg_training_loss_epoch:.5f}, "
            f"validation loss: {avg_validation_loss_epoch:.5f}"
        )

        # Log to wandb if enabled
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/epoch": epoch,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/loss": avg_training_loss_epoch,
                    "validate/loss": avg_validation_loss_epoch,
                }
            )

    # --- Finalization ---

    # Prepare saving model and hyperparameters
    CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)

    # Save trained model and serialized model hyperparameters
    torch.save(model.state_dict(), CHECKPOINTS_PATH / f"{run_name}.pth")
    (CHECKPOINTS_PATH / f"{run_name}.json").write_text(
        json.dumps(all_hyperparameters, indent=4, ensure_ascii=False)
    )

    # Finish wandb run if enabled
    if wandb_run is not None:
        wandb_run.finish()
