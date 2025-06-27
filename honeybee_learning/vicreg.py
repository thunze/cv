"""VICReg model and training pipeline."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import torchvision
import wandb
from lightly.data import LightlyDataset
from lightly.loss import VICRegLoss
from lightly.models.modules.heads import VICRegProjectionHead
from lightly.utils.lars import LARS
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

__all__ = ["train_vicreg"]


# Paths
CHECKPOINTS_PATH = Path("./checkpoints")  # Path to save checkpoints to


# Device configuration
DEVICE = "cuda"  # We expect a GPU to be available.
DATALOADER_NUM_WORKERS = 8  # Number of workers for the `DataLoader`


# Hyperparameters

## Basic training parameters
BATCH_SIZE = 2048  # Lightly example: 256
EPOCHS = 1000  # Lightly example: 10

## Loss parameters
VICREG_LOSS_LAMBDA = 25  # Variance loss weight
VICREG_LOSS_MU = 25  # Covariance loss weight
VICREG_LOSS_NU = 1  # Invariance loss weight

## Optimizer parameters
_LARS_BASE_LEARNING_RATE = 0.2
LARS_LEARNING_RATE = BATCH_SIZE / 256 * _LARS_BASE_LEARNING_RATE
LARS_WEIGHT_DECAY = 1e-6

## Learning rate scheduler parameters
LR_SCHEDULER_WARMUP_EPOCHS = 10  # Number of warmup epochs
LR_SCHEDULER_WARMUP_START_FACTOR = 0.1  # Start factor for learning rate during warmup
LR_SCHEDULE_COSINE_ETA_MIN = 2e-3  # Minimum learning rate for cosine scheduler

## Projection head configuration. See `VICRegProjectionHead` for more details.
PROJECTION_HEAD_INPUT_DIM = 2048
PROJECTION_HEAD_HIDDEN_DIM = 8192
PROJECTION_HEAD_OUTPUT_DIM = 8192
PROJECTION_HEAD_NUM_LAYERS = 3


# Hyperparameters to log for the run
ALL_HYPERPARAMETERS = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "vicreg_loss_lambda": VICREG_LOSS_LAMBDA,
    "vicreg_loss_mu": VICREG_LOSS_MU,
    "vicreg_loss_nu": VICREG_LOSS_NU,
    "_lars_base_learning_rate": _LARS_BASE_LEARNING_RATE,
    "lars_learning_rate": LARS_LEARNING_RATE,
    "lars_weight_decay": LARS_WEIGHT_DECAY,
    "lr_scheduler_warmup_epochs": LR_SCHEDULER_WARMUP_EPOCHS,
    "lr_scheduler_warmup_start_factor": LR_SCHEDULER_WARMUP_START_FACTOR,
    "lr_schedule_cosine_eta_min": LR_SCHEDULE_COSINE_ETA_MIN,
    "projection_head_input_dim": PROJECTION_HEAD_INPUT_DIM,
    "projection_head_hidden_dim": PROJECTION_HEAD_HIDDEN_DIM,
    "projection_head_output_dim": PROJECTION_HEAD_OUTPUT_DIM,
    "projection_head_num_layers": PROJECTION_HEAD_NUM_LAYERS,
}


# Weights & Biases (wandb) configuration
WANDB_ENTITY = "thunze"  # Team name
WANDB_PROJECT = "honeybee-learning"  # Project name
WANDB_CONFIG = ALL_HYPERPARAMETERS


class HoneybeeDataset(Dataset):  # Placeholder for now
    def __init__(self, *, mode: Literal["train", "validate", "test"]):
        super().__init__()


class VICReg(nn.Module):
    def __init__(self):
        super().__init__()

        # Resize input images to 224x224, as expected by the ResNet backbone
        self.resize = torchvision.transforms.Resize((224, 224))

        # Remove the last fully connected layer to use ResNet as a backbone
        resnet = torchvision.models.resnet50()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.projection_head = VICRegProjectionHead(
            input_dim=PROJECTION_HEAD_INPUT_DIM,
            hidden_dim=PROJECTION_HEAD_HIDDEN_DIM,
            output_dim=PROJECTION_HEAD_OUTPUT_DIM,
            num_layers=PROJECTION_HEAD_NUM_LAYERS,
        )

    def forward(self, x):
        x_r = self.resize(x)
        h = self.backbone(x_r).flatten(start_dim=1)  # Don't flatten across batches
        z = self.projection_head(h)
        return z


def load_dataset(*, mode: Literal["train", "validate", "test"]) -> LightlyDataset:
    torch_dataset = HoneybeeDataset(mode=mode)

    # Lightly models require a `LightlyDataset`
    lightly_dataset = LightlyDataset.from_torch_dataset(torch_dataset)

    return lightly_dataset


def train_vicreg(*, log_to_wandb: bool = False) -> None:
    # Initialize wandb run if enabled
    if log_to_wandb:
        wandb_run = wandb.init(
            entity=WANDB_ENTITY, project=WANDB_PROJECT, config=WANDB_CONFIG
        )
    else:
        wandb_run = None

    # Load training and validation data
    train_dataset = load_dataset(mode="train")
    validate_dataset = load_dataset(mode="validate")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,  # For stability, drop the last batch if it's < batch size
        num_workers=DATALOADER_NUM_WORKERS,
    )
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,  # For stability, drop the last batch if it's < batch size
        num_workers=DATALOADER_NUM_WORKERS,
    )

    # Prepare model
    model = VICReg()
    model = model.to(DEVICE)  # Move model to target device

    # Prepare loss and optimizer
    criterion = VICRegLoss(
        lambda_param=VICREG_LOSS_LAMBDA,
        mu_param=VICREG_LOSS_MU,
        nu_param=VICREG_LOSS_NU,
    )
    optimizer = LARS(
        model.parameters(),
        lr=LARS_LEARNING_RATE,
        weight_decay=LARS_WEIGHT_DECAY,
    )

    # Prepare learning rate scheduler
    # First linear warmup, then switch to cosine annealing
    warmup_iterations = LR_SCHEDULER_WARMUP_EPOCHS * len(train_dataloader)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=LR_SCHEDULER_WARMUP_START_FACTOR,
        end_factor=1.0,
        total_iters=warmup_iterations,
    )
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=0,  # We delay the scheduler separately using `SequentialLR`
        eta_min=LR_SCHEDULE_COSINE_ETA_MIN,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_iterations],  # After warmup, switch to cosine annealing
    )

    # Train the model
    print("Starting VICReg training...")
    model.train()  # Set the model to training mode
    model.zero_grad()  # Zero the gradients before training, just to be safe

    # Iterate over epochs
    for epoch in range(EPOCHS):
        training_loss_epoch = 0  # Aggregate training loss for the epoch

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

        model.eval()  # Set the model to evaluation mode

        # Validate the model after one epoch of training
        with torch.no_grad():
            validation_loss_epoch = 0  # Aggregate validation loss for the epoch

            # One pass through the validation dataset
            for batch in validate_dataloader:
                # `x0` and `x1` are two views of the same honeybee.
                x0, x1 = batch[0]  # TODO: See above comment about dataset

                # Move data to target device
                x0 = x0.to(DEVICE)
                x1 = x1.to(DEVICE)

                # Forward pass
                z0 = model(x0)
                z1 = model(x1)

                # Compute validation loss
                batch_loss = criterion(z0, z1)
                validation_loss_epoch += batch_loss.detach()

        model.train()  # Set the model back to training mode

        # Compute average losses for the epoch
        avg_training_loss_epoch = training_loss_epoch / len(train_dataloader)
        avg_validation_loss_epoch = validation_loss_epoch / len(validate_dataloader)

        # Log to standard output
        print(
            f"epoch: {epoch:>02}, "
            f"training loss: {avg_training_loss_epoch:.5f}, "
            f"validation loss: {avg_validation_loss_epoch:.5f}"
        )

        # Log to wandb if enabled
        if wandb_run is not None:
            wandb.log(
                {
                    "train/epoch": epoch,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/loss": avg_training_loss_epoch,
                    "validate/loss": avg_validation_loss_epoch,
                }
            )

    # Prepare saving model and hyperparameters
    model_filename_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename_base = f"vicreg_{model_filename_suffix}"
    CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)

    # Save trained model and serialized model hyperparameters
    torch.save(model.state_dict(), CHECKPOINTS_PATH / f"{model_filename_base}.pth")
    (CHECKPOINTS_PATH / f"{model_filename_base}.json").write_text(
        json.dumps(ALL_HYPERPARAMETERS, indent=4, ensure_ascii=False)
    )

    # Finish wandb run if enabled
    if wandb_run is not None:
        wandb_run.finish()
