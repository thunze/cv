"""SimCLR model and training pipeline."""

from __future__ import annotations

import numpy as np
import torchvision
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.models.utils import get_weight_decay_parameters
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler
from torch import nn

from .config import DEVICE
from .dataset import get_dataloader
from .train import train

__all__ = ["SimCLR", "train_simclr"]


# Hyperparameters

## Basic training parameters
BATCH_SIZE = 2048  # Figure 9; small epoch size + large batch size = good performance
EPOCHS = 800  # Paper goes up to 800

## Parameters for validating the model on linear predictors
LINEAR_PREDICTORS_TRAIN_EPOCHS = 3  # Number of epochs for which to train predictors
# Learning rate to use for predictors; makes no difference when batch size = 4096
LINEAR_PREDICTORS_LEARNING_RATE = 0.075 * np.sqrt(BATCH_SIZE)

## Loss parameters
SIMCLR_LOSS_TEMPERATURE = 0.1  # Default: 0.1

## Optimizer parameters
_LARS_BASE_LEARNING_RATE = 0.3
LARS_LEARNING_RATE = BATCH_SIZE / 256 * _LARS_BASE_LEARNING_RATE  # Specified in paper
LARS_MOMENTUM = 0.9
LARS_WEIGHT_DECAY = 1e-6

## Learning rate scheduler parameters
LR_SCHEDULER_WARMUP_EPOCHS = 10

## Projection head configuration. See `SimCLRProjectionHead` for more details.
PROJECTION_HEAD_INPUT_DIM = 512
PROJECTION_HEAD_HIDDEN_DIM = 2048  # Input and hidden dim of same size
PROJECTION_HEAD_OUTPUT_DIM = 128
PROJECTION_HEAD_NUM_LAYERS = 3


# Hyperparameters to log for the run
ALL_HYPERPARAMETERS = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "simclr_loss_temperature": SIMCLR_LOSS_TEMPERATURE,
    "_lars_base_learning_rate": _LARS_BASE_LEARNING_RATE,
    "lars_learning_rate": LARS_LEARNING_RATE,
    "lars_momentum": LARS_MOMENTUM,
    "lars_weight_decay": LARS_WEIGHT_DECAY,
    "lr_scheduler_warmup_epochs": LR_SCHEDULER_WARMUP_EPOCHS,
    "projection_head_input_dim": PROJECTION_HEAD_INPUT_DIM,
    "projection_head_hidden_dim": PROJECTION_HEAD_HIDDEN_DIM,
    "projection_head_output_dim": PROJECTION_HEAD_OUTPUT_DIM,
    "projection_head_num_layers": PROJECTION_HEAD_NUM_LAYERS,
}


class SimCLR(nn.Module):
    """SimCLR model with ResNet backbone and projection head."""

    def __init__(self):
        super().__init__()

        # Remove the last fully connected layer to use ResNet as a backbone
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.projection_head = SimCLRProjectionHead(
            input_dim=PROJECTION_HEAD_INPUT_DIM,
            hidden_dim=PROJECTION_HEAD_HIDDEN_DIM,  # Input and hidden dim of same size
            output_dim=PROJECTION_HEAD_OUTPUT_DIM,
            num_layers=PROJECTION_HEAD_NUM_LAYERS,
        )
        self.output_dim = PROJECTION_HEAD_OUTPUT_DIM  # For convenience

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (..., channels, height, width), representing the
                input images.

        Returns:
            Output tensor of shape (..., feature dimensions), representing the
            projected features.
        """
        h = self.backbone(x).flatten(start_dim=1)  # Don't flatten across samples
        z = self.projection_head(h)
        return z


def train_simclr(*, log_to_wandb: bool = False) -> None:
    """Train the SimCLR model on the honeybee dataset.

    Args:
        log_to_wandb: Whether to log training progress to Weights & Biases (wandb).
    """

    # Prepare loading training and validation data
    train_pair_dataloader = get_dataloader(
        pairs=True, mode="train", batch_size=BATCH_SIZE
    )
    validate_pair_dataloader = get_dataloader(
        pairs=True, mode="validate", batch_size=BATCH_SIZE
    )

    # Prepare model
    model = SimCLR()
    model = nn.DataParallel(model)  # Enable data parallelism for multi-GPU training
    model = model.to(DEVICE)  # Move model to target device

    # Prepare loss function
    criterion = NTXentLoss(
        temperature=SIMCLR_LOSS_TEMPERATURE,
        gather_distributed=True,  # Use all negatives from all GPUs
    )

    # Prepare optimizer
    # For performance reasons, don't apply weight decay to norm and bias parameters.
    params_weight_decay, params_no_weight_decay = get_weight_decay_parameters(
        [model.module.backbone, model.module.projection_head]
    )
    optimizer = LARS(
        [
            {"name": "simclr_weight_decay", "params": params_weight_decay},
            {
                "name": "simclr_no_weight_decay",
                "params": params_no_weight_decay,
                "weight_decay": 0.0,
            },
        ],
        lr=LARS_LEARNING_RATE,
        momentum=LARS_MOMENTUM,
        weight_decay=LARS_WEIGHT_DECAY,
    )

    # Prepare learning rate scheduler
    warmup_iterations = LR_SCHEDULER_WARMUP_EPOCHS * len(train_pair_dataloader)
    total_iterations = EPOCHS * len(train_pair_dataloader)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=warmup_iterations,  # Number of warmup training steps (not epochs)
        max_epochs=total_iterations,  # Total number of training steps (not epochs)
    )

    # Train the model
    train(
        model=model,
        train_pair_dataloader=train_pair_dataloader,
        validate_pair_dataloader=validate_pair_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        all_hyperparameters=ALL_HYPERPARAMETERS,
        log_to_wandb=log_to_wandb,
    )
