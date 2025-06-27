"""VICReg model and training pipeline."""

from __future__ import annotations

import torchvision
from lightly.loss import VICRegLoss
from lightly.models.modules.heads import VICRegProjectionHead
from lightly.models.utils import get_weight_decay_parameters
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler
from torch import nn

from .config import DEVICE
from .dataset import get_dataloader
from .train import train

__all__ = ["train_vicreg"]


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
LARS_MOMENTUM = 0.9
LARS_WEIGHT_DECAY = 1e-6

## Learning rate scheduler parameters
LR_SCHEDULER_WARMUP_EPOCHS = 10  # Number of warmup epochs
LR_SCHEDULE_COSINE_MIN = 1e-2  # Minimum cosine LR factor, scales from 0.002 to 0.2

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
    "lars_momentum": LARS_MOMENTUM,
    "lars_weight_decay": LARS_WEIGHT_DECAY,
    "lr_scheduler_warmup_epochs": LR_SCHEDULER_WARMUP_EPOCHS,
    "lr_schedule_cosine_min": LR_SCHEDULE_COSINE_MIN,
    "projection_head_input_dim": PROJECTION_HEAD_INPUT_DIM,
    "projection_head_hidden_dim": PROJECTION_HEAD_HIDDEN_DIM,
    "projection_head_output_dim": PROJECTION_HEAD_OUTPUT_DIM,
    "projection_head_num_layers": PROJECTION_HEAD_NUM_LAYERS,
}


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


def train_vicreg(*, log_to_wandb: bool = False) -> None:
    # Load training and validation data
    train_dataloader = get_dataloader(mode="train", batch_size=BATCH_SIZE)
    validate_dataloader = get_dataloader(mode="validate", batch_size=BATCH_SIZE)

    # Prepare model
    model = VICReg()
    model = model.to(DEVICE)  # Move model to target device

    # Prepare loss function
    criterion = VICRegLoss(
        lambda_param=VICREG_LOSS_LAMBDA,
        mu_param=VICREG_LOSS_MU,
        nu_param=VICREG_LOSS_NU,
    )

    # Prepare optimizer
    # For performance reasons, don't apply weight decay to norm and bias parameters.
    params_weight_decay, params_no_weight_decay = get_weight_decay_parameters(
        [model.backbone, model.projection_head]
    )
    optimizer = LARS(
        [
            {"name": "vicreg_weight_decay", "params": params_weight_decay},
            {
                "name": "vicreg_no_weight_decay",
                "params": params_no_weight_decay,
                "weight_decay": 0.0,
            },
        ],
        lr=LARS_LEARNING_RATE,
        momentum=LARS_MOMENTUM,
        weight_decay=LARS_WEIGHT_DECAY,
    )

    # Prepare learning rate scheduler
    warmup_iterations = LR_SCHEDULER_WARMUP_EPOCHS * len(train_dataloader)
    total_iterations = EPOCHS * len(train_dataloader)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=warmup_iterations,  # Number of warmup training steps (not epochs)
        max_epochs=total_iterations,  # Total number of training steps (not epochs)
        start_value=1.0,  # Maximum cosine learning rate factor
        end_value=LR_SCHEDULE_COSINE_MIN,  # Minimum cosine learning rate factor
    )

    # Train the model
    train(
        model=model,
        train_dataloader=train_dataloader,
        validate_dataloader=validate_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        all_hyperparameters=ALL_HYPERPARAMETERS,
        log_to_wandb=log_to_wandb,
    )
