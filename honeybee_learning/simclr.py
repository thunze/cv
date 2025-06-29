import numpy as np
import torchvision
from config import DEVICE
from dataset import get_dataloader
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.models.utils import get_weight_decay_parameters
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler
from torch import nn
from train import train

__all__ = ["train_simclr"]

# Hyperparameters

## Basic training parameters
BATCH_SIZE = 4096  # Figure 9; small epoch size + large batch size = good performance
EPOCHS = 1000  # Paper goes up to 800

## Parameters for validating the model on linear predictors
LINEAR_PREDICTORS_TRAIN_EPOCHS = 800  # Number of epochs for which to train predictors
LINEAR_PREDICTORS_LEARNING_RATE = 0.075 * np.sqrt(
    BATCH_SIZE
)  # Learning rate to use for predictors; makes no difference when batch size = 4096

## Loss parameters
TEMPERATURE = 0.1  # Default 0.1
GATHERED_DISTRIBUTED = True  # Trains on more negatives samples; if more than 1 GPU used

## Optimizer parameters
LARS_LEARNING_RATE = 0.3 * BATCH_SIZE / 256  # As specified in paper
LARS_MOMENTUM = 0.9
LARS_WEIGHT_DECAY = 1e-6

## Learning rate scheduler parameters
LR_SCHEDULER_WARMUP_EPOCHS = 10

## Projection head configuration. See `SimCLRProjectionHead` for more details.
PROJECTION_HEAD_INPUT_DIM = 2048
PROJECTION_HEAD_HIDDEN_DIM = 2048
PROJECTION_HEAD_OUTPUT_DIM = 128
PROJECTION_HEAD_NUM_LAYERS = 2


# Hyperparameters to log for the run
ALL_HYPERPARAMETERS = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "simclr_loss_temp": TEMPERATURE,
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

        # Resize input images to 224x224, as expected by the ResNet backbone
        self.resize = torchvision.transforms.Resize((224, 224))

        resnet = torchvision.models.resnet50()
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone = backbone

        # Input and hidden dim of same size
        self.projection_head = SimCLRProjectionHead(
            input_dim=PROJECTION_HEAD_INPUT_DIM,
            hidden_dim=PROJECTION_HEAD_HIDDEN_DIM,
            output_dim=PROJECTION_HEAD_OUTPUT_DIM,
            num_layers=PROJECTION_HEAD_NUM_LAYERS,
        )

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (..., channels, height, width), representing the
                input images.

        Returns:
            Output tensor of shape (..., feature dimensions), representing the
            projected features.
        """
        x_r = self.resize(x)
        h = self.backbone(x_r).flatten(start_dim=1)
        z = self.projection_head(h)
        return z


def train_simclr(*, log_to_wandb: bool = False) -> None:
    """Training pipeline for SimCLR model

    Args:
        log_to_wandb: Whether to log training progress to Weights & Biases (wandb).
    """

    # Load training and validation data
    train_dataloader = get_dataloader(mode="train", batch_size=BATCH_SIZE)
    validate_dataloader = get_dataloader(mode="validate", batch_size=BATCH_SIZE)
    train_pair_dataloader = get_dataloader(
        pairs=True, mode="train", batch_size=BATCH_SIZE
    )
    validate_pair_dataloader = get_dataloader(
        pairs=True, mode="validate", batch_size=BATCH_SIZE
    )
    # Prepare model
    model = SimCLR()
    model = model.to(DEVICE)  # Move model to target device

    # Prepare loss function
    criterion = NTXentLoss(
        temperature=TEMPERATURE, gather_distributed=GATHERED_DISTRIBUTED
    )

    # Prepare optimizer
    # For performance reasons, don't apply weight decay to norm and bias parameters.
    params_weight_decay, params_no_weight_decay = get_weight_decay_parameters(
        [model.backbone, model.projection_head]
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
    scheduler = (
        CosineWarmupScheduler(
            optimizer, warmup_epochs=warmup_iterations, max_epochs=total_iterations
        ),
    )

    # Train the model
    train(
        model=model,
        train_dataloader=train_dataloader,
        validate_dataloader=validate_dataloader,
        train_pair_dataloader=train_pair_dataloader,
        validate_pair_dataloader=validate_pair_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        linear_predictors_train_epochs=LINEAR_PREDICTORS_TRAIN_EPOCHS,
        linear_predictors_learning_rate=LINEAR_PREDICTORS_LEARNING_RATE,
        all_hyperparameters=ALL_HYPERPARAMETERS,
        log_to_wandb=log_to_wandb,
    )
