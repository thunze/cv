# Imports
import torchvision
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from torch import nn
from lightly.utils.lars import LARS

from lightly.loss.ntx_ent_loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead

from config import DEVICE
from scheduler import CosineAnnealingWithLinearRampLR
from dataset import get_dataloader
from train import train

__all__ = ["train_simclr"]

# Hyperparameters

## Basic training parameters
BATCH_SIZE = 4096 # As defined in original paper (Figure 9; small epoch size + large batch size = good performance)
EPOCHS = 100

## Loss parameters
TEMPERATURE = 0.1 # Default 0.1

## Optimizer parameters
LARS_LEARNING_RATE=0.3*BATCH_SIZE/256 # as defined in paper
LARS_MOMENTUM=0.9
LARS_WEIGHT_DECAY=1e-6
LARS_TRUST_COEF=1e-3

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
    "lars_trust_coeff": LARS_TRUST_COEF,
    "projection_head_input_dim": PROJECTION_HEAD_INPUT_DIM,
    "projection_head_hidden_dim": PROJECTION_HEAD_HIDDEN_DIM,
    "projection_head_output_dim": PROJECTION_HEAD_OUTPUT_DIM,
    "projection_head_num_layers": PROJECTION_HEAD_NUM_LAYERS,
}



class SimCLR(nn.Module):
    """
        SimCLR model adapted for the honeybee problem task
    """
    def __init__(self):
        super().__init__()
        
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
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


def train_simclr():
    '''
        Training pipeline for SimCLR model
    '''
    # Load training and validation data
    train_dataloader = get_dataloader(mode="train", batch_size=BATCH_SIZE)
    validate_dataloader = get_dataloader(mode="validate", batch_size=BATCH_SIZE)

    # Prepare model
    model = SimCLR()
    model = model.to(DEVICE)  # Move model to target device

    # Prepare loss function
    criterion = NTXentLoss(
        temperature=TEMPERATURE,
    )

    # Prepare optimizer
    # For performance reasons, don't apply weight decay to norm and bias parameters.
    #params_weight_decay, params_no_weight_decay = get_weight_decay_parameters(
    #    [model.backbone, model.projection_head]
    #)
    
    optimizer = LARS(
        lr=LARS_LEARNING_RATE, 
        momentum=LARS_MOMENTUM,
        weight_decay=LARS_WEIGHT_DECAY,
        trust_coef=LARS_TRUST_COEF
    )
    
    # Prepare learning rate scheduler
    total_iterations = EPOCHS * len(train_dataloader)
    scheduler = CosineAnnealingWithLinearRampLR(
        optimizer,
        T_max=total_iterations,  # Total number of training steps (not epochs)
        ramp_len=10 # Linear ramp scheduler runs (10 epochs is default in original paper)
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