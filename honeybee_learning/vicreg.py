"""VICReg model and training pipeline."""

from __future__ import annotations

import torch
import torchvision
from lightly.data import LightlyDataset
from lightly.loss import VICRegLoss
from lightly.models.modules.heads import VICRegProjectionHead
from torch import nn
from torch.utils.data import DataLoader, Dataset

__all__ = ["train_vicreg"]


# Configuration
DEVICE = "cuda"  # We expect a GPU to be available.
DATALOADER_NUM_WORKERS = 8  # Number of workers for the `DataLoader`


# Hyperparameters

## Projection head configuration. See `VICRegProjectionHead` for more details.
PROJECTION_HEAD_INPUT_DIM = 512
PROJECTION_HEAD_HIDDEN_DIM = 2048
PROJECTION_HEAD_OUTPUT_DIM = 2048
PROJECTION_HEAD_NUM_LAYERS = 2

## Training settings
BATCH_SIZE = 256
LEARNING_RATE = 0.06
EPOCHS = 10


class HoneybeeDataset(Dataset):  # Placeholder for now
    pass


class VICReg(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = VICRegProjectionHead(
            input_dim=PROJECTION_HEAD_INPUT_DIM,
            hidden_dim=PROJECTION_HEAD_HIDDEN_DIM,
            output_dim=PROJECTION_HEAD_OUTPUT_DIM,
            num_layers=PROJECTION_HEAD_NUM_LAYERS,
        )

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)  # Don't flatten across batches
        z = self.projection_head(x)
        return z


def build_model():
    resnet = torchvision.models.resnet18()

    # Remove the last fully connected layer to use the model as a backbone
    backbone = nn.Sequential(*list(resnet.children())[:-1])

    # Create the VICReg model with the backbone
    model = VICReg(backbone)
    return model


def load_dataset():
    torch_dataset = HoneybeeDataset()

    # Lightly models require a `LightlyDataset`
    lightly_dataset = LightlyDataset.from_torch_dataset(torch_dataset)

    return lightly_dataset


def train_vicreg():
    # Load data
    dataset = load_dataset()
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,  # TODO: keep?
        num_workers=DATALOADER_NUM_WORKERS,
    )

    # Prepare model
    model = build_model()
    model = model.to(DEVICE)  # Move to target device

    # Prepare training components
    criterion = VICRegLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    print("Starting VICReg training...")
    model.train()  # Set the model to training mode
    model.zero_grad()  # Zero the gradients before training, just to be safe

    # Iterate over epochs
    for epoch in range(EPOCHS):
        epoch_loss = 0  # Aggregate loss for the epoch

        # Iterate over the dataset
        for batch in dataloader:
            # `x0` and `x1` are two views of the same honeybee.
            x0, x1 = batch[0]  # TODO: This may need to be adjusted based on the dataset

            # Move data to the target device
            x0 = x0.to(DEVICE)
            x1 = x1.to(DEVICE)

            # Forward pass
            z0 = model(x0)
            z1 = model(x1)

            # Compute training loss
            batch_loss = criterion(z0, z1)
            epoch_loss += batch_loss.detach()

            # Backpropagation and optimization
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()  # Zero the gradients for the next iteration

        # Log the average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
