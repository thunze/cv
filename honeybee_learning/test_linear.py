"""Functions for testing an unsupervised representation learning model on the
honeybee dataset by training and testing a set of linear predictors on top of the
model's frozen encoder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import wandb
from torch import nn

from .config import DEVICE, TOTAL_NUMBER_OF_BEES, WANDB_ENTITY, WANDB_PROJECT
from .dataset_test import HoneybeeRepresentationSample, get_representation_dataloader

__all__ = ["train_and_test_linear_predictors"]


# Hyperparameters for linear evaluation head training
LINEAR_PREDICTORS_INPUT_DIM = 128  # Output dimension of both SimCLR and VICReg models
LINEAR_PREDICTORS_BATCH_SIZE = 256  # Batch size to use for training and testing
LINEAR_PREDICTORS_EPOCHS = 10  # Number of epochs for which to train predictors
LINEAR_PREDICTORS_LEARNING_RATE = 1e-3  # Learning rate to use for predictors


class LinearEvaluationHead(nn.Module):
    """Linear evaluation head used to train and test linear predictors on top of a
    frozen self-supervised representation learning model.

    Args:
        model_output_dim: Output dimension of the encoder, i.e., the input dimension
            of the linear evaluation head. This should match the output dimension of
            the self-supervised representation learning model used to generate the
            representations of the honeybee images.
    """

    def __init__(self, model_output_dim: int):
        super().__init__()

        # Define linear layers for the three tasks
        self.classifier_id = nn.Linear(
            model_output_dim, TOTAL_NUMBER_OF_BEES
        )  # Multi-class
        self.classifier_class = nn.Linear(model_output_dim, 1)  # Binary (0/1)
        self.regressor_angle = nn.Linear(model_output_dim, 1)  # Real-valued

    def forward(self, z):
        """Forward pass through the linear evaluation head.

        Args:
            z: Representation tensor of shape (batch_size, model_output_dim),
                where `model_output_dim` is the output dimension of the encoder.

        Returns:
            Tuple of (bee ID logits, bee class logit, bee angle prediction). Note
            that each of these tensors may have an additional batch dimension.
        """
        id_logits = self.classifier_id(z)
        class_logit = self.classifier_class(z)
        angle_pred = self.regressor_angle(z)
        return id_logits, class_logit, angle_pred


def train_and_test_linear_predictors(
    representations_path: Path, *, log_to_wandb: bool = False
) -> None:
    """Train three linear predictors on top of a frozen self-supervised representation
    learning model, and evaluate their performance on the honeybee dataset.

    Uses a combination of the training and validation splits of the honeybee dataset
    to train the linear predictors, and the test split to evaluate their performance.

    The linear predictors are trained in a supervised manner to predict:

    - The bee ID via multi-class classification.
    - The bee class (within comb or not within comb) via binary classification.
    - The bee orientation angle via regression.

    All three predictors are wrapped in a single `LinearEvaluationHead` module for
    more efficient training and testing.

    The self-supervised representation learning model is not used directly here.
    Instead, the precalculated representations of the honeybee images are loaded from
    `representations_path`, which should point to a file containing the
    representations of the honeybee images in the dataset in the same order as the
    metadata file containing the labels for the three tasks at `METADATA_PATH`.

    For more information on the process of precalculating the representations, see
    the `test_precalculate` module.

    Args:
        representations_path: Path to the file containing precalculated
            representations of all honeybee images in the dataset.
        log_to_wandb: Whether to log training and testing progress to
            Weights & Biases (wandb).
    """
    # Prepare logging for the run
    run_name = f"test_linear_{representations_path.stem}"
    print(f"Starting training run {run_name!r}...")

    # Initialize wandb run if enabled
    if log_to_wandb:
        wandb_run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            anonymous="must",  # Force anonymous login, needed to create anonymous key
            name=run_name,
            config={
                "representations_path": str(representations_path),
                "linear_predictors_input_dim": LINEAR_PREDICTORS_INPUT_DIM,
                "linear_predictors_batch_size": LINEAR_PREDICTORS_BATCH_SIZE,
                "linear_predictors_epochs": LINEAR_PREDICTORS_EPOCHS,
                "linear_predictors_learning_rate": LINEAR_PREDICTORS_LEARNING_RATE,
            },
        )
    else:
        wandb_run = None

    # Initialize the linear evaluation head
    linear_evaluation_head = LinearEvaluationHead(LINEAR_PREDICTORS_INPUT_DIM)
    linear_evaluation_head = linear_evaluation_head.to(DEVICE)

    # Get data loaders for training and testing
    train_dataloader = get_representation_dataloader(
        representations_path,
        mode="train_and_validate",
        batch_size=LINEAR_PREDICTORS_BATCH_SIZE,
    )
    test_dataloader = get_representation_dataloader(
        representations_path,
        mode="test",
        batch_size=LINEAR_PREDICTORS_BATCH_SIZE,
    )

    # Prepare loss functions for the three tasks
    criterion_id = nn.CrossEntropyLoss()  # For bee ID classification (multi-class)
    criterion_class = nn.BCEWithLogitsLoss()  # For bee class classification (binary)
    criterion_angle = nn.MSELoss()  # For bee orientation regression (real-valued)

    # Prepare optimizer for the linear evaluation head
    optimizer = torch.optim.Adam(
        linear_evaluation_head.parameters(),
        lr=LINEAR_PREDICTORS_LEARNING_RATE,
    )

    # --- Training ---
    print(f"Training linear evaluation head for {LINEAR_PREDICTORS_EPOCHS} epochs...")

    linear_evaluation_head.train()  # Set the model to training mode
    linear_evaluation_head.zero_grad()  # Zero gradients, just to be safe

    total_train_samples = len(train_dataloader.dataset)

    # Iterate over epochs
    for epoch in range(LINEAR_PREDICTORS_EPOCHS):
        training_loss_epoch_id = 0
        training_loss_epoch_class = 0
        training_loss_epoch_angle = 0

        print(f"\nEpoch {epoch + 1}/{LINEAR_PREDICTORS_EPOCHS}: Training...")

        batch: HoneybeeRepresentationSample

        # Train for one epoch
        # One pass through the training dataset
        for i, batch in enumerate(train_dataloader):
            print(f"\tTraining on batch {i + 1}/{len(train_dataloader)}...")

            z, id_, class_, angle = batch

            # Data conversions, move data to target device
            # Note: PyTorch DataLoader already returns tensors
            z = z.to(DEVICE)
            id_ = id_.to(DEVICE)
            class_ = class_.to(DEVICE, dtype=torch.float32)
            angle = angle.to(DEVICE, dtype=torch.float32)

            # Forward pass
            id_logits, class_logit, angle_pred = linear_evaluation_head(z)

            # Compute training loss
            batch_loss_id = criterion_id(id_logits, id_)
            batch_loss_class = criterion_class(class_logit.squeeze(), class_)
            batch_loss_angle = criterion_angle(angle_pred.squeeze(), angle)
            batch_loss_total = batch_loss_id + batch_loss_class + batch_loss_angle

            training_loss_epoch_id += batch_loss_id.item()
            training_loss_epoch_class += batch_loss_class.item()
            training_loss_epoch_angle += batch_loss_angle.item()

            # Backpropagation and optimization
            batch_loss_total.backward()
            optimizer.step()  # Update model parameters
            optimizer.zero_grad()  # Zero the gradients for the next iteration

        # Compute average training loss for the epoch
        avg_training_loss_epoch_id = training_loss_epoch_id / total_train_samples
        avg_training_loss_epoch_class = training_loss_epoch_class / total_train_samples
        avg_training_loss_epoch_angle = training_loss_epoch_angle / total_train_samples

    # --- Testing ---
    print("\nTesting linear evaluation head on test dataset...")

    linear_evaluation_head.eval()  # Set the model to evaluation mode
    total_test_samples = len(test_dataloader.dataset)

    test_loss_epoch_id = 0
    test_loss_epoch_class = 0
    test_loss_epoch_angle = 0

    id_predictions, id_targets = [], []
    class_predictions, class_targets = [], []
    angle_predictions, angle_targets = [], []

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            print(f"\tTesting on batch {i + 1}/{len(test_dataloader)}...")

            z, id_, class_, angle = batch

            # Data conversions, move data to target device
            # Note: PyTorch DataLoader already returns tensors
            z = z.to(DEVICE)
            id_ = id_.to(DEVICE)
            class_ = class_.to(DEVICE, dtype=torch.float32)
            angle = angle.to(DEVICE, dtype=torch.float32)

            # Collect targets for accuracy and MAE calculations
            id_targets.extend(id_.cpu().numpy())
            class_targets.extend(class_.cpu().numpy())
            angle_targets.extend(angle.cpu().numpy())

            # Forward pass
            id_logits, class_logit, angle_pred = linear_evaluation_head(z)

            # Compute test loss
            batch_loss_id = criterion_id(id_logits, id_)
            batch_loss_class = criterion_class(class_logit.squeeze(), class_)
            batch_loss_angle = criterion_angle(angle_pred.squeeze(), angle)
            batch_loss_total = batch_loss_id + batch_loss_class + batch_loss_angle

            test_loss_epoch_id += batch_loss_id.item()
            test_loss_epoch_class += batch_loss_class.item()
            test_loss_epoch_angle += batch_loss_angle.item()

            # Collect predictions for accuracy and MAE calculations
            id_predictions.extend(id_logits.argmax(dim=1).cpu().numpy())
            class_predictions.extend(
                (torch.sigmoid(class_logit.squeeze()).cpu().numpy() > 0.5).astype(int)
            )
            angle_predictions.extend(angle_pred.squeeze().cpu().numpy())

    # Compute test metrics
    avg_test_loss_id = test_loss_epoch_id / total_test_samples
    avg_test_loss_class = test_loss_epoch_class / total_test_samples
    avg_test_loss_angle = test_loss_epoch_angle / total_test_samples

    avg_test_accuracy_id = float(
        (np.array(id_predictions) == np.array(id_targets)).mean()
    )
    avg_test_accuracy_class = float(
        (np.array(class_predictions) == np.array(class_targets)).mean()
    )
    avg_test_mae_angle = float(
        np.abs(np.array(angle_predictions) - np.array(angle_targets)).mean()
    )
