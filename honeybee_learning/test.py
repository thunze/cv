"""Model testing functions.

Applied only to the final model to evaluate its performance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import DEVICE, REPRESENTATIONS_PATH
from .dataset import HoneybeeSample

__all__ = ["train_and_test_linear_predictors"]


# Batch size for precalculating representations of cropped honeybee images.
REPRESENTATION_PRECALCULATION_BATCH_SIZE = 64

# Hyperparameters for training linear predictors
LINEAR_PREDICTORS_TRAIN_EPOCHS = 10  # Number of epochs for which to train predictors
LINEAR_PREDICTORS_LEARNING_RATE = 1e-3  # Learning rate to use for predictors


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
    train_and_validate_dataloader = get_single_dataloader(
        mode="train_and_validate", batch_size=batch_size
    )
    test_dataloader = get_single_dataloader(mode="test", batch_size=batch_size)

    num_representations = (
        len(train_and_validate_dataloader) + len(test_dataloader)
    ) * batch_size

    representations = np.empty(
        (num_representations, model.module.output_dim), dtype=np.float32
    )

    # Freeze the model
    model.eval()  # Set the model to evaluation mode

    # TODO: Map crop indices to representations

    with torch.no_grad():
        # Precalculate representations for the training and validation dataset
        print("\nCalculating representations for the training and validation crops...")
        for i, batch in enumerate(train_and_validate_dataloader):
            print(f"\tProcessing batch {i + 1}/{len(train_and_validate_dataloader)}")
            x, _, _, _ = batch
            x = x.to(DEVICE)  # Move data to the target device
            z = model(x)  # Get the representations
            start_index = i * batch_size
            end_index = start_index + batch_size
            representations[start_index:end_index] = z.cpu().numpy()

        test_repr_offset = len(train_and_validate_dataloader) * batch_size

        # Precalculate representations for the test dataset
        print("\nCalculating representations for the test crops...")
        for i, batch in enumerate(test_dataloader):
            print(f"\tProcessing batch {i + 1}/{len(test_dataloader)}")
            x, _, _, _ = batch
            x = x.to(DEVICE)
            z = model(x)
            start_index = test_repr_offset + i * batch_size
            end_index = start_index + batch_size
            representations[start_index:end_index] = z.cpu().numpy()

    # Save the representations to a file
    representations_filepath = (
        REPRESENTATIONS_PATH / f"{model_filepath.stem}_representations.npy"
    )
    print(f"\nSaving representations to {representations_filepath!r}...")
    np.save(representations_filepath, representations)


def train_and_test_linear_predictors(
    model: nn.DataParallel,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    total_number_of_bees: int,
    train_epochs: int,
    learning_rate: float,
) -> None:
    """With `model` frozen, train three linear predictors on the training dataset
    provided by `train_dataloader`, and test them on the test dataset provided by
    `test_dataloader`.

    The linear predictors are trained in a supervised manner to predict:

    - The bee ID via multi-class classification.
    - The bee class (within comb or not within comb) via binary classification.
    - The bee orientation angle via regression.

    All three predictors are wrapped in a single `LinearEvaluationHead` module for
    more efficient training and testing.

    Args:
        model: Model to validate.
        train_dataloader: `DataLoader` providing the training `HoneybeeDataset`; used
            to train the linear predictors.
        test_dataloader: `DataLoader` providing the test `HoneybeeDataset`; used to
            evaluate the performance of the linear predictors.
        total_number_of_bees: Total number of unique bees in the honeybee dataset
            (across all splits); used to define the output dimension of the bee ID
            classifier.
        train_epochs: Number of epochs to train the linear evaluation head on the
            training dataset for.
        learning_rate: Learning rate for the linear evaluation head optimizer.
    """
    assert torch.is_grad_enabled()  # We need gradients for training
    assert not model.training

    # Freeze the encoder
    for param in model.parameters():
        param.requires_grad = False

    class LinearEvaluationHead(nn.Module):
        """Linear evaluation head placed on top of `model`.

        Args:
            model_output_dim: Output dimension of the projection head of `model`,
                i.e., the dimension of the output of the encoder.
        """

        def __init__(self, model_output_dim: int):
            super().__init__()

            # Define linear layers for the three tasks
            self.classifier_id = nn.Linear(
                model_output_dim, total_number_of_bees
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

    # Initialize the linear evaluation head
    linear_evaluation_head = LinearEvaluationHead(model.module.output_dim)
    linear_evaluation_head = linear_evaluation_head.to(DEVICE)

    # Prepare loss functions for the three tasks
    criterion_id = nn.CrossEntropyLoss()  # For bee ID classification (multi-class)
    criterion_class = nn.BCEWithLogitsLoss()  # For bee class classification (binary)
    criterion_angle = nn.MSELoss()  # For bee orientation regression (real-valued)

    # Prepare optimizer for the linear evaluation head
    optimizer = torch.optim.Adam(linear_evaluation_head.parameters(), lr=learning_rate)

    # --- Training ---
    print(f"\tTraining linear evaluation head for {train_epochs} epochs...")

    linear_evaluation_head.train()  # Set the model to training mode
    linear_evaluation_head.zero_grad()  # Zero gradients, just to be safe

    total_train_samples = len(train_dataloader.dataset)

    # Iterate over epochs
    for _ in range(train_epochs):
        training_loss_epoch_id = 0
        training_loss_epoch_class = 0
        training_loss_epoch_angle = 0

        batch: HoneybeeSample

        # Train for one epoch
        # One pass through the training dataset
        for batch in train_dataloader:
            x, id_, class_, angle = batch

            # Data conversions, move data to target device
            # Note: PyTorch DataLoader already returns tensors
            x = x.to(DEVICE)
            id_ = id_.to(DEVICE)
            class_ = class_.to(DEVICE, dtype=torch.float32)
            angle = angle.to(DEVICE, dtype=torch.float32)

            # Forward pass
            z = model(x)
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
    print("\tTesting linear evaluation head on test dataset...")

    linear_evaluation_head.eval()  # Set the model to evaluation mode
    total_test_samples = len(test_dataloader.dataset)

    test_loss_epoch_id = 0
    test_loss_epoch_class = 0
    test_loss_epoch_angle = 0

    id_predictions, id_targets = [], []
    class_predictions, class_targets = [], []
    angle_predictions, angle_targets = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            x, id_, class_, angle = batch

            # Data conversions, move data to target device
            # Note: PyTorch DataLoader already returns tensors
            x = x.to(DEVICE)
            id_ = id_.to(DEVICE)
            class_ = class_.to(DEVICE, dtype=torch.float32)
            angle = angle.to(DEVICE, dtype=torch.float32)

            # Collect targets for accuracy and MAE calculations
            id_targets.extend(id_.cpu().numpy())
            class_targets.extend(class_.cpu().numpy())
            angle_targets.extend(angle.cpu().numpy())

            # Forward pass
            z = model(x)
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

    # Unfreeze the encoder
    for param in model.parameters():
        param.requires_grad = True
