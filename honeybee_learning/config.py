"""Configuration for the honeybee representation learning project."""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "BASE_PATH",
    "CHECKPOINTS_PATH",
    "CROPS_PATH",
    "FIGURES_PATH",
    "FRAMES_PATH",
    "METADATA_PATH",
    "PREPROCESSING_LOG_PATH",
    "REPRESENTATIONS_PATH",
    "TRAJECTORIES_PATH",
    "VIDEOS_PATH",
    "TOTAL_NUMBER_OF_BEES",
    "MAX_FRAME_DIFFERENCE",
    "DATASET_CREATE_SHUFFLE",
    "DATASET_CREATE_SHUFFLE_SEED",
    "TRAIN_RATIO",
    "VALIDATION_RATIO",
    "DEVICE",
    "DATALOADER_NUM_WORKERS",
    "CHECKPOINT_EVERY_N_EPOCHS",
    "VISUALIZATION_NUM_SAMPLES",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
]


# Paths
BASE_PATH = Path("/scratch/cv-course2025/group7")
CHECKPOINTS_PATH = BASE_PATH / "checkpoints"
CROPS_PATH = BASE_PATH / "crops128" / "crops.npy"
FIGURES_PATH = BASE_PATH / "figures"
FRAMES_PATH = BASE_PATH / "frames"
METADATA_PATH = BASE_PATH / "crops128" / "metadata224.npy"
PREPROCESSING_LOG_PATH = BASE_PATH / "processing128.log"
REPRESENTATIONS_PATH = BASE_PATH / "representations"
TRAJECTORIES_PATH = BASE_PATH / "trajectories"
VIDEOS_PATH = BASE_PATH / "videos"

# Data
TOTAL_NUMBER_OF_BEES = 876  # Total number of unique bees in the honeybee dataset
MAX_FRAME_DIFFERENCE = 5  # Max. number of frames that count bee as temporally adjacent
DATASET_CREATE_SHUFFLE = True  # Whether to use shuffling in dataset creation or not
DATASET_CREATE_SHUFFLE_SEED = 42  # Seed to use when shuffling in dataset creation
TRAIN_RATIO = 0.7  # Amount of the data to use for training
VALIDATION_RATIO = 0.15  # Amount of the data to use for validation
RATIO_SAMPLE = 1.0  # Portion of the data used for sampling train/val data (1.0 = 100%)

# Training and validation
DEVICE = "cuda"  # Device to train and run the model on, typically a GPU
DATALOADER_NUM_WORKERS = 16  # Number of processes used by `DataLoader` instances
CHECKPOINT_EVERY_N_EPOCHS = 10  # Save model checkpoint every N epochs
VISUALIZATION_NUM_SAMPLES = 6000  # Number of samples to use in dim. reduction methods

# Weights & Biases (wandb) configuration
WANDB_ENTITY = "thunze"  # Team name
WANDB_PROJECT = "honeybee-learning"  # Project name
