"""Functions for evaluating an unsupervised representation learning model on the
honeybee dataset by visualizing the learned representations with the help of
dimensionality reduction techniques and the labels provided by the dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from honeybee_learning.config import (
    DATASET_CREATE_SHUFFLE_SEED,
    FIGURES_PATH,
    VISUALIZATION_NUM_SAMPLES,
)
from honeybee_learning.dataset_test import get_representation_dataloader

__all__ = ["evaluate", "evaluate_samples"]


def get_pca_projections(representations, n_components=3):
    """Given a set of representations in the shape (N, D) with N being the number of
    samples and D being the dimensionality, standardize them using `StandardScaler`
    and then reduce them to `n_components` dimensions using PCA.

    Args:
        representations: Representations to apply PCA on.
        n_components: Dimensionality to reduce to (3 by default).

    Returns:
        Projections of shape (N, `n_components`).
    """
    my_pca = PCA(n_components=n_components)
    standardized = StandardScaler().fit_transform(representations)
    projections = my_pca.fit_transform(standardized)
    return projections


def get_tsne_projections(representations, n_components=3, pca_components=32):
    """Given a set of representations in the shape (N, D) with N being the number of
    samples and D being the dimensionality, standardize them using `StandardScaler`,
    apply PCA and then project them into `n_components` dimensions using t-SNE.

    Args:
        representations: Representations to apply t-SNE on.
        n_components: Dimensionality to project into (3 by default).
        pca_components: Dimensionality output of the PCA applied (32 by default).

    Returns:
        Projections of shape (N, `n_components`).
    """
    my_tsne = TSNE(n_components=n_components)
    standardized = StandardScaler().fit_transform(representations)

    pca = PCA(n_components=pca_components)
    reduced = pca.fit_transform(standardized)
    projections = my_tsne.fit_transform(reduced)
    return projections


def evaluate(representations_filepath: Path, plot_figs=False):
    """Using a filepath to precalculated representations, sample from them, project
    them into 3-dimensional space and then plot the results.

    Args:
        representations_filepath: Path to the `.npy` file containing the
            representations.
    """
    # Create dataloader
    dataloader = get_representation_dataloader(
        representations_filepath, mode="test", batch_size=256
    )
    # Load representation and metadata from the dataloader
    representations, labels = extract_representations_and_labels(dataloader)

    # Create and save plots
    evaluate_samples(
        representations, labels, representations_filepath.stem, plot_figs=plot_figs
    )


def evaluate_samples(representations, labels, data_title, plot_figs=False):
    """Take two samples from the representations and labels provided, then calculate
    the 3-dimensional projections and plot them using `matplotlib.pyplot`.

    Figures are saved in the directory specified in the `config` module.

    Args:
        representations: The learned representations of shape (N, D).
        labels: The labels belonging to the representations with shape (N, 3).
        data_title: Title to use for titling and saving the plots.
        plot_figs: Whether to plot the figures immediately.
    """

    # Take a sample, so we have a relatively equal split of different bees for
    # visualizing bee ID
    sample_id_representations, sample_id_labels = create_sample(
        representations, labels, sample_size=VISUALIZATION_NUM_SAMPLES, random=False
    )

    # Take a random sample for visualizing class and angle
    random_sample_representations, random_sample_labels = create_sample(
        representations, labels, sample_size=VISUALIZATION_NUM_SAMPLES, random=True
    )

    # Calculate projections for both samples
    sample_id_projections = []
    sample_id_projections.append(
        (get_tsne_projections(sample_id_representations), "t-sne")
    )
    sample_id_projections.append(
        (get_pca_projections(sample_id_representations), "pca")
    )

    random_sample_projections = []
    random_sample_projections.append(
        (get_tsne_projections(random_sample_representations), "t-sne")
    )
    random_sample_projections.append(
        (get_pca_projections(random_sample_representations), "pca")
    )

    # Create plots for the projections
    plots = []
    # Plots for the honeybee-id projections
    for projection, name in sample_id_projections:
        plots.append(
            (plot_bee_id(projection, sample_id_labels[:, 0], name), f"{name}_bee_id")
        )

    # Plots for the randomly sampled projections
    for projection, name in random_sample_projections:
        plots.append(
            (plot_class(projection, random_sample_labels[:, 1], name), f"{name}_class")
        )
        plots.append(
            (plot_angle(projection, random_sample_labels[:, 2], name), f"{name}_angle")
        )

    # Create a folder to save all the figures in
    plot_folder = FIGURES_PATH / data_title
    os.makedirs(plot_folder, exist_ok=True)

    # Save all created plots, optionally display them
    for fig, plot_name in plots:
        plot_filepath = plot_folder / f"{plot_name}.png"
        fig.savefig(plot_filepath)
        if plot_figs:
            plt.show()
        plt.close(fig)

    # Calculate silhouette score for all targets
    score_full_id = silhouette_score(representations, labels[:, 0])
    score_full_class = silhouette_score(representations, labels[:, 1])
    score_full_angle = silhouette_score(representations, labels[:, 2])

    # Calculate silhouette scores for the two samples
    score_sample_id = silhouette_score(
        sample_id_representations, sample_id_labels[:, 0]
    )
    score_random_class = silhouette_score(
        random_sample_representations, random_sample_labels[:, 1]
    )
    score_random_angle = silhouette_score(
        random_sample_representations, random_sample_labels[:, 2]
    )

    # Write silhouette scores to a text file
    score_file_path = plot_folder / "silhouette_scores.txt"
    with open(score_file_path, "w") as f:
        f.write("Silhouette Scores for all representations:\n")
        f.write(f"Bee ID: {score_full_id:.4f}\n")
        f.write(f"Class: {score_full_class:.4f}\n")
        f.write(f"Angle: {score_full_angle:.4f}\n\n")

        f.write("Silhouette Scores for sample_id sample:\n")
        f.write(f"Bee ID: {score_sample_id:.4f}\n\n")

        f.write("Silhouette Scores for random_sample sample:\n")
        f.write(f"Class: {score_random_class:.4f}\n")
        f.write(f"Angle: {score_random_angle:.4f}\n")


def create_sample(representations, labels, sample_size=5000, random=True):
    """Create a subsample of the given dataset with optional balanced sampling.

    If `random` is `True`, creates a random sample from representations and labels
    and returns it. If `random` is `False`, samples them, so we get a relatively even
    distribution over all bee IDs.

    Args:
        representations: Representations to draw the sample from.
        labels: Labels to draw the sample from.
        sample_size: Number of samples to draw.
        random: Whether to perform random sampling or draw a sample evenly distributed
            across all bee IDs.

    Returns:
        Tuple containing the sampled representations and labels.
    """

    if random:
        rng = np.random.default_rng(DATASET_CREATE_SHUFFLE_SEED)
        indices = rng.choice(len(representations), size=sample_size, replace=False)

        return representations[indices], labels[indices]

    else:
        rng = np.random.default_rng(DATASET_CREATE_SHUFFLE_SEED)

        bee_ids = labels[:, 0]
        unique_ids = np.unique(bee_ids)

        # Map bee ids to their corresponding indices
        id_to_indices = {bid: np.where(bee_ids == bid)[0] for bid in unique_ids}
        per_id = max(1, sample_size // len(unique_ids))
        selected_indices = []

        # Randomly sample up to per_id samples for each bee.
        for bid, indices in id_to_indices.items():
            num_to_sample = min(per_id, len(indices))
            chosen = rng.choice(indices, size=num_to_sample, replace=False)
            selected_indices.extend(chosen)

        rng.shuffle(selected_indices)
        selected_indices = selected_indices[:sample_size]

        return representations[selected_indices], labels[selected_indices]


def extract_representations_and_labels(dataloader):
    """Given a data loader for the `HoneybeeRepresentationDataset`, extract the
    representations and labels.

    Supports batch sizes > 1.

    Args:
        dataloader: `DataLoader` for the `HoneybeeRepresentationDataset`.

    Returns:
        A tuple containing the representations, labels.
    """
    representations = []
    labels = []

    for i, batch in enumerate(
        dataloader
    ):  # batch is a batch of HoneybeeRepresentationSample
        # batch.z shape: (batch_size, feature_dim)
        representations.append(batch.z.cpu().numpy())

        # batch.id_, batch.class_, batch.angle are sequences of length batch_size
        batch_labels = list(zip(batch.id_, batch.class_, batch.angle))
        labels.extend(batch_labels)

    representations = np.vstack(representations)
    labels = np.array(labels)

    return representations, labels


def plot_bee_id(projections, labels, reduction_name):
    """Create a scatter plot for 3-dimensional representations of honeybee images,
    colored by the bee_id contained in the labels.

    Args:
        projections: Array of shape (N, 3) containing the projected representations.
        labels: Associated labels containing the bee_ids.

    Returns:
        The generated figure.
    """
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        projections[:, 0],
        projections[:, 1],
        projections[:, 2],
        c=labels,
        cmap="tab20",
        s=5,
        alpha=0.8,
    )
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.set_title(f"{reduction_name} projection colored by Bee ID")

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Bee ID")

    return fig


def plot_class(projections, labels, reduction_name):
    """Create a scatter plot for 3-dimensional representations of honeybee images,
    colored by the class contained in the labels.

    Args:
        projections: Array of shape (N, 3) containing the projected representations.
        labels: Associated labels containing the class.

    Returns:
        The generated figure.
    """
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")

    # Separate classes
    class0_mask = labels == 0
    class1_mask = labels == 1

    ax.scatter(
        projections[class0_mask, 0],
        projections[class0_mask, 1],
        projections[class0_mask, 2],
        c="royalblue",
        label="Class 0",
        s=5,
        alpha=0.8,
    )
    ax.scatter(
        projections[class1_mask, 0],
        projections[class1_mask, 1],
        projections[class1_mask, 2],
        c="crimson",
        label="Class 1",
        s=5,
        alpha=0.8,
    )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.set_title(f"{reduction_name} Projection by Binary Class")

    ax.legend(title="Class", loc="best")

    return fig


def plot_angle(projections, labels, reduction_name):
    """Create a scatter plot for 3-dimensional representations of honeybee images,
    colored by the angle contained in the labels.

    Visualizes the points in 3D space using a continuous HSV colormap to represent
    the angles in degrees.

    Args:
        projections: Array of shape (N, 3) containing the projected representations.
        labels: Associated labels containing the angle in degrees.

    Returns:
        The generated figure.
    """
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")

    # Normalize angles 0-359 to 0-1 for HSV colormap
    angle_norm = labels / 360.0

    scatter = ax.scatter(
        projections[:, 0],
        projections[:, 1],
        projections[:, 2],
        c=angle_norm,
        cmap="hsv",
        s=5,
        alpha=0.8,
    )
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.set_title(f"{reduction_name} Projection colored by Angle")

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Angle (degrees)")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0°", "90°", "180°", "270°", "360°"])

    return fig
