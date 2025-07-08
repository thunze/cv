import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
from honeybee_learning.dataset_test import HoneybeeRepresentationSample
import numpy as np
from honeybee_learning.config import DATASET_CREATE_SHUFFLE_SEED

NUM_SAMPLE = 1000

#TODO: Add docstrings, make plots better

def get_pca_projections(representations, n_components = 3):

    my_pca = PCA(n_components=n_components)

    projections = my_pca.fit_transform(representations)

    return projections

def get_tsne_projections(representations, n_components = 3):

    my_tsne = TSNE(n_components=n_components)

    projections = my_tsne.fit_transform(representations)

    return projections

def evaluate():

    # TODO: Add dataloader handling
    # Create dataloader
    dataloader = None

    # Load representation and metadata
    representations, labels = extract_representations_and_labels(dataloader)

    # Create and save plots
    evaluate_samples(representations, labels, "title ", plot_figs=True)


def evaluate_samples(representations, labels,data_title, plot_figs = False):

    # Take a sample, so we have a relatively equal split of different bees for visualizing bee id
    sample_id_representations, sample_id_labels = create_sample(representations, labels, sample_size=NUM_SAMPLE, random=False)

    # Take a random sample for visualizing class and angle
    random_sample_representations, random_sample_labels = create_sample(representations, labels, sample_size=NUM_SAMPLE, random=True)

    # Calculate projections for both samples
    sample_id_projections = []
    sample_id_projections.append((get_tsne_projections(sample_id_representations),"t-sne"))
    sample_id_projections.append((get_pca_projections(sample_id_representations), "pca"))

    random_sample_projections = []
    random_sample_projections.append((get_tsne_projections(random_sample_representations),"t-sne"))
    random_sample_projections.append((get_pca_projections(random_sample_representations), "pca"))

    # Create plots for the projections
    plots = []
    for (projection, name) in sample_id_projections:
        plots.append((plot_bee_id(projection, sample_id_labels[:,0]), f"{name}_bee_id"))

    for (projection, name) in random_sample_projections:
        plots.append((plot_class(projection, random_sample_labels[:, 1]), f"{name}_class"))
        plots.append((plot_angle(projection, random_sample_labels[:, 2]), f"{name}_angle"))

    PLOTS_PATH = Path("./")

    for fig, plot_name in plots:
        title = f"{data_title}_{plot_name}"
        fig.suptitle(title)
        plot_filepath = (PLOTS_PATH / f"{title}.png")
        fig.savefig(plot_filepath)
        if plot_figs:
            plt.show()
        plt.close(fig)


def create_sample(representations, labels,sample_size = 5000 ,random = True):
    """
    If random is true, creates a randomly sampled sample from representations and labels and returns it.
    If it is false, samples them, so we get a relatively even distribution over all bee ids.
    :param representations:
    :param labels:
    :param random:
    :return:
    """

    if random:
        rng = np.random.default_rng(DATASET_CREATE_SHUFFLE_SEED)
        indices = rng.choice(len(representations), size = sample_size, replace = False)

        return representations[indices], labels[indices]

    else:
        rng = np.random.default_rng(DATASET_CREATE_SHUFFLE_SEED)
        bee_ids = labels[:, 0]
        unique_ids = np.unique(bee_ids)
        id_to_indices = {bid: np.where(bee_ids == bid)[0] for bid in unique_ids}
        per_id = max(1, sample_size // len(unique_ids))
        selected_indices = []

        for bid, indices in id_to_indices.items():
            num_to_sample = min(per_id, len(indices))
            chosen = rng.choice(indices, size=num_to_sample, replace=False)
            selected_indices.extend(chosen)

        rng.shuffle(selected_indices)
        selected_indices = selected_indices[:sample_size]

        return representations[selected_indices], labels[selected_indices]





def extract_representations_and_labels(dataloader):

    representations = []
    labels = []

    for sample in dataloader: #type: HoneybeeRepresentationSample
        representations.append(sample.z.cpu().numpy())
        labels.append((sample.id_, sample.class_, sample.angle))

    representations = np.vstack(representations)
    labels = np.array(labels)

    return representations, labels


def plot_bee_id(projections, labels):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(projections[:, 0], projections[:, 1], projections[:, 2],
                         c=labels, cmap='tab20', s=15, alpha=0.8)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.set_title("Projection colored by Bee ID")

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Bee ID")

    return fig

def plot_class(projections, labels):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")

    # Separate classes
    class0_mask = labels == 0
    class1_mask = labels == 1

    ax.scatter(projections[class0_mask, 0], projections[class0_mask, 1], projections[class0_mask, 2],
               c="royalblue", label="Class 0", s=15, alpha=0.8)
    ax.scatter(projections[class1_mask, 0], projections[class1_mask, 1], projections[class1_mask, 2],
               c="crimson", label="Class 1", s=15, alpha=0.8)

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.set_title("Projection by Binary Class")

    ax.legend(title="Class", loc='best')

    return fig

def plot_angle(projections, labels):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize angles 0-359 to 0-1 for HSV colormap
    angle_norm = labels / 360.0

    scatter = ax.scatter(projections[:, 0], projections[:, 1], projections[:, 2],
                         c=angle_norm, cmap='hsv', s=15, alpha=0.8)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.set_title("Projection colored by Angle")

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Angle (degrees)")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0°", "90°", "180°", "270°", "360°"])

    return fig

