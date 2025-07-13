"""Image transformations optionally used for data augmentation in the training."""

from __future__ import annotations

import random

import torchvision.transforms.functional as F
from torchvision import transforms

__all__ = ["crop_resize_flip"]


def random_transform(img):
    """Apply the following transformations to the input image `img`:

    - Random cropping and resizing to the original size
    - Random vertical flip (50% chance)
    - Random horizontal flip (50% chance)
    - Random rotation (0, 90, 180 or 270°)

    Args:
        img: The input image.

    Returns:
        The transformed output image.
    """
    # Random crop parameters for this image
    i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(128, 128))
    img = F.crop(img, i, j, h, w)

    # Resize back to desired size
    img = F.resize(img, (224, 224))

    # Random horizontal flip
    if random.random() > 0.5:
        img = F.hflip(img)

    # Random vertical flip
    if random.random() > 0.5:
        img = F.vflip(img)

    # Random rotation
    angle = random.choice([0, 90, 180, 270])
    img = F.rotate(img, angle)

    return img


def crop_resize_flip(image_pair):
    """Apply the following transformations to the pair of input images `image_pair`:

    - Random cropping and resizing to the original size
    - Random vertical flip (50% chance)
    - Random horizontal flip (50% chance)
    - Random rotation (0, 90, 180 or 270°)

    Args:
        image_pair: The input image pair as a tuple of tensors.

    Returns:
        The transformed images as a tuple of tensors.
    """
    img1, img2 = image_pair
    img1 = random_transform(img1)
    img2 = random_transform(img2)
    return img1, img2
