import torchvision.transforms.functional as F
import torchvision.transforms
import random

from torchvision import transforms

__all__ = ["crop_resize_flip"]

def random_transform(img):
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
    img1, img2 = image_pair
    img1 = random_transform(img1)
    img2 = random_transform(img2)
    return img1, img2
