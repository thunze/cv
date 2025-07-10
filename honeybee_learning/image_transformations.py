import torchvision.transforms.functional as F
import torchvision.transforms
import random

from torchvision import transforms

__all__ = ["crop_resize_flip"]

def crop_resize_flip(image_pair):

    img1, img2, = image_pair

    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=(64, 64))
    img1 = F.crop(img1, i, j, h, w)
    img2 = F.crop(img2, i, j, h, w)

    # Resize back to original size (or any desired final size)
    img1 = F.resize(img1, (224,224))
    img2 = F.resize(img2, (224,224))

    # Random horizontal flip
    if random.random() > 0.5:
        img1 = F.hflip(img1)
        img2 = F.hflip(img2)

    # Random vertical flip
    if random.random() > 0.5:
        img1 = F.vflip(img1)
        img2 = F.vflip(img2)

    # Random rotation
    angle = random.choice([0, 90, 180, 270])
    img1 = F.rotate(img1, angle)
    img2 = F.rotate(img2, angle)

    return img1, img2