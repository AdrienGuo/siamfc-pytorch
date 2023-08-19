import numpy as np

from .augmentation import Augmentations


class PCBAugmentation(Augmentations):
    def __init__(self, args) -> None:
        self.clahe = args.clahe
        self.flip = args.flip

    def __call__(self, img, box):
        # Flip
        if self.flip and self.flip > np.random.random():
            img, box = self.vertical_flip(img, box)
            img, box = self.horizontal_flip(img, box)

        # CLAHE 3.0
        if self.clahe:
            img = self.CLAHE(img)

        return img, box
