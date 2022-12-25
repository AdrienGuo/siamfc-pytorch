import os

import cv2

__all__ = ['save_img', 'create_dir']


def save_img(img, filename):
    cv2.imwrite(filename, img)
    print(f"Save image to: {filename}")


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Create directory: {path}")
