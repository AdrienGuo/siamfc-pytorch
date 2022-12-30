import os

import cv2

__all__ = ['save_img', 'create_dir']


def save_img(img, filename):
    cv2.imwrite(filename, img)
    print(f"Save image to: {filename}")


def create_dir(dir, sub_dir=None):
    """Create one directory recursively."""
    if sub_dir is not None:
        if isinstance(sub_dir, list):
            for dir_name in sub_dir:
                dir = os.path.join(dir, dir_name)
        else:
            dir = os.path.join(dir, sub_dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"Create new dir: {dir}")
    return dir


def create_dirs(*dirs):
    """Create multiple directories."""
    for dir in dirs:
        create_dir(dir)