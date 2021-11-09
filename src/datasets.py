"""Extra generative modeling benchmark datasets not provided by Paddle."""

import os
import urllib

import PIL
import numpy as np
import paddle
from paddle import distribution
from paddle.nn import functional as F
import paddle.io
from paddle.vision import datasets
from paddle.vision import transforms


def _dynamically_binarize(x):
    return paddle.bernoulli(x)

def _resize_to_32(x):
    return F.pad(x, [2, 2, 2, 2])


def get_mnist_loaders(batch_size, dynamically_binarize=False, resize_to_32=False):
    """Create train and test loaders for the MNIST dataset.

    Args:
        batch_size: The batch size to use.
        dynamically_binarize: Whether to dynamically  binarize images values to {0, 1}.
        resize_to_32: Whether to resize the images to 32x32.
    Returns:
        Tuple of (train_loader, test_loader).
    """
    transform = [transforms.ToTensor()]
    if dynamically_binarize:
        transform.append(_dynamically_binarize)
    if resize_to_32:
        transform.append(_resize_to_32)
    transform = transforms.Compose(transform)
    train_loader = paddle.io.DataLoader(
        datasets.MNIST(mode='train', transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = paddle.io.DataLoader(
        datasets.MNIST(mode='test', transform=transform),
        batch_size=batch_size,
    )
    return train_loader, test_loader
