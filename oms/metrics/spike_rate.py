# Imports
import torch
import torch.nn.functional as F
import math
import numpy as np

__all__ = ['spike_rate_metric']


def spike_rate_metric(img1: np.ndarray) -> float:
    """
    Calculate the spike rate of the input image. (just number of spikes in an image / area of image)
    """
    area = img1.squeeze().shape[0] * img1.squeeze().shape[1]

    return np.sum(img1) / area
