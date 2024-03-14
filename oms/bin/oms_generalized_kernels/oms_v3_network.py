import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


def _gaussian_filter(radius: int, sigma=1.0, muu=0) -> np.ndarray:
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
    kernel_size = radius * 2 + 1

    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size, dtype=np.float32),
                       np.linspace(-1, 1, kernel_size, dtype=np.float32))
    dst = np.sqrt(x ** 2 + y ** 2)

    # lower normal part of gaussian
    normal = 1 / (2.0 * np.pi * sigma ** 2)

    # Calculating Gaussian filter
    gauss = np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2))) * normal

    return gauss


def construct_circ_filter(radius: int) -> torch.tensor:
    """
    Constructs a circular filter
    :param radius: radius of the filter
    :return kernel: circular filter of size (2 * radius + 1, 2 * radius + 1)
    """
    kernel = torch.zeros((2 * radius + 1, 2 * radius + 1), dtype=torch.float32, requires_grad=True)

    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            if (i - radius) ** 2 + (j - radius) ** 2 <= (radius / 2) ** 2:
                kernel[i, j] += 1.
            if (radius / 2) ** 2 < (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
                kernel[i, j] += .5

    print(kernel.requires_grad)

    return kernel


class OMSV3Network(nn.Module):

    def __init__(self, center_rad=2, surround_rad=5, batch_size=1, motion_surround_weight=None):
        super().__init__()

        # Initialize layers
        self.center = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(center_rad * 2) + 1, padding="same", dtype=torch.float32
        )
        self.surround = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(surround_rad * 2) + 1, padding="same", dtype=torch.float32
        )

        # self.motion_surround_weight = motion_surround_weight if motion_surround_weight else 1

        with torch.no_grad():
            self.center.weight = nn.Parameter(construct_circ_filter(center_rad).reshape(1, 1, center_rad * 2 + 1, center_rad * 2 + 1))
            self.surround.weight = nn.Parameter(construct_circ_filter(surround_rad).reshape(1, 1, surround_rad * 2 + 1, surround_rad * 2 + 1))

        self.batch_size = batch_size

    def forward(self, x):
        center_weights = torch.clone(self.center.weight)
        center_weights = F.pad(center_weights, (2, 2, 2, 2))

        center_events = F.conv2d(
            input=x,
            weight=center_weights,
            stride=1,
            padding="same"
        )

        surround_events = self.surround(x)

        oms_events = surround_events - center_events
        oms_events = (oms_events-oms_events.min()) / oms_events.max()

        return oms_events

        """# Subtract center from weighted_surround
        out = self.surround(x) - self.center(x)

        # normalize frames after subtracting
        out = (out - out.min()) / out.max()

        return out"""
