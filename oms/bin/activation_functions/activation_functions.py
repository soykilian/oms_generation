import numpy as np
import torch

__all__ = ['neuro_activation_function', 'hardware_activation_function']


def neuro_activation_function(
        center: np.ndarray, surround: np.ndarray, threshold: float,
        motion_surround_weight: float = 1.0, debug: bool = False,
) -> np.ndarray:
    """
    Neuroscience-backed oms activation function.

    :param center: np.ndarray of center kernel applied to video frames. Shape: (height, width)
    :param surround: np.ndarray of surround kernel applied to video frames. Shape: (height, width)
    :param threshold: float value to threshold the difference between center and surround
    :param motion_surround_weight: float value to weight the surround kernel. Default: 1.0.
    :param debug: bool value to print debug statements. Default: False.

    :returns: np.ndarray of oms frames. Shape: (height, width)
    """

    if debug:
        print(f"shape of center {center.shape}, shape of surround {surround.shape}, threshold {threshold}")

    # Subtract center from weighted_surround
    events = np.abs(center - motion_surround_weight * surround)

    # normalize frames after subtracting
    events = (events - events.min()) / events.max()

    if debug:
        print(f"shape of events (center - surround) {events.shape}")

    oms_frame = np.zeros_like(surround.squeeze())  # create a copy of center shape but all zeros

    # events.squeeze() because events is shape (1, height, width). We want (height, width)
    # at all indices where events is greater than oms_threshold, set to 255
    oms_frame[np.where(events.squeeze() >= threshold)] = 255
    oms_frame[np.where(events.squeeze() < threshold)] = 0

    return oms_frame


def hardware_activation_function(
    center: np.ndarray, surround: np.ndarray, threshold: float,
    debug: bool = False,
) -> np.ndarray:
    """
    Hardware-backed oms activation function.

    :param center: np.ndarray of center kernel applied to video frames. Shape: (height, width)
    :param surround: np.ndarray of surround kernel applied to video frames. Shape: (height, width)
    :param threshold: float value to normalize the area difference of the surround kernel.
        contrast_ratio = surround area / center area
    :param debug: bool value to print debug statements. Default: False.

    :return: np.ndarray of oms frames. Shape: (height, width)
    """

    oms_frame = np.zeros_like(surround.squeeze())  # create a copy of center shape but all zeros

    # wherever the center is greater than the surround per unit area, set to 255
    oms_frame[np.where(center.squeeze() >= (surround.squeeze() / threshold))] = 255  # TODO: figure out why switching the sign made this amazing

    return oms_frame
