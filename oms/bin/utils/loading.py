import numpy as np
import h5py

__all__ = ["load_oms"]


def load_oms(path: str = None) -> np.ndarray:
    """
    This function takes in a path to an oms file (.h5) and returns the oms frames as a numpy array.
    """

    with h5py.File(path, 'r') as f:

        oms_frames = f['oms_frames'][:]

    return oms_frames
