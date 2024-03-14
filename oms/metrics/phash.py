import numpy as np
import imagehash
from PIL import Image


def phash_comp(a, b):
    """
    Compares two frames using the perceptual hash algorithm.
    The lower the output, the better. 0 is the same image.
    :param a: np.ndarray of shape (height, width)
    :param b: np.ndarray of shape (height, width)
    """

    a = a.transpose()
    b = b.transpose()
    arr_a = np.squeeze(a)
    arr_a *= 255
    arr_a = arr_a.astype(np.uint8)

    arr_b = np.squeeze(b)
    arr_b *= 255
    arr_b = arr_b.astype(np.uint8)

    # Load the images
    image1 = Image.fromarray(arr_a)
    image2 = Image.fromarray(arr_b)

    # Generate perceptual hashes
    hash1 = imagehash.phash(image1)
    hash2 = imagehash.phash(image2)

    # Calculate Hamming distance
    distance = hash1 - hash2
    return distance
