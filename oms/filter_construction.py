import numpy as np


__all__ = [
    "construct_circ_filter",
    "construct_square_filter",
    "gaussian_filter",
]


def construct_circ_filter(radius: int) -> np.ndarray:
    """
    Constructs a circular filter
    :param radius: radius of the filter
    :return kernel: circular filter of size (2 * radius + 1, 2 * radius + 1)
    """
    kernel = np.zeros((2 * radius, 2 * radius), dtype=np.float32)

    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            if (i - radius) ** 2 + (j - radius) ** 2 <= (radius / 2) ** 2:
                kernel[i, j] = 1
            if (radius / 2) ** 2 < (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
                kernel[i, j] = .5

    return kernel


def gaussian_filter(radius: int, sigma=1.0, muu=0) -> np.ndarray:
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
    kernel_size = radius * 2

    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x ** 2 + y ** 2)

    # lower normal part of gaussian
    normal = 1 / (2.0 * np.pi * sigma ** 2)

    # Calculating Gaussian filter
    gauss = np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2))) * normal

    return gauss


def construct_square_filter(radius: int) -> np.ndarray:
    """
    Constructs a square filter of ones
    :param radius: radius of filter
    :return kernel: square filter of size (2 * radius, 2*radius)
    """

    kernel = np.ones((2 * radius, 2 * radius), dtype=np.float32)
    return kernel


def construct_laplacian_filter(radius: int) -> np.ndarray:
    """
    Constructs a laplacian filter
    :param radius: radius of filter
    :return kernel: laplacian filter of size (2 * radius + 1, 2 * radius + 1)
    """

    raise NotImplementedError


if __name__ == "__main__":

    gauss_test = gaussian_filter(5)
    gauss_test = (gauss_test - gauss_test.min()) / (gauss_test.max() - gauss_test.min())

    print(gauss_test, "\n")

    circ_test = construct_circ_filter(2)

    print(circ_test, "\n")

    square_test = construct_square_filter(2)

    print(square_test, "\n")

