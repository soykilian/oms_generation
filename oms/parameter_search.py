# This file implements bayesian optimization to find the
# best parameters for the 3 different filters we have.
# The black-box function we're optimizing is the loss function
# which is defined below.

# Import standard libraries.
import sys

import numpy as np
import torchvision
from skimage.metrics import structural_similarity as SSIM
from skopt.space import Categorical, Space

# Import project-specific libraries
from version_2.bin.oms_fixed_kernels.oms import oms
from version_2.experiments.experiment_1_vsim.filter_construction import *


def bo_evaluate_point(
        test_point: list, params_dir, loader,
        fps, width, height, stride, activation_func
) -> float:

    """
    Evaluates a test point by running the OMS algorithm on the data and comparing the results to the reference frames.

    :param test_point: list of parameters to test
    :param params_dir: directory of all parameters files in the dataset
    :param data_paths: list of paths to the data and targets: [data_path, targets_path]
    :param fps: frames per second of the data
    :param width: width of the data
    :param height: height of the data
    :param kernel_name: name of the kernel to use
    :param kernel_func: kernel function to use
    :param activation_func: activation function to use

    :return: ssim loss between the predicted frames and the true frames
    """

    threshold = test_point[3]
    center_kernel_radius = test_point[1]
    surround_kernel_radius = test_point[2]
    kernel_name = test_point[0]

    match kernel_name:
        case "gaussian":
            kernel_func = gaussian_filter
        case "arb_kern":
            kernel_func = construct_circ_filter
        case "square":
            kernel_func = construct_square_filter
        case _:
            raise ValueError(f"Invalid kernel name: {kernel_name}")

    for i, (data, targets, file_name, class_name) in enumerate(iter(loader)):
        # file_name is the name of the video file, like data_1 or data_2 etc...
        # join the params folder dir and the data name (data_N)
        params_file = params_dir

        # Data shape is (1, time steps, 180, 240)
        data, targets = data.squeeze(), targets.squeeze()  # remove batch dim

        # run OMS
        oms_frames = oms(
            params_dir=params_file + '.yaml',
            save_dir=None,
            dvs_frames=data,
            compile_video=False,
            num_frames=data.shape[0],
            filter_func=kernel_func,
            activation_func=activation_func,
            fps=fps,
            width=width,
            height=height,
            center_rad=center_kernel_radius,
            surround_rad=surround_kernel_radius,
            oms_threshold=threshold,
            stride=stride,  # figure out what hardware stride should do
            compile_to_h5py=False,
            disable_progress_metrics=True,
            bo=True,
        )

        # Returns a terribly high number if the entire frame zeros out
        if type(oms_frames) is str or type(oms_frames) is bool:
            return sys.maxsize

        # METRICS CALCULATION

        num_frames = oms_frames.shape[0]  # number of frames in the video
        ssim_hist = []

        # iterate through the frames and calculate the metrics comparing the oms frames to the target frames
        for ind in range(num_frames):
            if stride != 1:
                target = torchvision.transforms.Resize(
                    (int(height / stride), int(width / stride))
                )(targets[ind][np.newaxis, np.newaxis, ...]).numpy()
            else:
                target = targets[ind].numpy()

            acc = SSIM(oms_frames[ind], target.squeeze(), data_range=1.)
            ssim_hist.append(acc)

    # calculate the average of the metrics
    ssim_avg = np.mean(np.stack(ssim_hist))

    return 1 - ssim_avg


def create_search_space() -> Space:
    """
    Creates a search space for the Bayesian Optimization
    """
    return Space([
        Categorical(["gaussian", "arb_kern", "square"], name="Filter Type"),
        Categorical(np.arange(1, 4), name="Center Radius"),
        Categorical(np.arange(2, 8), name="Surround Radius"),
        Categorical(np.linspace(.4, 1.1, 1000), name="OMS Threshold"),
    ])
