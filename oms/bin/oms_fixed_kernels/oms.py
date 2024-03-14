import sys
import traceback
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
import yaml
from tqdm import tqdm
import torchvision
import skvideo.io
import h5py


def construct_circ_filter(radius: int) -> np.ndarray:
    """
    Constructs a circular filter
    :param radius: radius of the filter
    :return kernel: circular filter of size (2 * radius + 1, 2 * radius + 1)
    """
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)

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
    kernel_size = radius * 2 + 1

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


def apply_convolution(x: torch.tensor, kernel: torch.tensor, stride: int = 1) -> torch.tensor:
    """
    Applies convolution to a tensor
    :param x: tensor to apply convolution to (values either 0 or 255)
    :param kernel: kernel to use
    :param stride: stride of the convolution
    :return: tensor after convolution (values between 0 and 1)
    """
    # global sobel_y_kernel
    # global sobel_x_kernel

    # reshape kernel to (1, 1, kernel_height, kernel_width)
    kernel = kernel.reshape((1, 1, kernel.squeeze().shape[0], kernel.squeeze().shape[1])).float()
    x = x.squeeze()[np.newaxis, np.newaxis, ...].float()

    # apply convolution (functional API)
    if stride == 1:
        # padding='same' only works for stride of 1
        convolved = F.conv2d(x, kernel, stride=stride, padding='same')
    else:
        convolved = F.conv2d(x, kernel, stride=stride)

    # remove batch dimension
    convolved = convolved.squeeze(0)

    return convolved


def oms(
        params_dir: str,
        save_dir: str = None,
        dvs_frames: np.ndarray = None,
        num_frames: int = None,
        debug: bool = False,
        filter_func: Callable = None,
        activation_func: Callable = None,
        compile_video: bool = True,
        disable_progress_metrics: bool = False,
        stride: int = 1,
        **kwargs,
):
    """
    Computes OMS frames from a video file of DVS frames saves OMS frames to a video file.
    Requires preprocessed DVS frames. Returns True if the function ran successfully and a video is compiles.
    Returns False if there was an error in the size of the kernel radii. Center rad must be smaller than surround rad.

    :param params_dir: String; Path to yaml parameters file.
    :param save_dir: String; save location.
    :param dvs_frames: np.ndarray; NumPy array of dvs frames ready to be computed to OMS
    :param num_frames: Optional; number of frames to compute DVS for. If not specified, computes for entire video.
    :param debug: Optional; prints debug statements.
    :param filter_func: Optional; Function; type of filter to use.
    :param activation_func: Optional; Function; type of activation function to use for OMS calculation.
    :param compile_video: Optional; Bool; whether to compile the OMS frames into a video.
    :param disable_progress_metrics: Optional; Bool; whether to use tqdm.
    :param stride: Optional; Int; stride of the convolution.
    :param **kwargs: Optional; if you want to set any parameter manually for experiments do so here.

    :return: Bool; False if error with radii of filters or True if complete successfully.
    """

    # ---------------------
    # -- LOAD PARAMETERS --
    # ---------------------

    # Load DVS params from yaml parameters file
    with open(params_dir) as file:
        params = yaml.safe_load(file)

        file_dir = params['video']['dir']
        fps = params['video']['fps']
        width = params['video']['width']
        w_resize_factor = params['params']['w_resize_factor']
        height = params['video']['height']
        h_resize_factor = params['params']['h_resize_factor']
        resize_factor_motion = params['params']['resize_factor_motion']
        center_kernel_radius = params['params']['center_kernel_radius']
        surround_kernel_radius = params['params']['surround_kernel_radius']
        motion_surround_weight = params['params']['motion_surround_weight']
        THRESHOLD = params['params']['oms_threshold']

    # used for experimenting
    if "file_dir" in kwargs.keys():
        file_dir = kwargs["file_dir"]
    if "fps" in kwargs.keys():
        fps = kwargs["fps"]
    if "width" in kwargs.keys():
        width = kwargs["width"]
    if "w_resize_factor" in kwargs.keys():
        w_resize_factor = kwargs["w_resize_factor"]
    if "height" in kwargs.keys():
        height = kwargs["height"]
    if "h_resize_factor" in kwargs.keys():
        h_resize_factor = kwargs["h_resize_factor"]
    if "resize_factor_motion" in kwargs.keys():
        resize_factor_motion = kwargs["resize_factor_motion"]
    if "oms_threshold" in kwargs.keys():
        THRESHOLD = kwargs["oms_threshold"]
    if "center_rad" in kwargs.keys():
        center_kernel_radius = kwargs["center_rad"]
    if "surround_rad" in kwargs.keys():
        surround_kernel_radius = kwargs["surround_rad"]
    if "motion_surround_weight" in kwargs.keys():
        motion_surround_weight = kwargs["motion_surround_weight"]

    filter_construction_function = filter_func

    # prints the name of the function that wil be used to construct the filter kernel
    if debug:
        print(filter_construction_function.__name__)

    # When experimenting with different filter sizes, if there is an error in filter differences
    # just print that there is an error and continue the experiment to the next preset
    try:
        assert center_kernel_radius < surround_kernel_radius
    except AssertionError:
        _, _, tb = sys.exc_info()
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]

        print('An error occurred on line {} in statement {}'.format(line, text))
        return False

    # Define toTensor
    toTensor = torchvision.transforms.ToTensor()

    # Initialize center and surround kernel according to the type of initializer function passed in
    center_kernel = toTensor(filter_construction_function(radius=center_kernel_radius))
    surround_kernel = toTensor(filter_construction_function(radius=surround_kernel_radius))

    # if stride is not one, we need to pad the center kernel to make sure that the resolution down sample is the same
    # for both center and surround after convolution
    if stride != 1:
        surround_size = (surround_kernel.squeeze().shape[0] - 1) // 2  # division to the extra dims on both sides evenly
        center_size = (center_kernel.squeeze().shape[0] - 1) // 2  # division to the extra dims on both sides evenly
        pad = surround_size-center_size
        # add 0 pad around the kernel to make sure that resolution down sample is the same for both center and surround
        # after convolution
        center_kernel = F.pad(center_kernel, (pad, pad, pad, pad), "constant", 0)

        if debug:
            print(pad)
            print(center_kernel, "\n")
            print(surround_kernel)

    # --------------------
    # --- COMPUTE OMS ----
    # --------------------

    # Create placeholder for resultant OMS frames
    oms_frames = np.zeros((num_frames, int(height/stride), int(width/stride)), dtype=np.uint8)

    # Iterate through DVS frames
    for i in tqdm(range(num_frames), desc="Computing OMS frames", disable=disable_progress_metrics):

        # Set frame to the relevant DVS frame
        frame = dvs_frames[i]

        # Convert frame to correct input for the convolution
        if not torch.is_tensor(frame):
            frame = toTensor(frame).float()

        # CONVOLUTION

        # Apply filters along image
        center = apply_convolution(x=frame, kernel=center_kernel, stride=stride)
        surround = apply_convolution(x=frame, kernel=surround_kernel, stride=stride)

        # ACTIVATION FUNCTION

        # Apply activation function to center and surround
        oms_frame = activation_func(
            center=center.numpy(),
            surround=surround.numpy(),
            threshold=THRESHOLD,
            debug=debug,
        )

        oms_frames[i] = oms_frame

        # Check if there are any events in the frame. If there are none, it should print error
        if oms_frame.max() == 0 and "bo" in kwargs.keys():
            return "error"

    # return just oms_frames if not compiling video
    if not compile_video:

        if "compile_to_h5py" in kwargs.keys():

            if kwargs["compile_to_h5py"]:

                # save to h5py file
                with h5py.File(save_dir + ".h5", "w") as f:
                    f.create_dataset("oms_frames", data=oms_frames)

        return oms_frames

    # --------------------
    # ----- SAVE OMS -----
    # --------------------

    # Write to video file
    """
    outfile = save_dir + ".mp4"
    writer = skvideo.io.FFmpegWriter(outfile, inputdict={"-r": str(fps), '-pix_fmt': "gray"},
                                     outputdict={'-vcodec': 'libx264', '-pix_fmt': "gray", '-r': str(fps)})

    for i in tqdm(range(oms_frames.shape[0]), desc="writing video"):
        writer.writeFrame(oms_frames[i])

    # Release capture and destroy windows
    writer.close()
    """

    print("finished writing video")

    return True


def plot_frame(frame: torch.tensor) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots a black and white frame in Matplotlib
    :param frame:
    :return: fig, ax
    """
    # create figure and axes object. 1 row, 2 columns
    fig, ax = plt.subplots()

    # # convert to grayscale
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # plot original frame
    ax.imshow(frame[0], cmap='gray')

    return fig, ax
