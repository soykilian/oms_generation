import numpy as np
import skvideo.io
from tqdm import tqdm


def plot_animation(frames: np.ndarray, figsize=(5, 5), fps=120, save_path=None):
    """Helper function that animates a tensor of frames of shape (TCHW). If you run this in a
    Jupyter notebook, you can display the animation inline like shown in the example below.

    Parameters:
        frames: numpy array or tensor of shape (TCHW)
        figsize: tuple(int, int) specifying the size of the figure


    Returns:
        The animation object. Store this in a variable to keep it from being garbage collected until displayed.
    """

    if frames.shape[1] == 2:
        rgb = np.zeros((frames.shape[0], 3, *frames.shape[2:]))
        rgb[:, 1:, ...] = frames
        frames = rgb
    if frames.shape[1] in [1, 2, 3]:
        frames = np.moveaxis(frames, 1, 3)

    frames = (frames / frames.max()) * 255

    print(frames.shape)

    writer = skvideo.io.FFmpegWriter(save_path, inputdict={"-r": str(fps)},
                                     outputdict={'-vcodec': 'libx264', '-r': str(fps)},
                                     verbosity=0)

    for i in tqdm(range(frames.shape[0]), desc="writing video"):
        writer.writeFrame(frames[i])

    # Release capture and destroy windows
    writer.close()

    return True
