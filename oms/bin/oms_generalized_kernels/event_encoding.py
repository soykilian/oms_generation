import numpy as np
import h5py


def events_to_frames(
        events: np.ndarray,
        width: int,
        height: int,
        duration: float,
        frame_rate: int,
        save_frames=False,
        **kwargs
) -> np.ndarray:
    """
    Takes DVS events and converts them to frames.
    :param events: numpy array of events: [timestamp (microsecond), x, y, polarity (1 or 0)]
    :param width: pixels
    :param height: pixels
    :param duration: in seconds
    :param frame_rate: frames per second (fps) of the video
    :param save_frames: If True, must pass `save_dir` argument. Saves frames to .h5py file
    :return: numpy array of frames
    """

    # Calculate number of frames
    num_frames: int = int(duration * frame_rate)

    # time between frames
    delta_t: float = 1 / float(frame_rate)

    # Initialize frames
    frames = np.zeros((num_frames, width, height), dtype=np.uint8)

    # if save_frames is true, will save to h5py file
    if save_frames:
        save_dir = kwargs['save_dir']
        file = h5py.File(save_dir, 'w')
        dset = file.create_dataset('frames', (num_frames, width, height), dtype=np.uint8, data=frames)

    return frames
