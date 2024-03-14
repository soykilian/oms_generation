import numpy as np
import cv2
import yaml
from tqdm import tqdm
import skvideo.io


def dvs(
        params_dir: str = None,
        save_dir: str = None,
        num_frames=None,
        compile_video=False,
        debug=False,
        disable_progress_metrics: bool = False,
        **kwargs
):
    """
    Computes DVS frames from a video file
    :param params_dir: Required; String directory of params.yaml file
    :param save_dir: Required; String directory of video file
    :param num_frames: Optional; number of frames to compute DVS for. If not specified, computes for entire video
    :param compile_video: if False, returns numpy array of dvs_frames for further processing.
        If True returns None and saves DVS frames to video using OpenCV 2
    :param debug: Optional; prints debug statements
    :param disable_progress_metrics: Optional; Bool; whether or not to use tqdm

    :return: None; saves DVS frames to a video file or returns dvs frames in np array if computing with oms
    """

    # ---------------------
    # -- LOAD PARAMETERS --
    # ---------------------

    if params_dir is not None:
        with open(params_dir) as file:
            params = yaml.safe_load(file)

            file_dir = params['video']['dir']
            fps = params['video']['fps']
            width = params['video']['width']
            w_resize_factor = params['params']['w_resize_factor']
            height = params['video']['height']
            h_resize_factor = params['params']['h_resize_factor']
            resize_factor_motion = params['params']['resize_factor_motion']
            THRESHOLD = params['params']['dvs_threshold']

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
    if "dvs_threshold" in kwargs.keys():
        THRESHOLD = kwargs["dvs_threshold"]

    # --------------------
    # --- COMPUTE DVS ----
    # --------------------

    # Load video file as cv2 VideoCapture object and count frames
    cap = cv2.VideoCapture(file_dir)

    # if not specified, use cv2 to get number of frames in the video file
    if num_frames is None:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if debug:
        print(f"file_dir: {file_dir}")
        print(f"num_frames: {num_frames}")

    # Create placeholder for resultant dvs frames
    out = np.zeros((num_frames, int(height * h_resize_factor), int(width * w_resize_factor)), dtype=np.uint8)

    if debug:
        print(out.shape)

    # Iterate through until capture is exhausted
    for i in tqdm(range(num_frames), desc="Computing DVS frames", disable=disable_progress_metrics):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to B&W

        if i == 1 and debug:
            print(frame.shape)
            print(frame.max())
            print(frame.min())

        if i > 0:  # second frame onwards
            diff_frame = np.abs(frame - last_frame)  # difference between frames

            # Threshold
            diff_frame[diff_frame < THRESHOLD] = 0
            diff_frame[diff_frame >= THRESHOLD] = 255

            diff_frame = cv2.resize(diff_frame, dsize=None, fx=resize_factor_motion, fy=resize_factor_motion)  # resize
            out[i, :, :] = diff_frame  # add to output

        last_frame = frame  # save last frame

        if not ret:
            break

    if debug:
        print(f"finished computing frames {out.shape}")

    # --------------------
    # ----- SAVE DVS -----
    # --------------------

    # returns dvs VSim_dataset as NumPy array if computing along with oms
    if not compile_video:
        # returning num_frames in case it was computed with the
        # cv2 video capture which will not be used again in oms loop
        # out.shape[0] == num_frames
        return out

    # write to video file if computing dvs standalone
    if compile_video:
        outfile = save_dir + ".mp4"
        writer = skvideo.io.FFmpegWriter(outfile, inputdict={"-r": str(fps), '-pix_fmt': "gray"},
                                         outputdict={'-vcodec': 'libx264', '-pix_fmt': "gray", '-r': str(fps)},
                                         verbosity=1)

        for i in tqdm(range(out.shape[0]), desc="writing video"):
            if i > 0:  # first frame is blank so we skip
                writer.writeFrame(out[i])

        # Release capture and destroy windows
        writer.close()

    cap.release()
    cv2.destroyAllWindows()

    print("finished writing video")
