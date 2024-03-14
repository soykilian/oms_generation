import numpy as np
import cv2
import yaml
from tqdm import tqdm
import skvideo.io
import os
import subprocess
import platform

# TODO: figure out which parameters are important to keep
# TODO: Go through each parameter and write a sentence or two about what it does


def dvs_v2e(
        params_dir: str = None,
        save_dir: str = None,
        num_frames=None,
        compile_video=False,
        path_to_v2e: str = None,
        output_dir: str = None,
        mode: str = None,
        **kwargs
):
    """
    Computes DVS frames from a video file.

    REQUIREMENTS:
    -> A local conda environment called "v2e" set up to the parameters of v2e from
    their GitHub page is REQUIRED as this function calls the v2e.py script from that conda env.

    -> the v2e GitHub repository must be cloned and its path must be in config.py

    PARAMETERS:
    :param params_dir: Required; String directory of params.yaml file
    :param save_dir: Required; String directory of video file
    :param num_frames: Optional; number of frames to compute DVS for. If not specified, computes for entire video
    :param compile_video: if False, returns numpy array of dvs_frames for further processing.
        If True returns None and saves DVS frames to video using OpenCV 2
    :param mode: either 'ref' or 'main'. If 'ref', computes DVS frames for reference video and saves events
        to 'ref_events.h5'. If 'main', computes DVS for main video.
    :param path_to_v2e: String; path to v2e.py file
    :param output_dir: String; path to output directory

    RESULTS:
    :return: None; saves DVS frames to a video file or returns dvs frames in np array if computing with oms
    """

    assert os.path.exists(path_to_v2e), "Path to v2e.py does not exist. Please check config.py or clone v2e github " \
                                        "repository at this link (https://github.com/SensorsINI/v2e)"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        # if the mode is reference and the output directory is not empty, create a new directory called 'ref'
        if not os.listdir(output_dir) == [] and mode == 'ref':
            new_path = os.path.join(output_dir, 'ref/')
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            output_dir = new_path
        if not os.listdir(output_dir) == []:
            print("Output directory must be empty")
            return False

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

            match mode:
                case "ref":
                    file_dir = os.path.join(os.path.dirname(file_dir), 'input_REF.mp4')
                    events_filename: str = 'ref-events.h5'  # if in reference mode, save as ref-events.h5
                case _:
                    events_filename: str = 'events.h5'
                    pass

    output_folder: str = output_dir
    # folder to store outputs. (default: v2e-output)

    overwrite: bool = False
    # overwrites files in existing folder (checks existence of non-empty output_folder). (default: True)

    unique_output_folder: bool = True
    # makes unique output folder based on output_folder, e.g. output1 (if non-empty output_folder already exists)
    # (default: False)

    out_filename: str = events_filename
    # Output DVS events as hdf5 event database. (program_default: 'events.h5')

    davis_output: bool = False
    # If also save frames in HDF5. (default: False)

    # Output DVS Video Options
    # ---------------------------
    skip_video_output: bool = False
    # Skip producing video outputs, including the original video, SloMo video, and DVS video (default: False)

    dvs_exposure: str = f"duration {1/float(fps)}"
    #  Mode to finish DVS frame event integration: duration time: Use fixed accumulation time in seconds, e.g.
    #  --dvs_exposure duration .005; count n: Count n events per frame, -dvs_exposure count 5000;
    #  area_event N M: frame ends when any area of M x M pixels fills with N events,
    #  -dvs_exposure area_count 500 64 (default: duration 0.01)

    output_mode = "dvs240"
    #  This option sets the output size of the v2e.
    #  Supported models: "dvs128" (128x128), "dvs240" (240x180), "dvs346" (346x260), "dvs640" (640x480),
    #  "dvs1024" (1024x768).

    # Input Options
    # ---------------------------
    avi_frame_rate = fps
    input_frame_rate = fps  # from params.yaml
    # Manually define the video frame rate when the video is presented as a list of image files.
    # When the input video is a video file, this option will be ignored.

    input_slowmotion_factor = 1  # @param {type:"number"}
    # Sets the known slow-motion factor of the input video, i.e. how much the video is slowed down, i.e.,
    # the ratio of shooting frame rate to playback frame rate. input_slowmotion_factor<1 for sped-up video
    # and input_slowmotion_factor>1 for slowmotion video. If an input video is shot at 120fps yet is presented as a
    # 30fps video (has specified playback frame rate of 30Hz, according to file's FPS setting),
    # then set --input_slowdown_factor=4.It means that each input frame represents (1/30)/4 s=(1/120)s.
    # If input is video with intended frame intervals of 1ms that is in AVI file with default 30 FPS playback spec,
    # then use ((1/30)s)*(1000Hz)=33.33333. (default: 1.0)

    # DVS Time Resolution Options
    # ---------------------------
    disable_slomo: bool = True
    # Disables slomo interpolation; the output DVS events will have exactly the timestamp resolution of the source video
    # (which is perhaps modified by --input_slowmotion_factor). (default: False)

    timestamp_resolution: int = 1
    # (Ignored by --disable_slomo.) Desired DVS timestamp resolution in seconds;
    # determines slow motion upsampling factor; the video will be upsampled from source fps to achieve the at least
    # this timestamp resolution.I.e. slowdown_factor = (1/fps)/timestamp_resolution; using a high resolution
    # e.g. of 1ms will result in slow rendering since it will force high upsampling ratio.
    # Can be combind with --auto_timestamp_resolution to limit upsampling to a maximum limit value. (default: None)

    auto_timestamp_resolution: bool = True
    # @markdown - (Ignored by --disable_slomo.) If True (default), upsampling_factor is automatically determined to
    # limit maximum movement between frames to 1 pixel. If False, --timestamp_resolution sets the upsampling factor for
    # input video. Can be combined with --timestamp_resolution to ensure DVS events have at most some resolution.
    # (default: False)

    # DVS Model Options
    # ---------------------------
    condition = "Clean"  # @param ["Custom", "Clean", "Noisy"]
    # Custom: Use following slidebar to adjust your DVS model.
    # Clean: a preset DVS model, generate clean events, without non-idealities.
    # Noisy: a preset DVS model, generate noisy events.

    thres = 0.2  # @param {type:"slider", min:0.05, max:1, step:0.01} # TODO: This is an important param
    # threshold in log_e intensity change to trigger a positive/negative event. (default: 0.2)

    sigma = 0.03  # @param {type:"slider", min:0.01, max:0.25, step:0.001} # TODO: This is a param
    # 1-std deviation threshold variation in log_e intensity change. (default: 0.03)

    cutoff_hz = 200  # @param {type:"slider", min:0, max:300, step:1} # TODO: What is this?
    # photoreceptor first-order IIR lowpass filter cutoff-off 3dB frequency in Hz -
    # see https://ieeexplore.ieee.org/document/4444573 (default: 300)

    leak_rate_hz = 5.18  # @param {type:"slider", min:0, max:100, step:0.01} # TODO: What is this?
    # leak event rate per pixel in Hz - see https://ieeexplore.ieee.org/abstract/document/7962235 (default: 0.01)

    shot_noise_rate_hz = 2.716  # @param {type:"slider", min:0, max:100, step:0.001} # TODO: What is this?
    # Temporal noise rate of ON+OFF events in the darkest parts of scene; reduced in brightest parts. (default: 0.001)

    if condition == "Clean":
        thres = 0.2
        sigma = 0.02
        cutoff_hz = 0
        leak_rate_hz = 0
        shot_noise_rate_hz = 0
    elif condition == "Noisy":
        thres = 0.2
        sigma_thres = 0.05
        cutoff_hz = 30
        leak_rate_hz = 0.1
        shot_noise_rate_hz = 5

    v2e_command = ["v2e.py"]

    # set the input folder
    # the video_path can be a video file or a folder of images
    v2e_command += ["-i", file_dir]

    # set the output folder
    v2e_command += ["-o", output_folder]

    # if the output will rewrite the previous output
    if overwrite:
        v2e_command.append("--overwrite")

    # if the output folder is unique
    v2e_command += ["--unique_output_folder", "{}".format(unique_output_folder).lower()]

    # set to ignore gui
    # v2e_command += ["--ignore-gooey", "True"]

    # set output configs, for the sake of this tutorial, let's just output HDF5 record
    if davis_output:
        v2e_command += ["--davis_output"]
    v2e_command += ["--dvs_h5", out_filename]
    v2e_command += ["--dvs_aedat2", "None"]
    v2e_command += ["--dvs_text", "None"]

    # in Colab, let's say no preview
    v2e_command += ["--no_preview"]

    # if skip video output
    if skip_video_output:
        v2e_command += ["--skip_video_output"]
    else:
        # set DVS video rendering params
        v2e_command += ["--dvs_exposure", dvs_exposure]

    # set slomo related options
    v2e_command += ["--input_frame_rate", "{}".format(input_frame_rate)]
    v2e_command += ["--avi_frame_rate", "{}".format(avi_frame_rate)]
    # v2e_command += ["--input_slowmotion_factor", "{}".format(input_slowmotion_factor)]

    # set slomo VSim_dataset
    if disable_slomo:
        v2e_command += ["--disable_slomo"]
        v2e_command += ["--auto_timestamp_resolution", "false"]
    else:
        v2e_command += ["--slomo_model", slomo_model]
        if auto_timestamp_resolution:
            v2e_command += ["--auto_timestamp_resolution", "{}".format(auto_timestamp_resolution).lower()]
        else:
            v2e_command += ["--timestamp_resolution", "{}".format(timestamp_resolution)]

    # threshold
    v2e_command += ["--pos_thres", "{}".format(thres)]
    v2e_command += ["--neg_thres", "{}".format(thres)]

    # sigma
    v2e_command += ["--sigma_thres", "{}".format(sigma)]

    # DVS non-idealities
    v2e_command += ["--cutoff_hz", "{}".format(cutoff_hz)]
    v2e_command += ["--leak_rate_hz", "{}".format(leak_rate_hz)]
    v2e_command += ["--shot_noise_rate_hz", "{}".format(shot_noise_rate_hz)]

    # append output mode
    v2e_command += [f"--{output_mode}"]
    # v2e_command += [f"--output_width={output_width}"]
    # v2e_command += [f"--output_height={output_height}"]

    test = platform.system()

    match test:
        case "Windows":
            shell = True
        case "Darwin":
            shell = False
        case "Linux":
            shell = False
        case _:
            shell = True

    # Final v2e command

    final_v2e_command = " ".join(v2e_command)

    print(f"The Final v2e command: /bin/bash -i -c cd {path_to_v2e} && conda run -n v2e -v {final_v2e_command}",  "\n")

    # TODO: test to see if this works on windows and mac
    result = subprocess.run(
        # rf'bash -c "cd {path_to_v2e} && conda run -n v2e -v {final_v2e_command}"',
        ['bash', '-i', '-c', f"cd {path_to_v2e} && conda run -n v2e -v {final_v2e_command}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=shell,
        encoding='utf-8'
    )  # works

    print("stdout: ", result.stdout)
    print("stderr: ", result.stderr)
    print("return code: ", result.returncode)

    return True
