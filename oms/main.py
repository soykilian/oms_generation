import os
import numpy as np
from datetime import datetime as dt
from skimage.metrics import structural_similarity as SSIM
from oms.bin.oms_fixed_kernels.oms import oms

from oms.metrics.spike_rate import spike_rate_metric
from torch.utils.data import DataLoader


width = 346
height = 260
fps = 40
params_dir = os.path.join(os.getcwd(), "oms", "data_1")
def generate_oms(dataset, kernel, save_dir,generate_video= False):
    loader = DataLoader(dataset)
    for i, (data, targets, file_name, class_name) in enumerate(iter(loader)):
        data, targets = data.squeeze(), targets.squeeze()
        params_file = os.path.join(params_dir)
         
        # if there is not a folder for the class of results, make one
        if not os.path.exists(os.path.join(save_dir, class_name[0])):
            os.mkdir(os.path.join(save_dir, class_name[0]))

        # if there is not a folder for the results, make one
        if not os.path.exists(os.path.join(save_dir, class_name[0], file_name[0])):
            os.mkdir(os.path.join(save_dir, class_name[0], file_name[0]))
        
        #file_path = os.path.join(save_dir, file_name[0] + "neuroscience")
        oms_frames = oms(
            params_dir=params_file + '.yaml',
            save_dir=os.path.join(save_dir, class_name[0], file_name[0] + "/", file_name[0] + "_neuroscience"),
            dvs_frames=data,
            compile_video=generate_video,
            num_frames=data.shape[0],
            filter_func=kernel["filter"],
            activation_func=kernel["activation_function"],
            fps=fps,
            width=width,
            height=height,
            center_rad=kernel["r_c"],
            surround_rad=kernel["r_s"],
            oms_threshold=kernel["threshold"],
            stride=kernel["stride"],  # figure out what hardware stride should do
            compile_to_h5py=True,
        )
        num_frames = oms_frames.shape[0]  # number of frames in the video
        ssim_hist = []
        spike_rate_hist = []

        # iterate through the frames and calculate the metrics comparing the oms frames to the target frames
        for ind in range(num_frames):
            if kernel["stride"] != 1:
                target = torchvision.transforms.Resize(
                    (int(height / stride), int(width / stride))
                )(targets[ind][np.newaxis, np.newaxis, ...]).numpy()
            else:
                target = targets[ind].numpy()

            if oms_frames[ind].shape != target.squeeze().shape:
                print("WARNING: OMS frame and target frame are not the same size")
                print(f"OMS frame shape: {oms_frames[ind].shape}")
                print(f"Target frame shape: {target.squeeze().shape}")

            acc = SSIM(oms_frames[ind], target.squeeze(), data_range=1.)
            spike_rate = spike_rate_metric(oms_frames[ind])
            ssim_hist.append(acc)
            spike_rate_hist.append(spike_rate)

    # calculate the average of the metrics
    ssim_avg = np.mean(np.stack(ssim_hist))
    spike_rate_avg = np.sum(np.stack(spike_rate_hist)) / len(spike_rate_hist)  # average the spike rates

    print(f"SSIM average: {ssim_avg}")
    print(f"spike_rate: {spike_rate_avg} \n")

    # VISUALIZATION

    # dataset.plot_events(fps=fps, save_dir=os.path.join(os.getcwd(), 'results/', f"dvs_data.mp4"))

    # SAVING

   # for path in os.listdir(save_dir):
    #     if "box" in path or "floor" in path or "table" in path or "wall" in path or "fast" in path:
     #       shutil.move(os.path.join(save_dir, path), experiment_path)

