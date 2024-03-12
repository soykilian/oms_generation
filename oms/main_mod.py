import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import os
import shutil
from datetime import datetime as dt
from skimage.metrics import structural_similarity as SSIM
import sys
sys.path.insert(0, '/home/mavi/iris')
from version_2.bin.oms_fixed_kernels.oms import oms
from version_2.experiments.experiment_1_vsim.filter_construction import *
from version_2.bin.activation_functions.activation_functions import *
from version_2.metrics.spike_rate import spike_rate_metric
from version_2.metrics.phash import phash_comp

#from evimo_dataset import EVIMODataset
from mod_dataset import MODDataset
#from bdd_dataset import bdd_dataset
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_dir = "/home/shared/MOD/"
params_dir = os.path.join("/home/mavi/iris/version_2/experiments/experiment_3_vsim/data_1")
#save_dir =  '/home/shared/datasets/even_bdd/426x240/oms/'
#save_dir = os.path.join(os.getcwd(), 'results/')
save_dir = os.path.join(os.getcwd(), 'results/')



# constant params

#width = 423
#height = 238
width = 346
height = 260
fps = 40

center_kernel_radius = 2
surround_kernel_radius = 4

# bo params
oms_threshold = .9

kernels = {
     #"circle_1_.5": (construct_circ_filter, hardware_activation_function, 1, .75),
     #"hardware": (gaussian_filter, hardware_activation_function, 2 * surround_kernel_radius, 1.25),
     #"hardware": (construct_square_filter, hardware_activation_function, 2 * surround_kernel_radius, .5),
    "neuroscience": (gaussian_filter, neuro_activation_function, 1, 1),
}


# data
filedir = '/home/shared/MOD/frames.npy'
maskdir = '/home/shared/MOD/masks.npy'
data = np.load(filedir)
targets = torch.tensor(np.load(maskdir)).float()
#dataset = MODDataset(filedir, maskdir)
#data_path = '/home/shared/datasets/even_bdd/426x240/dvs_yolo_dataset/'
#bdd_dataset = bdd_dataset(data_path+'val')
# noinspection duplicatedcode
#loader = dataloader(bdd_dataset)
#loader = DataLoader(dataset)


# procedure
results = [ f"results from {str(dt.now().ctime())} \n", "parameters: \n",
                      f"\t width: {width} \n", f"\t height: {height} \n", f"\t fps: {fps} \n",
                      f"\t center_kernel_radius: {center_kernel_radius} \n",
                      f"\t surround_kernel_radius: {surround_kernel_radius} \n"]

# iterate through different types of kernel
for kernel, (kernel_func, activ_func, stride, threshold) in kernels.items():

    print(kernel)
    print(kernel_func.__name__, "\n")

    params_file = os.path.join(params_dir)

    #print(f"{data.shape} {targets.shape} {file_name} {class_name}")
    #data = file['spike_tensor']
    #targets = file['full_mask_tensor']
    # Data shape is (1, time steps, 180, 240)
    #data, targets = data.squeeze(), targets.squeeze()  # remove batch dim

    # if there is not a folder for the class of results, make one
    class_name = "room"
    file_name = "seq_room1_obj1"
    if not os.path.exists(os.path.join(save_dir, class_name)):
        os.mkdir(os.path.join(save_dir, class_name))

    # if there is not a folder for the results, make one
    if not os.path.exists(os.path.join(save_dir, class_name, file_name)):
        os.mkdir(os.path.join(save_dir, class_name, file_name))

    if True:
        # run OMS and make the video output (set to False if you don't want to make the video)
        oms(
            params_dir=params_file + '.yaml',
            save_dir=os.path.join(os.getcwd(), 'results/', class_name, file_name + "/", file_name + f"_{kernel}"),
            dvs_frames=data,
            compile_video=True,
            num_frames=data.shape[0],
            filter_func=kernel_func,
            activation_func=activ_func,
            fps=fps,
            width=width,
            height=height,
            center_rad=center_kernel_radius,
            surround_rad=surround_kernel_radius,
            oms_threshold=threshold,
            stride=stride,  # figure out what hardware stride should do
        )

    # run OMS
    oms_frames = oms(
        params_dir=params_file + '.yaml',
        save_dir=os.path.join(os.getcwd(), 'results/', class_name, file_name + "/", file_name + f"_{kernel}"),
        dvs_frames=data,
        compile_video=False,
        num_frames=data.shape[0],
        filter_func=kernel_func,
        activation_func=activ_func,
        fps=fps,
        width=width,
        height=height,
        center_rad=center_kernel_radius,
        surround_rad=surround_kernel_radius,
        oms_threshold=threshold,
        stride=stride,  # figure out what hardware stride should do
        compile_to_h5py=True,
    )

    # METRICS CALCULATION

    num_frames = oms_frames.shape[0]  # number of frames in the video
    ssim_hist = []
    spike_rate_hist = []
    phash_hist = []

    # iterate through the frames and calculate the metrics comparing the oms frames to the target frames
    for ind in range(num_frames):
        if stride != 1:
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
        phash = phash_comp(oms_frames[ind], target.squeeze())

        ssim_hist.append(acc)
        spike_rate_hist.append(spike_rate)
        phash_hist.append(phash)

    # calculate the average of the metrics
    ssim_avg = np.mean(np.stack(ssim_hist))
    spike_rate_avg = np.sum(np.stack(spike_rate_hist)) / len(spike_rate_hist)  # average the spike rates
    phash_avg = np.mean(np.stack(phash_hist))

    print(f"SSIM average: {ssim_avg}")
    print(f"spike_rate: {spike_rate_avg} \n")

    # adding metrics to results list so that we can record them at the end
    results.append(f"{kernel}, {kernel_func.__name__}, {activ_func.__name__}:  \n")
    results.append(f"\t output_shape {oms_frames.shape} \n")
    results.append(f"\t SSIM: {str(ssim_avg)} \n")
    results.append(f"\t spike_rate {spike_rate_avg} \n")
    results.append(f"\t Perceptual Hash (PHASH) Comparison: {phash_avg} \n")
    results.append(f"\t stride: {stride} \n")
    results.append(f"\t threshold: {threshold} \n")


# record the metrics to .txt file
with open(os.path.join(os.getcwd(), 'results/', f"metrics.txt"), "w") as f:
    f.writelines(results)

# VISUALIZATION

# dataset.plot_events(fps=fps, save_dir=os.path.join(os.getcwd(), 'results/', f"dvs_data.mp4"))

# SAVING

# put all the results file under one file in the save_dir with the date
experiment_path = os.path.join(save_dir, f"{str(dt.now().ctime()).replace(' ', '_')}")
os.mkdir(experiment_path)


# all the files generated in the experiment are moved under the experiment folder
for path in os.listdir(save_dir):
    if "box" in path or "floor" in path or "table" in path or "wall" in path or "fast" in path:
        shutil.move(os.path.join(save_dir, path), experiment_path)
