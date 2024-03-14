import tonic
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset
import torch
from snntorch import surrogate
import snntorch.functional.reg as reg
import numpy as np
import os
import time
from datetime import datetime as dt
from torchsummary import summary

from oms_generalized_kernels.spiking_trainer import spiking_train_loop
from oms_generalized_kernels.trainer import train_loop
from oms_generalized_kernels.spiking_oms_network import OMSv3Network
from oms_generalized_kernels.event_dataset import IrisDVSDataset


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
results_dir = os.path.join(ROOT_DIR, 'results/')
model_dir = os.path.join(ROOT_DIR, 'models/')


# PARAMETERS

# Training/Data Params
num_epochs = 25
max_iters_per_epoch = 5
batch_size = 1
shuffle = True
sensor_size = (240, 180, 2)  # Davis240

# Model Params
beta = .7  # leaky parameter
spike_grad = surrogate.atan()  # surrogate gradient for spiking neurons

# Optimizer Params
lr = 10e-4  # learning rate
betas = (0.9, 0.999)  # betas for Adam optimizer

# DATA

# frame_transform = tonic.transforms.Compose([
#     # tonic.transforms.Denoise(filter_time=10000),
#     # tonic.transforms.ToFrame(
#     #     sensor_size=sensor_size,
#     #     time_window=1000, # distributes events into time bins of 1000 microseconds each
#     # )
# ])

# augmentation_transforms = tonic.transforms.Compose([
#     torch.fromnumpy,
#     # APPLY Augmentation if I want
# ])

train_dataset = IrisDVSDataset(
    sensor_size=sensor_size,
    transform=None,
    data_dir=results_dir
)

# Cached Datasets using tonic.DiskCachedDataset for speed
# cached_train_dataset = DiskCachedDataset(train_dataset, cache_path='./cache/iris/train')
# cached_test_dataset = None

# Data is shuffled every epoch which is useful for networks that get hung up on order like CNNs
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)  # Pytorch Dataloader
test_loader = None  # Pytorch Dataloader

# NETWORK
device = torch.device("cpu")
net = OMSv3Network(beta=beta, spike_grad=spike_grad, batch_size=batch_size)

# OPTIMIZERS
optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=betas, weight_decay=.05)  # weight_decay=None

# TRAINING
# loss_hist, acc_hist = train_loop(
#     num_epochs=num_epochs,
#     num_iterations=max_iters_per_epoch,
#     net=net,
#     optimizer=optimizer,
#     train_loader=train_loader,
#     batch_size=batch_size,
# )

loss_hist, acc_hist = spiking_train_loop(
    num_epochs=num_epochs,
    num_iterations=max_iters_per_epoch,
    net=net,
    optimizer=optimizer,
    regularization_fn=None,
    train_loader=train_loader,
    batch_size=batch_size,
)

# TESTING/EVALUATION

# TODO: Implement testing/evaluation

# VISUALIZATION

# TODO: Implement visualization
summary(net, (1, 2, 180, 240), batch_dim=None)

# SAVE MODEL
if os.path.exists(os.path.join(model_dir, f"{str(dt.now().ctime()).replace(' ', '_')}.pth")):
    time.sleep(200)
torch.save(net.state_dict(), model_dir + f"{str(dt.now().ctime()).replace(' ', '_')}.pth")

