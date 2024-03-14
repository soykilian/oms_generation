from typing import Callable

import tonic
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset
import torch
from snntorch import utils
from tqdm import tqdm
import numpy as np
import os

from oms_generalized_kernels.spiking_oms_network import OMSv3Network


def spiking_forward_pass(net, optimizer, data, targets, loss_fn, regularization_fn: Callable = None):
    # VSim_dataset should be shape (num time steps, batch_size, channels, height, width)
    # net.to(device)
    # VSim_dataset.to(device)

    batch_size = data.shape[0]
    num_time_steps = data.shape[1]

    data = data.reshape(num_time_steps, batch_size, 2, 180, 240)
    targets = targets.reshape(num_time_steps, batch_size, 1, 180, 240)

    spk_rec = []
    loss_trunc = 0
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in tqdm(range(data.shape[0])):  # VSim_dataset.size(0) = number of time steps
        # print(VSim_dataset[step].shape)
        spk_out, _ = net(data[step] / data[step].max())  # norm VSim_dataset to 1
        spk_rec.append(spk_out.view(batch_size, 1, 180, 240))

        # loss calculation for each time step
        loss = loss_fn(spk_out, targets[step])

        if regularization_fn:
            loss += regularization_fn(spk_out)

        loss_trunc += loss.item()

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_avg = loss_trunc / num_time_steps

    return loss_avg  # torch.stack(spk_rec)


def spiking_train_loop(
        num_epochs: int,
        num_iterations: int,
        net: torch.nn,
        optimizer: torch.optim,
        regularization_fn: Callable = None,
        train_loader: DataLoader = None,
        batch_size: int = 1,
):
    """
    Train loop for OMS model.
    :param num_epochs: number of epochs to train
    :param num_iterations: number of iterations to train
    :param net: network to train
    :param optimizer: optimizer to use
    :param regularization_fn: regularization function to use (can only use snn.function.reg.l1_rate_sparsity)
    :param train_loader: training VSim_dataset loader
    :param batch_size: batch size

    :return: loss_hist, acc_hist
    """
    loss_hist = []
    acc_hist = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        for i, (data, targets) in enumerate(iter(train_loader)):
            # VSim_dataset = VSim_dataset.to(device)
            # targets = targets.to(device)

            net.train()

            # these two lines add together the polarity channels explore how to interpret without this later
            # spk_rec = spk_rec[:4500, :, 0] + spk_rec[:4500, :, 1]
            # print(targets.shape)
            targets = targets[:, :4500, 0] + targets[:, :4500, 1]

            loss_avg = spiking_forward_pass(
                net, optimizer, data[:, :4500].float(), targets.float(), OMSv3Network.oms_loss, regularization_fn
            )

            del data
            # loss_val = OMSv3Network.oms_loss(spk_rec.view(batch_size, spk_rec.shape[0], 2, 180, 240), targets.float())

            # loss_val = OMSv3Network.oms_loss(
            #     spk_rec.view(batch_size, spk_rec.shape[0], 180, 240),
            #     targets.float().view(batch_size, spk_rec.shape[0], 180, 240),
            # )
            # loss_val = F.mse_loss(
            #     spk_rec.view(batch_size, spk_rec.shape[0], 180, 240),
            #     targets.float().view(batch_size, spk_rec.shape[0], 180, 240),
            #     reduction="sum"
            # )

            # Gradient calculation + weight update
            # optimizer.zero_grad()
            # loss_val.backward()
            # optimizer.step()

            # Store loss history for future plotting

            loss_hist.append(loss_avg)

            with torch.no_grad():
                params = net.conv_filters[0][0], net.conv_filters[0][1]
                print(net.conv_filters[0][0], net.conv_filters[0][1])
                if params[0].max() <= 0 and params[1].max() <= 0:
                    print("spk1 is all 0 you may want to terminate training")

            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_avg:.4f}")

            # acc = SF.accuracy_rate(spk_rec, targets)
            # acc_hist.append(acc)
            # print(f"Accuracy: {acc * 100:.2f}%\n")

            # clear memory
            del targets, loss_avg,  # acc, spk_rec,

            # training loop breaks after 6 iterations
            if i == num_iterations:
                break

    return loss_hist, acc_hist
