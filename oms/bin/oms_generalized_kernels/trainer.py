from torch.utils.data import DataLoader
import torch
import torchvision
import torch.nn.functional as F
from snntorch import functional as SF
from tqdm import tqdm
import numpy as np
import os
from skimage.metrics import structural_similarity as SSIM
from torchmetrics.image import StructuralSimilarityIndexMeasure


def forward_pass(net, data):
    # VSim_dataset should be shape (num time steps, batch_size, height, width)

    batch_size = data.shape[0]
    num_time_steps = data.shape[1]

    data = data.reshape(num_time_steps, batch_size, 180, 240)

    out_rec = []

    for step in tqdm(range(data.shape[0])):  # VSim_dataset.size(0) = number of time steps
        # print(data[step].shape)
        out = net(data[step] / data[step].max())  # norm VSim_dataset to 1

        out[torch.where(out >= .85)] = 1
        out[torch.where(out < .85)] = 0

        out_rec.append(out.view(batch_size, 180, 240))

    return torch.stack(out_rec)


def ssim_loss(out, target):

    ssim = StructuralSimilarityIndexMeasure(data_range=1.)

    ssim_hist = ssim(out, target)

    return ssim_hist


def train_loop(
        num_epochs: int,
        num_iterations: int,
        net: torch.nn,
        optimizer: torch.optim,
        train_loader: DataLoader,
        batch_size: int = 1,
):
    """
    Train loop for OMS model.
    :param num_epochs: number of epochs to train
    :param num_iterations: number of iterations to train
    :param net: network to train
    :param optimizer: optimizer to use
    :param train_loader: training VSim_dataset loader
    :param batch_size: batch size

    :return: loss_hist, acc_hist
    """
    loss_hist = []
    acc_hist = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        for i, (data, targets) in enumerate(iter(train_loader)):
            # looks like v2e data.max and target.max is around 15-17

            # VSim_dataset = VSim_dataset.to(device)
            # targets = targets.to(device)

            # some issues where targets are a few frames different than data
            targets = targets[:, :560]
            data = data[:, :560]

            net.train()
            out = forward_pass(net, data.float())

            del data

            # these two lines add together the polarity channels explore how to interpret without this later
            targets = targets.reshape(targets.shape[1], targets.shape[0], 180, 240)

            targets = (targets - targets.min()) / targets.max()  # normalize to 1

            loss_val = 1 - ssim_loss(out, targets)

            # loss_val = F.l1_loss(
            #     out.view(batch_size, out.shape[0], 180, 240),
            #     targets.float().view(batch_size, targets.shape[0], 180, 240),
            # )
            # loss_val = F.mse_loss(
            #     spk_rec.view(batch_size, spk_rec.shape[0], 180, 240),
            #     targets.float().view(batch_size, spk_rec.shape[0], 180, 240),
            #     reduction="sum"
            # )

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

            # iterate through the frames and calculate the metrics comparing the oms frames to the target frames

            with torch.no_grad():
                ssim_hist = [ssim_loss(out, targets)]

                acc = np.mean(np.stack(ssim_hist))
                acc_hist.append(acc)
                print(f"Accuracy: {acc * 100:.2f}%\n")

                if epoch % 2 == 0:
                    for parameter in net.parameters():
                        if len(parameter.shape) > 2:
                            print(parameter)

            # clear memory
            del targets, out, loss_val, acc

            # training loop breaks after 6 iterations
            if i == num_iterations:
                break

    return loss_hist, acc_hist
