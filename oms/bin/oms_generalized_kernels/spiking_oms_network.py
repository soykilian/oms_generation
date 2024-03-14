import tonic
import tonic.transforms as transforms
import torch
import torchvision
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OMSv3Network(nn.Module):

    def __init__(self, beta=.5, spike_grad=surrogate.atan(), batch_size=1):
        super().__init__()

        # Initialize layers
        self.conv1 = nn.Conv2d(2, 1, 11, padding="same", dtype=torch.float32)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=.5)

        # self.fc1 = nn.Linear(43200, 180*240)
        # self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        # with torch.no_grad():
        #     self.conv1.weight = torch.nn.Parameter(
        #         torch.from_numpy(
        #             np.concatenate(
        #                 (construct_circ_filter(5), construct_circ_filter(5))
        #             ).reshape(1, 2, 11, 11)
        #         ).float()
        #     )
        torch.nn.init.ones_(self.conv1.weight)

        self.batch_size = batch_size

    def forward(self, x):

        # print(x.shape)
        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        # mem2 = self.lif2.init_leaky()

        # cur1 = F.max_pool2d(self.conv1(x), 2)
        cur1 = self.conv1(x)

        # print(cur1.max())

        # print(cur1.shape)
        # print(cur1.max())

        spk1, mem1 = self.lif1(cur1, mem1)

        # if spk1.max() == 0:
        #     print("spk1 is all 0 you may want to terminate training")
        #     print(self.conv1.weight.VSim_dataset.numpy())
        # assert spk1.max() != 0, "spk1 is all 0"

        # cur2 = self.fc1(spk1.view(self.batch_size, -1))
        # cur2 = spk1.view(self.batch_size, -1)
        #
        # print(cur2.shape)
        # print(cur2.max())  # ensure that the max is not 0
        #
        # spk2, mem2 = self.lif2(cur2, mem2)  # this layer is somehow setting all VSim_dataset to 0
        #
        # print(spk2)  # ensure that the max is not 0

        # print(spk2.shape)

        return spk1, mem1

    @property
    def conv_filters(self):
        return self.conv1.weight.data.numpy()

    @staticmethod
    def oms_loss(
       x: torch.Tensor,
       y: torch.Tensor,
    ):
        """
        Loss function for OMS model
        :param x: input tensor
        :param y: target tensor
        :return: loss
        """

        assert x.shape == y.shape, f"Input: {x.shape} and target: {y.shape} must have the same shape"

        # Dice loss
        # x = x.contiguous()
        # y = y.contiguous()
        # intersection = (x * y).sum(dim=2).sum(dim=2)
        # loss = (1 - ((2. * intersection + 1.) / (x.sum(dim=2).sum(dim=2) + y.sum(dim=2).sum(dim=2) + 1.)))

        # Calculate return loss
        loss = F.l1_loss(x, y, reduction='mean')
        # loss = SF.ce_count_loss()(x, y)

        # loss = nn.CrossEntropyLoss()(x, y)
        # loss = nn.L1Loss()(x, y)

        return loss
