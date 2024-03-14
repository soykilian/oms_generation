import torch
import math
import numpy as np
import torch
import torch.nn as nn
from .slayer import spikeLayer
from PIL import Image
import os

class spikeLoss(torch.nn.Module):   
    '''
    This class defines different spike based loss modules that can be used to optimize the SNN.

    NOTE: By default, this class uses the spike kernels from ``slayer.spikeLayer`` (``snn.layer``).
    In some cases, you may want to explicitly use different spike kernels, for e.g. ``slayerLoihi.spikeLayer`` (``snn.loihi``).
    In that scenario, you can explicitly pass the class name: ``slayerClass=snn.loihi`` 

    Usage:

    >>> error = spikeLoss.spikeLoss(networkDescriptor)
    >>> error = spikeLoss.spikeLoss(errorDescriptor, neuronDesc, simulationDesc)
    >>> error = spikeLoss.spikeLoss(netParams, slayerClass=slayerLoihi.spikeLayer)
    '''
    def __init__(self, errorDescriptor, neuronDesc, simulationDesc, slayerClass=spikeLayer):
        super(spikeLoss, self).__init__()
        self.neuron = neuronDesc
        self.simulation = simulationDesc
        self.errorDescriptor = errorDescriptor
        self.slayer = slayerClass(self.neuron, self.simulation)
        
    def __init__(self, networkDescriptor, slayerClass=spikeLayer):
        super(spikeLoss, self).__init__()
        self.neuron = networkDescriptor['neuron']
        self.simulation = networkDescriptor['simulation']
        self.slayer = slayerClass(self.neuron, self.simulation)
        self.sigmoid = nn.Sigmoid()

    def crop_like(input, target):
        
        if input.size()[2:] == target.size()[2:]:
            return input
        else:
            return input[:, :target.size(1), :target.size(2), :target.size(3)]     

    def spikeTime(self, spikeOut, spikeDesired):
        '''
        Calculates spike loss based on spike time.
        The loss is similar to van Rossum distance between output and desired spike train.

        .. math::

            E = \int_0^T \\left( \\varepsilon * (output -desired) \\right)(t)^2\\ \\text{d}t 

        Arguments:
            * ``spikeOut`` (``torch.tensor``): spike tensor
            * ``spikeDesired`` (``torch.tensor``): desired spike tensor

        Usage:

        >>> loss = error.spikeTime(spikeOut, spikeDes)
        '''
        # Tested with autograd, it works
        # assert self.errorDescriptor['type'] == 'SpikeTime', "Error type is not SpikeTime"
        # error = self.psp(spikeOut - spikeDesired) 
        diff = spikeOut -spikeDesired
        #rint(diff.unique())
        error = self.slayer.psp(diff) 
        #rint(error.unique())
        #print(self.simulation['Ts'])
        return 1/2 * torch.sum(diff**2) * self.simulation['Ts']

    def MembraneSpikeTime(self, mask, spikeOut, spikeDesired, membraneOut, theta=0.22):
        XORmask = torch.logical_xor(mask, spikeOut)

        error = torch.mul(XORmask, membraneOut - theta * spikeDesired) 
        return 1/2 * torch.sum(error**2) * self.simulation['Ts']


    def numSpikes(self, spikeOut, gt, numSpikesScale=1):
        '''
        Calculates spike loss based on number of spikes within a `target region`.
        The `target region` and `desired spike count` is specified in ``error.errorDescriptor['tgtSpikeRegion']``
        Any spikes outside the target region are penalized with ``error.spikeTime`` loss..

        .. math::
            e(t) &= 
            \\begin{cases}
            \\frac{acutalSpikeCount - desiredSpikeCount}{targetRegionLength} & \\text{for }t \in targetRegion\\\\
            \\left(\\varepsilon * (output - desired)\\right)(t) & \\text{otherwise}
            \\end{cases}
            
            E &= \\int_0^T e(t)^2 \\text{d}t

        Arguments:
            * ``spikeOut`` (``torch.tensor``): spike tensor
            * ``desiredClass`` (``torch.tensor``): one-hot encoded desired class tensor. Time dimension should be 1 and rest of the tensor dimensions should be same as ``spikeOut``.

        Usage:

        >>> loss = error.numSpikes(spikeOut, target)
        '''
        # Tested with autograd, it works
        # Tested with autograd, it works
        # assert self.errorDescriptor['type'] == 'NumSpikes', "Error type is not NumSpikes"
        # desiredClass should be one-hot tensor with 5th dimension 1
        tgtSpikeRegion = self.errorDescriptor['tgtSpikeRegion']
        tgtSpikeCount  = self.errorDescriptor['tgtSpikeCount']
        startID = np.rint( tgtSpikeRegion['start'] / self.simulation['Ts'] ).astype(int)
        stopID  = np.rint( tgtSpikeRegion['stop' ] / self.simulation['Ts'] ).astype(int)
        
        actualSpikes = torch.sum(spikeOut[...,startID:stopID], 4, keepdim=True).cpu().detach().numpy() * self.simulation['Ts']
        desiredSpikes = gt[...,startID:stopID]
        errorSpikeCount = (actualSpikes - desiredSpikes) / (stopID - startID) * numSpikesScale
        targetRegion = np.zeros(spikeOut.shape)
        targetRegion[:,:,:,:,startID:stopID] = 1
        spikeDesired = torch.FloatTensor(targetRegion * spikeOut.cpu().data.numpy()).to(spikeOut.device)
        
        error = self.slayer.psp(spikeOut - spikeDesired)

        error += torch.FloatTensor(errorSpikeCount * targetRegion).to(spikeOut.device)

        return 1/2 * torch.sum(error**2) * self.simulation['Ts']

    def BCE(self, spikeOut, gt_fg):
        '''
        Calculates the binary cross entropy of the output and desired spike train.'''
        # clamp the bce value to 100 per pixel? avoid nan
        # See if autograd works
        # assert self.errorDescriptor['type'] == 'BCE', "Error type is not BCE"
        gt_bg = torch.empty_like(gt_fg, dtype=torch.int8)
        torch.logical_not(gt_fg, out= gt_bg)
        pred_fg = torch.sum(spikeOut, axis=4).to(float)
        spike_bg = torch.empty_like(spikeOut, dtype=torch.int8)
        torch.logical_not(spikeOut, out= spike_bg)
        pred_bg = torch.sum(spike_bg, axis=4).to(float)
        loss = torch.nn.BCEWithLogitsLoss()
        bce_loss = loss(pred_fg, pred_bg)
        return -bce_loss
        """
        #fg_lprobs = pred_fg/50
        #bg_lprobs = pred_bg/50 

        #print(fg_lprobs.unique())
        #print(bg_lprobs.unique())
        logs = torch.log(fg_lprobs)
        if True in torch.isinf(torch.log(fg_lprobs)):
            indexes = torch.where(torch.isinf(logs))
            print(gt_fg[indexes].unique())
        bce_loss = -(gt_fg * torch.log(fg_lprobs) + gt_bg * torch.log(bg_lprobs))
        bce_loss[torch.isnan(bce_loss)] = 0
        bce_loss[torch.isinf(bce_loss)] = 100
        print(bce_loss.unique())
        for i in range(bce_loss.shape[0]):
            plt.figure()
            plt.subplot(1,3,1)
            plt.imshow(bce_loss[i,0,:,:].cpu().detach().numpy())
            plt.subplot(1,3,2)
            plt.imshow(gt_fg[i,0,:,:].cpu().detach().numpy())
            plt.subplot(1,3,3)
            plt.imshow(pred_fg[i,0,:,:].cpu().detach().numpy())
        print(torch.mean(bce_loss))
        return  torch.mean(bce_loss)
        """
    def MSE(self, spikeOut, spikeDesired):

        return torch.mean(torch.sqrt(torch.sum((spikeOut - spikeDesired) ** 2, dim=1)))
    
    def getIOU(self, spike_pred, spike_gt, dr= False):
        spike_pred = spike_pred.detach().cpu().numpy()
        spike_gt = spike_gt.detach().cpu().numpy()
        spike_gt[spike_gt>1] = 1
        spike_pred[spike_pred>1] = 1
        intersection = np.sum(np.logical_and(spike_pred, spike_gt))
        union = np.sum(np.logical_or(spike_pred, spike_gt))
        if dr:
            #intersection = np.sum(np.logical_and(spike_pred, spike_gt))
            false_positive = np.sum(np.logical_and(np.logical_not(spike_gt), spike_pred))
            return 1 if intersection > 0.5*np.sum(spike_gt) and intersection > false_positive else 0
        return intersection/union
