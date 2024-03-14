import numpy as np
import os
import torch
from tqdm import tqdm
import zipfile

from matplotlib import pyplot as plt

from datasets.evimo_dataset import EVIMODataset
from datasets.mod_dataset import MODDataset
import dataloader.base as base

from utils.masks_to_boxes import masks_to_boxes, detection_rate
from utils.gpu import moveToGPUDevice
from utils.rbase import RBase

from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import default_collate
import torch.nn as nn 
from PIL import Image
import slayerpytorch as snn

"""
Runner class was taken and modified from https://github.com/prgumd/SpikeMS/blob/main/runner.py
"""
def my_collate(batch):
    batch = list(filter (lambda x:x is not None, batch))
    if not batch:
        return None

    return default_collate(batch)

class Runner(RBase):

    def __init__(self, oms_datadir, masks_datadir,
                 crop, maxBackgroundRatio, datasetType,
                    checkpoint, modeltype, 
                    log_config, general_config, 
                    maskDir, incrementalPercent,
                    saveImages, saveImageInterval, imageDir, imageLabel=""):
        super().__init__(masks_datadir, log_config, general_config)
       
        self.output_dir = self.log_config.getOutDir() 
        self.genconfigs = snn.params(general_config)
        self.checkpoint = checkpoint
        self.modeltype = modeltype
        self.maskDir = maskDir
        self.incrementalPercent = incrementalPercent
        self.saveImages = True
        self.saveImageInterval = saveImageInterval
        # TODO
        # Change to the path where the OMS and groundtruth masks .png files are
        self.imageDir = oms_datadir
        self.imageLabel = imageLabel
        if(datasetType == "EVIMO"):
            database = EVIMODataset(oms_datadir, masks_datadir, 100, False)
            print("EVIMO used")
        elif(datasetType == "MOD"):
            database = MODDataset(os.path.join(masks_datadir, 'room1obj1-001.hdf5'), os.path.join(masks_datadir,'seq_room1_obj1/masks/'), os.path.join(oms_datadir,'room/seq_room1_obj1','seq_room1_obj1_neuroscience.h5'), 100)
            print("MOD used")
        else:
            raise Exception("Only EVIMO or MOD datasets with hdf5 format generated by preprocessing scripts handled with this code.")

        num_workers = self.genconfigs['hardware']['readerThreads']
        batch_size = self.genconfigs['batchsize']
        self.loader = torch.utils.data.DataLoader(database, batch_size=8, shuffle=False, num_workers=8, collate_fn=my_collate, drop_last = False)
        self.tb_writer = SummaryWriter(self.output_dir)    

    def test(self):
        #self._loadNetFromCheckpoint(self.checkpoint, self.modeltype)
        total_dr = []
        total_input_IOU = 0 
        scalar_i = 0
        tot_frames = 0
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        if self.saveImages and not os.path.exists(self.imageDir):
                os.mkdir(self.imageDir)
                
        with torch.no_grad():
            for i, data in enumerate(self.loader):
                if (data == None):
                    continue
                data = moveToGPUDevice(data, device, None)
                oms_spikes_input = data['oms_spike_tensor']
                oms_spikes_masked = data['oms_mask']
                spikes_input = data['dvs_spike_tensor']

                ioucriterion = snn.loss(self.genconfigs).to(self.device) 

                #calculate metrics
                

                #spike_pred_2D = torch.sum(spike_pred, axis = (1,4))
                oms_mask_2D = torch.sum(oms_spikes_masked, axis = (0,1,4))
                oms_spike_2D = torch.sum(oms_spikes_input, axis=(0,1,4))
                input_dr = ioucriterion.getIOU(oms_spike_2D, oms_mask_2D, True)
                #oms_mask_2D = torch.sum(oms_spikes_masked, axis = (0))
                #oms_spike_2D = torch.sum(oms_spikes_input, axis=(0))
                input_iou = ioucriterion.getIOU(oms_spike_2D, oms_mask_2D)
                print("OMS INPUT IoU", input_iou)
                print("DR", input_dr)
                print("-------------------------------------------------")
                total_dr.append(input_dr)
                tot_frames += spikes_input.shape[0]
                total_input_IOU += input_iou*spikes_input.shape[0]
                scalar_i += 1

        #print("save to: ", self.output_dir)

        if self.saveImages:
            print("saving images to", os.getcwd(), self.imageDir)

        print("mean OMS IoU", total_input_IOU/tot_frames)
        print("DR (%)", total_dr.mean())
        #print("mean DVS IoU for {} batches of frames".format(tot_frames), total_IOU/tot_frames)