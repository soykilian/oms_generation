import os
import h5py
import cv2
from PIL import Image
from pathlib import Path

import numpy as np
import random
import tonic
import torch
from matplotlib import pyplot as plt
from snntorch import spikegen
import h5py
import re
import sys
from tqdm import tqdm
#sys.path.append("/home/mavi/iris")
#from version_2.bin.utils.plot_events import plot_animation

class EVIMODataset(tonic.Dataset):
    def __init__(self, data_dir, num_steps, dvs=False):
        super().__init__("./")

        self.oms_file = os.path.join("/scratch/mclerico/dataset/eval_96",data_dir)
        self.oms_file += "_frames"
        self.dvs_file = os.path.join("/scratch/mclerico/dataset/eval",data_dir)
        self.height = 260
        self.width = 346
        self.num_steps = num_steps
        self.maxBackgroundRatio = 1.5
        self.dvs = dvs
        self.oms_file =os.path.join(self.oms_file ,  self.oms_file.split('/')[-1] )
        self.oms_file+= '_neuroscience.h5'
        with h5py.File(self.oms_file, 'r') as f:
            self.len = f['oms_frames'][:].shape[0]


    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        path = self.dvs_file
        file_path = path + '/' +path.split('/')[-1]+'.hdf5'
        data = h5py.File(file_path, 'r')
        with h5py.File(self.oms_file, 'r') as f:
            oms_data = f['oms_frames'][:]
        #frames = torch.zeros((self.batch_size, 2, 260, 346,self.num_steps))
        #masks = torch.zeros((self.batch_size, 2, 260, 346,self.num_steps))
        frame, mask = self.generate_events(path, data, idx)
        if frame is None or mask is None :
            return None
        #oms_mask = torch.logical_and(torch.Tensor(oms_data[idx]), torch.Tensor(mask))
        
        oms_masked_frame = torch.empty_like(mask)
        oms_spike_mask = torch.empty_like(frame)
        for k in range(self.num_steps):
            oms_masked_frame[0,:,:,k]= torch.logical_and(frame[0, :,:,k], torch.Tensor(oms_data[idx]))
            oms_masked_frame[1, :, :, k]= torch.logical_and(frame[1,:,:,k], torch.Tensor(oms_data[idx]))
            oms_spike_mask[1,:,:,k] = torch.logical_and(oms_masked_frame[1,:,:,k], mask[1,:,:,k])
            oms_spike_mask[0,:,:,k] = torch.logical_and(oms_masked_frame[0,:,:,k], mask[0,:,:,k])
        """
        #frame_sum_2 = torch.sum(masked_frame, axis=(0,3))
        #print("MASKED FRAME SUM", torch.unique(frame_sum_2))
        #plt.figure()
        #plt.imshow(frame_sum_2) 
        #plt.savefig("./masked_frame" + str(idx)+".png")
        """
        return  {"oms_spike_tensor": oms_data[idx],
                 "oms_mask":oms_spike_mask,
                 "dvs_spike_tensor": frame,
                 "dvs_mask" : mask,
                 }

    def process_data(self, save_dir: str, class_dir: str):
        """
        Takes each .npz file and parses it. From the file we take out the frames and masks and save those to .npy files
        """
        save_dir = os.path.join(save_dir)
        files = os.listdir(class_dir)

        # Iterate through each file
        i = 0
        for file in files:
            frames, masks = self.load_data(os.path.join(class_dir,
                file), file)
            self.dvs_tensors[i:masks.shape[0]]= frames
            self.targets[i+masks.shape[0]] = masks
            file = Path(file)
            i+= masks.shape[0]
            del frames
            del masks


    @staticmethod
    def get_event_idxs(data, index, k=1):
        return data['events_idx'][index], data['events_idx'][index+k] - 1

    def generate_events(self, path, data, index, k=1):
        timeframes = data['timeframes']
        idx0, idx1 = EVIMODataset.get_event_idxs(data,index, 1)
        frame_events = data['events'][idx0:idx1]
        start_t = data['timeframes'][index][0]
        end_t = data['timeframes'][index+k-1][1]
        #t_window =  frame_events[-1,0] - frame_events[0,0]
        #dtype = np.dtype([("x", int), ("y", int), ("t", float), ("p", int)])
        spike_dvs_tensor = torch.zeros((2,self.height,
            self.width,self.num_steps))
        xs = frame_events[:, 1]
        ys = frame_events[:, 2]
        ps = frame_events[:, 3]
        ts = frame_events[:, 0]
        ts = (self.num_steps -1)*(ts-start_t)/(end_t -start_t)
        spike_dvs_tensor[ps, ys, xs, ts]=1
        spike_tensor_sum_plot = torch.sum(spike_dvs_tensor, axis=(0,3))
        full_mask_tensor = torch.zeros((2, self.height, self.width, self.num_steps))   

        for i in range(0, k):
            curr_start = int((self.num_steps) * (i)/k)
            curr_end = int((self.num_steps) * (i+1)/k)
            currfile_nm = path + "/img/" f"depth_mask_{int(timeframes[index,2])}.png"
            fullmask = np.asarray(Image.open(currfile_nm))[:,:,0]
            fullmask = fullmask.astype(bool).astype(float)

            kernel = np.ones((5, 5), 'uint8')
            fullmask1 = cv2.dilate(fullmask, kernel, iterations=1)

            fullmask = np.expand_dims(fullmask1, axis=(0,3))
            tiled = torch.from_numpy(np.tile(fullmask, (2,1,1, curr_end-curr_start)))      

            full_mask_tensor[:,:,:,curr_start:curr_end] = tiled  

        masked_spike_tensor = ((spike_dvs_tensor + full_mask_tensor) > 1).float()
        background_spikes = (spike_dvs_tensor + torch.logical_not(masked_spike_tensor).float())>1

        if torch.sum(background_spikes)/torch.sum(masked_spike_tensor) > self.maxBackgroundRatio:
            return None, None
        #return spike_dvs_tensor, fullmask1
        return spike_dvs_tensor, masked_spike_tensor

