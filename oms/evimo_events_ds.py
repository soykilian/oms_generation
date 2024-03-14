import os
import h5py
import cv2
from PIL import Image
from pathlib import Path
from masks_to_boxes import masks_to_boxes
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
from version_2.bin.utils.plot_events import plot_animation

class EVIMODataset(tonic.Dataset):
    """
    This class is used to load the EVIMO dataset from .npz files downloadable from
    https://better-flow.github.io/evimo/download_evimo.html.

    It will convert the .npz files to .npy files for lazy loading during training.
    to use the in code load this class into an object of type torch.utils.data.Dataset and pass it to a DataLoader.

    `data_dir` (absolute path) should take you to the directory containing the .npz files.

    The directory structure should look like this:

    dataset ==> *data_dir*
    ├── box
    │   ├── npz
    │   │   ├── seq_01.npz
    │   │   ├── seq_02.npz
    │   │   ├── seq_03.npz

    """

    def __init__(self, data_dir, num_steps, one_bbox: bool = True, batch_size:
            int = 32, target_size: int = 4, spike_target = True):
        super().__init__("./")

        self.data_dir = data_dir  # should be an absolute path
        if "train" in data_dir:
            self.no_files = 30
        else:
            self.no_files=21
        self.height = 260
        self.width = 346
        self.norm = [346,260,346,260]
        self.one_bbox = one_bbox
        self.num_steps = num_steps
        self.target_size = target_size
        self.spike_target = spike_target
        ## count number of frames
        self.dvs_event_files = []
        self.indexes = []
        self.curr_file = 0
        self.batch_size=batch_size
        # Iterate through each class of data
        for class_path in tqdm(os.listdir(self.data_dir)):
            class_dir = os.path.join(self.data_dir, class_path, "txt")
            save_dir = os.path.join(self.data_dir, class_path, "npy")
            files = os.listdir(class_dir)
            hdf5_files = [class_dir +'/'+ f for f in files]
            self.dvs_event_files+= hdf5_files
            # replace the strings with your training/testing file locations or pass as an argument
            #if self.data_dir is not None:
            #    print("Loading EVIMO dataset from .npz files, or from .npy files if already converted")
                # convert the .npz files to .npy files if they haven't been converted already
            #    self.process_data(save_dir=save_dir, class_dir=class_dir)
            # store the new data and target paths as lists
        #self.targets = torch.cat(self.targets, dim=0)
        print(self.no_files, len(self.dvs_event_files))


    def __len__(self):
        return int(54000/self.batch_size)


    def __getitem__(self, idx):
        path = self.dvs_event_files[self.curr_file]
        file_path = path + '/' +path.split('/')[-1]+'.hdf5'
        data = h5py.File(file_path, 'r')

        if data['events_idx'].shape[0] < self.batch_size:
            print("File: ", file_path, data['events_idx'].shape[0]-1)
        """
        if data['events_idx'].shape[0] < self.batch_size:
            path = self.dvs_event_files[self.curr_file++]
            file_path = path + '/' +path.split('/')[-1]+'.hdf5'
            data = h5py.File(file_path, 'r')
        """
        frames = torch.zeros((self.batch_size, 2, 260, 346,self.num_steps))
        masks = torch.zeros((self.batch_size, 2, 260, 346,self.num_steps))
        idxs = random.sample(range(0, data['events_idx'].shape[0]- 1), self.batch_size)
        for i, value in enumerate(idxs):
            frame, mask = self.generate_events(path, data, value)
            masks[i] =  mask
            frames[i] = frame
            del frame
            del mask
        if self.curr_file != self.no_files - 1:
            self.curr_file += 1
        else:
            self.curr_file = 0
        return  frames, masks

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
        t_window =  frame_events[-1,0] - frame_events[0,0]
        dtype = np.dtype([("x", int), ("y", int), ("t", float), ("p", int)])
        spike_dvs_tensor = torch.zeros((2,self.height,
            self.width,self.num_steps))
        xs = frame_events[:, 1]
        ys = frame_events[:, 2]
        ps = frame_events[:, 3]
        ts = frame_events[:, 0]
        ts = (self.num_steps -1)*(ts-start_t)/(end_t -start_t)
        spike_dvs_tensor[ps, ys, xs, ts]=1
        full_mask_tensor = torch.zeros((2, self.height, self.width, self.num_steps))   

        for i in range(0, k):
            curr_start = int((self.num_steps) * (i)/k)
            curr_end = int((self.num_steps) * (i+1)/k)
            currfile_nm = path + "/img/" f"depth_mask_{int(timeframes[index,2])}.png"
            fullmask = np.asarray(Image.open(currfile_nm))[:,:,0]
            fullmask = fullmask.astype(bool).astype(float)

            kernel = np.ones((5, 5), 'uint8')
            fullmask = cv2.dilate(fullmask, kernel, iterations=1)

            fullmask = np.expand_dims(fullmask, axis=(0,3))
            tiled = torch.from_numpy(np.tile(fullmask, (2,1,1, curr_end-curr_start)))      

            full_mask_tensor[:,:,:,curr_start:curr_end] = tiled  

        masked_spike_tensor = ((spike_dvs_tensor + full_mask_tensor) > 1).float()
        return spike_dvs_tensor, masked_spike_tensor


    def create_spikes(self,num_steps, p):
        zeros= torch.zeros(num_steps, dtype=torch.bool)
        random_indexes = random.sample(range(num_steps), int(p*num_steps))
        zeros[random_indexes]=1
        return zeros


    def parse_target(self, mask):
        obj_ids = torch.unique(mask)[1:]
        masks = mask == obj_ids[:, None, None]
        boxes = masks_to_boxes(masks)
        size = self.target_size
        if self.one_bbox:
            targets = torch.zeros((1,4), dtype=torch.uint8)
            if boxes.shape[0] !=0:
                new_box = [x/div for x,div in zip(boxes[0],self.norm)]
                targets = torch.Tensor([int(n*self.num_steps) for n in new_box])
            return targets
        if not self.spike_target:
            targets = torch.zeros(1,size)
            for j, bbox in enumerate(boxes):
                if bbox.any():
                    idx = 0
                    if size == 15:
                        targets[:, j*int(size/3)] = 1
                        idx = 1
                    targets[:, j*int(size/3)+idx:(j+1)*int(size/3)] = torch.tensor(bbox)
        else:
            targets = torch.zeros(1, self.num_steps, size)
            for j, bbox in enumerate(boxes):
                if bbox.any():
                    idx = 0
                    new_box = [x/div for x,div in zip(bbox,self.norm)]
                    if size == 15:
                        targets[:, :, j*int(size/3)] = torch.ones(self.num_steps)
                        idx = 1
                    targets[:, :, j*int(size/3)+idx:(j+1)*int(size/3)] = torch.stack([self.create_spikes(self.num_steps, n) for n in new_box], dim=1)
        return targets


    def get_mask(self, file: str):
        mask_img = Image.open(file)
        mask_npy = np.array(mask_img)[:,:,0]
        mask_tensor =  torch.tensor(mask_npy, dtype=torch.uint8)
        del mask_img
        del mask_npy
        return self.parse_target(mask_tensor)
    


    def load_data(self, path: str, seq: str):
        """
        Loads the data from the .npz file and returns the frames and masks as numpy arrays.

        :param path: path to the .npz file
        :return: frames and masks as numpy arrays
        """
        assert os.path.isfile(path +'/'+seq + '.hdf5')
        data = h5py.File(path+'/'+ seq+ '.hdf5', 'r')
        #frames = torch.zeros((data['events_idx'].shape[0]- 1, 2, self.height, self.width, self.num_steps), dtype=torch.bool)
        #masks = torch.zeros((data['events_idx'].shape[0]- 1,self.num_steps, self.target_size), dtype=torch.bool)
        self.indexes.append(data['events_idx'].shape[0]- 1)
        #for i in tqdm(range(data['events_idx'].shape[0] - 1)):
            #mask_file = self.generate_dvs_frames(data, i)
            #mask = self.get_mask(path +'/img/'+mask_file)
            ##frames[i:i+1:, :,:,:,:] = frame
            #masks[i:i+1, :, :] = mask
            #del frame
            #del mask

        #print(f"Masks Shape: {masks.shape}")
        #return masks

    def plot_events(self, fps: int, save_dir: str, index: int = 0):
        """
        Plots the events as a video using code modified from tonic.utils.plot_animation. Now uses
        skvideo.io.FFmpegWriter to write the video, and you can manually set the fps.
        :param self: np.ndarray of events frames. Shape: (num_frames, height, width)
        :param fps: frames per second of the video
        :param save_dir: directory to save the video; MUST END IN ".mp4"
        :param index: index of the data to plot
        """
        events = np.load(self.dvs_events[index])

        animation = plot_animation(
            frames=events,
            fps=fps,
            save_path=save_dir
        )
