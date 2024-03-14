import os
import numpy as np
import torch
import tonic


class MODDataset(tonic.Dataset):
    def __init__(self, data_dir):
        super().__init__("./")

        self.data_dir = data_dir  # should be an absolute path
        self.frames = os.path.join(data_dir, 'frames.npy')
        self.masks = os.path.join(data_dir, 'masks.npy')

    def __len__(self):
        #Only one validation sequence
        return 1

    def __getitem__(self, idx):

        # get the file name
        f_name ="seq_room1_obj1"

        # get the class name
        c_name = "room"

        # load the data and target from the paths
        frames = np.load(self.frames)
        #print("FRAMES", frames.shape)

        masks = np.load(self.masks)
        masks = torch.tensor(masks).float()

        return frames, masks, f_name, c_name