import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import h5py
import os
import pandas as pd
from PIL import Image
import tonic
import matplotlib.pyplot as plt



#filedir = "room1obj1-001.hdf5" # hdf5 file
#maskdir = "seq_room1_obj1/masks" 

class MODDatasetOMS(Dataset):
        def __init__(self, datafile: str, maskDir, crop=False, maxBackgroundRatio=1.5):
            self.datafile = datafile
            self.data = h5py.File(datafile, 'r')
            self.increment = 0.001
            self.k = 1
            with h5py.File(datafile, 'r') as f:
                self.events = f['events'][:]  # Assuming 'events' is the dataset containing event annotations
                self.images_idx = f['images_idx'][:]  # Assuming 'images_idx' contains indices corresponding to images
            self.maskDir = maskDir
            self.crop = crop
            self.maxBackgroundRatio = maxBackgroundRatio

            self.height = 346
            self.width = 260
            self.num_time_bins = 100 
            self.length = len(self.images_idx)-1  # Length of the datase
                
        def __len__(self):
            return self.length
        
        
        def get_event_idxs(self, idx):
             return self.data['images_idx'][idx], self.data['images_idx'][idx+1] - 1


        def get_start_stop(self, idx, k=1):
            print("Type of self.increment:", type(self.increment))
            print("Value of idx:", idx)
            print("Value of k:", k)
            return self.increment*(idx+1), self.increment*(idx + 1 + k)
        

        def __getitem__(self, idx):
            index0, index1 = self.get_event_idxs(idx)
            events = self.data['events'][index0:index1]
            dtype = np.dtype([("x", int), ("y", int), ("t", float), ("p", int)])
            events_np = np.zeros(events.shape[0], dtype=dtype)
            events_np["x"] = events[:, 1]
            events_np["y"] = events[:, 2]
            events_np["p"] = events[:, 3]
            events_np["t"] = events[:, 0]
            t_window =  events[-1,0] - events[0,0]
            transform = tonic.transforms.ToFrame(
            sensor_size=(346, 260, 2),
            time_window=t_window)
            frame = torch.tensor(transform(events_np), dtype=torch.float32).squeeze(0)
            #full_mask_tensor = torch.zeros((2, self.height, self.width, self.num_time_bins))   
            curr_masknm = os.path.join(self.maskDir, "mask_{:08d}.png".format(idx+1))
            fullmask = np.asarray(Image.open(curr_masknm))
            frame = torch.sum(frame, axis=0)  # combine all spike activity into single axis
            frame[frame > 0] = 1
            return frame, fullmask
            #return out  
            


#dataset = ClassifyDataset(filedir, maskdir)

# for i in range(len(dataset)):
#     if (i < 4997):
#         sample = dataset[i]
#         print(i)
#         data = {
#             'Sample': i + 1,
#             'Event': sample['event'],
#             'Image Index': sample['image_idx'],
#             'Masked Event Index': sample['masked_event_idx'],
#             'Masked Event': sample['masked_event']

#         }
#         datalist.append(data)

# df = pd.DataFrame(datalist)

# # Save DataFrame to Excel file
# excel_filename = 'output_data.xlsx'
# df.to_excel(excel_filename, index=False)

# print(f"Print statements saved to {excel_filename}")

# Keys show [events, images_idx, masked_event_idx, masked_events]


