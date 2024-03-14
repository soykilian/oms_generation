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
import matplotlib.pyplot as plt



#filedir = "room1obj1-001.hdf5" # hdf5 file
#maskdir = "seq_room1_obj1/masks" 

class MODDataset(Dataset):
        def __init__(self, datafile: str, maskDir:str, omsDir:str,num_steps:int ,crop=False, maxBackgroundRatio=1.5):
            self.datafile = datafile
            self.data = h5py.File(datafile, 'r')
            self.increment = 0.001
            self.k = 1
            self.oms_file = omsDir
            with h5py.File(datafile, 'r') as f:
                self.events = f['events'][:]  # Assuming 'events' is the dataset containing event annotations
                self.masked_event_idx = f['masked_event_idx'][:]  # Assuming 'masked_event_idx' contains indices corresponding to masked events
                self.masked_events = f['masked_events'][:]  # Assuming 'masked_events' contains masked event annotations
            with h5py.File(self.oms_file, 'r') as f:
                 self.oms_frames = f['oms_frames'][:]
            self.maskDir = maskDir
            self.crop = crop
            self.maxBackgroundRatio = 2.5
            self.height = 346
            self.width = 260
            self.num_time_bins = num_steps
            self.length = len(self.oms_frames)-1  # Length of the dataset

                
        def __len__(self):
            return self.length
        
        
        def get_event_idxs(self, idx):
             return self.data['images_idx'][idx], self.data['images_idx'][idx+1] - 1


        def get_start_stop(self, idx, k=1):
            return self.increment*(idx+1), self.increment*(idx + 1 + k)
        
        def get_masked_event_idxs(self, idx, k = 1):
            return self.data['masked_event_idx'][idx], self.data['masked_event_idx'][idx+1]
        
        

        def __getitem__(self, idx):
            start_time, stop_time = self.get_start_stop(idx, self.k)
            index0, index1 = self.get_event_idxs(idx)
            events = self.data['events'][index0:index1]

            #image_idx = self.images_idx[idx]

            masked_index0, masked_index1 = self.get_masked_event_idxs(idx, self.k)
            masked_events = self.data['masked_events'][masked_index0:masked_index1]


            ts = events[:,0]
            xs = events[:,1] 
            ys = events[:,2]
            ps = events[:,3]

            m_ts = masked_events[:,0]
            m_xs = masked_events[:,1] 
            m_ys = masked_events[:,2]
            m_ps = masked_events[:,3]

            ts = (self.num_time_bins-1) * (ts - start_time) /(stop_time - start_time)
            m_ts = (self.num_time_bins-1) * (m_ts - start_time) /(stop_time - start_time)
            
            spike_tensor = torch.zeros((2, self.width, self.height, self.num_time_bins))
            spike_tensor[ps, ys-1, xs-1, ts] = 1
            
            masked_spike_tensor = torch.zeros((2, self.width, self.height, self.num_time_bins))   
            masked_spike_tensor[m_ps, m_ys-1, m_xs-1, m_ts] = 1
           
            full_mask_tensor = torch.zeros((2, self.width, self.height, self.num_time_bins))   
            for i in range(0, self.k):
                curr_start = int((self.num_time_bins) * (i)/self.k)
                curr_end = int((self.num_time_bins) * (i+1)/self.k)

                curr_mask= os.path.join(self.maskDir, "mask_{:08d}.png".format(idx+i+1))
                fullmask = np.asarray(Image.open(curr_mask))
                fullmask = np.expand_dims(fullmask, axis=(0,3))
                tiled = torch.from_numpy(np.tile(fullmask, (2,1,1, curr_end-curr_start)))        

                full_mask_tensor[:,:,:,curr_start:curr_end] = tiled  

            assert not torch.isnan(spike_tensor).any()
            assert not torch.isnan(masked_spike_tensor).any()
            if (len(m_ps) == 0 or len(ps)/len(m_ps) > self.maxBackgroundRatio):
                return None
            oms_masked_frame = torch.empty_like(masked_spike_tensor)
            oms_spike_mask = torch.empty_like(masked_spike_tensor)
            for k in range(self.num_time_bins):
                oms_masked_frame[0,:,:,k]= torch.logical_and(spike_tensor[0, :,:,k], torch.Tensor(self.oms_frames[idx]))
                oms_masked_frame[1, :, :, k]= torch.logical_and(spike_tensor[1,:,:,k], torch.Tensor(self.oms_frames[idx]))
                oms_spike_mask[1,:,:,k] = torch.logical_and(oms_masked_frame[1,:,:,k], masked_spike_tensor[1,:,:,k])
                oms_spike_mask[0,:,:,k] = torch.logical_and(oms_masked_frame[0,:,:,k], masked_spike_tensor[0,:,:,k])
            return  {"oms_spike_tensor": oms_masked_frame,
                 "oms_mask":oms_spike_mask,
                 "dvs_spike_tensor": spike_tensor,
                 "dvs_mask" : masked_spike_tensor,
                 }

            


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



