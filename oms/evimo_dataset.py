import os
import h5py
from PIL import Image
from pathlib import Path

import numpy as np
import tonic
import torch


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

    def __init__(self, data_dir):
        super().__init__("./")

        self.data_dir = data_dir  # should be an absolute path

        # contains a list of the paths to each .npy file for every video in the database
        self.dvs_events: list = []
        self.masks: list = []

        # Iterate through each class of data
        for class_path in os.listdir(self.data_dir):

            class_dir = os.path.join(self.data_dir, class_path, "txt")
            save_dir = os.path.join(self.data_dir, class_path, "npy")

            # replace the strings with your training/testing file locations or pass as an argument
            if self.data_dir is not None:
                print("Loading EVIMO dataset from .npz files, or from .npy files if already converted")

                # convert the .npz files to .npy files if they haven't been converted already
                self.process_data(save_dir=save_dir, class_dir=class_dir)

            # store the new data and target paths as lists
            directory = os.listdir(save_dir)
            
            # list of all the dvs_event paths (.npy files)
            dvs_events: list = sorted([
                os.path.join(save_dir, file)
                for file in directory if file.endswith("_frames.npy")
            ])
            #print(dvs_events)

            # list of all the mask paths (.npy files)
            masks: list = sorted([
                os.path.join(save_dir, file)
                for file in directory if file.endswith("_masks.npy")
            ])

            self.dvs_events.extend(dvs_events)
            self.masks.extend(masks)

        """# create a new directory for the parsed files in the same level as the npz directory called npy
        self.save_dir = os.path.join(os.path.dirname(self.data_dir), "npy")

        # replace the strings with your training/testing file locations or pass as an argument
        if self.data_dir is not None:
            print("Loading EVIMO dataset from .npz files, or from .npy files if already converted")

            # convert the .npz files to .npy files if they haven't been converted already
            self.process_data(save_dir=self.save_dir)

        # store the new data and target paths as lists
        directory = os.listdir(self.save_dir)

        # list of all the dvs_event paths (.npy files)
        self.dvs_events: list = sorted([
            os.path.join(self.save_dir, file)
            for file in directory if file.endswith("_frames.npy")
        ])

        # list of all the mask paths (.npy files)
        self.masks: list = sorted([
            os.path.join(self.save_dir, file)
            for file in directory if file.endswith("_masks.npy")
        ])"""

    def __len__(self):
        assert len(self.dvs_events) == len(self.masks)
        return len(self.dvs_events)

    def __getitem__(self, idx):

        # get the file name
        f_name = Path(self.dvs_events[idx]).stem

        # get the class name
        c_name = Path(self.dvs_events[idx]).parent.parent.stem

        # load the data and target from the paths
        frames = np.load(self.dvs_events[idx])
        #print("FRAMES", frames.shape)

        masks = np.load(self.masks[idx])
        masks = torch.tensor(masks).float()


        return frames, masks, f_name[:-7], c_name

    def process_data(self, save_dir: str, class_dir: str):
        """
        Takes each .npz file and parses it. From the file we take out the frames and masks and save those to .npy files
        """

        save_dir = os.path.join(save_dir)

        # if the path already exists
        if os.path.exists(save_dir):
            # check if the number of files in the save_dir is equal to 2 * the number of .npz files in the data_dir
            # because there is one file for frames and one for masks for each .npz file
            converted_files = [file for file in os.listdir(save_dir) if
                    file.endswith(".npy")]
            if len(converted_files) == 2 * len(os.listdir(class_dir)):
                print("Already converted all files to .npy")
                return  # exit the function
        else:
            # if the path does not exist, create it
            os.mkdir(save_dir)

        # Get all the files in the directory
        files = os.listdir(class_dir)

        # Iterate through each file
        for file in files:
            #file += '/img/'
            # if the file is already converted, skip it
            #if Path(file).stem + "_frames.npy" in os.listdir(save_dir):
               # continue

            # Load the data from the .npz file
            frames, masks = EVIMODataset.load_data(os.path.join(class_dir,
                file), file)

            # convert to pathlib.Path object so that you can use the .stem method to get the name
            file = Path(file)

            # Save the frames and masks to .npy files
            np.save(os.path.join(save_dir, f"{file.stem}_frames.npy"), frames)
            np.save(os.path.join(save_dir, f"{file.stem}_masks.npy"), masks)

            del frames, masks


    @staticmethod
    def get_event_idxs(data, index, k=1):
        return data['events_idx'][index], data['events_idx'][index+k] - 1


    @staticmethod
    def generate_dvs_frames(data, index, k = 1):
        timeframes = data['timeframes']
        idx0, idx1 = EVIMODataset.get_event_idxs(data,index, 1)
        frame_events = data['events'][idx0:idx1]
        #print(frame_events)
        t_window =  frame_events[-1,0] - frame_events[0,0]
        dtype = np.dtype([("x", int), ("y", int), ("t", float), ("p", int)])
        events_np = np.zeros(frame_events.shape[0], dtype=dtype)
        events_np["x"] = frame_events[:, 1]
        events_np["y"] = frame_events[:, 2]
        events_np["p"] = frame_events[:, 3]
        events_np["t"] = frame_events[:, 0]
        #start_t = data['timeframes'][index][0]
        transform = tonic.transforms.ToFrame(
        sensor_size=(346, 260, 2),
        time_window=t_window)
        frame = torch.tensor(transform(events_np), dtype=torch.float32)
        return frame.squeeze(), f"depth_mask_{int(timeframes[index,2])}.png"

    @staticmethod
    def get_mask(file: str):
        mask_img = Image.open(file)
        return torch.tensor(np.array(mask_img)[:,:,0])
    
    @staticmethod
    def load_data(path: str, seq: str):
        """
        Loads the data from the .npz file and returns the frames and masks as numpy arrays.

        :param path: path to the .npz file
        :return: frames and masks as numpy arrays
        """
        assert os.path.isfile(path +'/'+seq + '.hdf5')
        data = h5py.File(path+'/'+ seq+ '.hdf5', 'r')
        frames = torch.zeros(data['events_idx'].shape[0]- 1, 2, 260, 346)
        masks = torch.zeros(data['events_idx'].shape[0]- 1, 260, 346)
        for i in range(data['events_idx'].shape[0] - 1):
            frame, mask_file = EVIMODataset.generate_dvs_frames(data, i)
            mask = EVIMODataset.get_mask(path +'/img/'+mask_file)
            frames[i:i+1:, :,:,:] = frame
            masks[i:i+1, :, :] = mask

        #print(f"Frames Shape: {frames.shape}")
        #print(f"Masks Shape: {mask.shape}")

        frames = torch.sum(frames, axis=1)  # combine all spike activity into single axis
        frames = frames[:, None, :, :]
        frames[frames > 0] = 1


        try:
            assert masks.shape == frames.shape
        except AssertionError:
            #print(f"Frames Shape: {frames.shape}, Masks Shape: {masks.shape}")
            if frames.shape[0] > masks.shape[0]:
                frames = frames[:masks.shape[0]]
            else:
                masks = masks[:frames.shape[0]]

        #print(f"Masks Shape: {masks.shape}")
        #print(f"Masks Unique: {torch.unique(masks)}")

        return frames, masks
