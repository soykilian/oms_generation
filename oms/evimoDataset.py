class EVIMODatasetBase(Dataset):
    def __init__(self, datafile, genconfigs, maskDir, crop, maxBackgroundRatio, incrementPercent):

        self.height = genconfigs['data']['height']
        self.width = genconfigs['data']['width'] 
        self.crop = crop
        self.maxBackgroundRatio = maxBackgroundRatio
        
        if (self.crop):
            self.height_c = genconfigs['data']['height_c'] 
            self.width_c = genconfigs['data']['width_c'] 

        self.k = genconfigs['data']['k']
        self.min_events = genconfigs['data']['minEvents'] 
        self.start = genconfigs['data']['start']
        
        self.num_time_bins = genconfigs['simulation']['tSample'] 

        self.maskDir = maskDir

        self.data = h5py.File(datafile, 'r')
        self.length =  len(self.data['events_idx']) - self.k - 1 
        print("EVIMO tot length", self.length)

        self.incrementPercent = incrementPercent

    def getHeightAndWidth(self):
        assert self.height
        assert self.width
        return self.height, self.width

    @staticmethod
    def isDataFile(filepath: str):
        suffix = Path(filepath).suffix
        return suffix == '.h5' or suffix == '.npz'

    def __len__(self):
        return self.length

    def _preprocess(self, events, start, stop):
        return self._collate(events, start, stop)

    def get_event_idxs(self, index):
        return self.data['events_idx'][index], self.data['events_idx'][index+self.k] - 1


    def __getitem__(self, index: int):
        idx0, idx1 = self.get_event_idxs(index)
    
        events = self.data['events'][idx0:idx1]

        start_t = self.data['timeframes'][index][0]
        stop_t = self.data['timeframes'][index+self.k-1][1]
        
        if(self.incrementPercent != 1):
            stop_t =  start_t + (stop_t - start_t)* self.incrementPercent
            events = events[events[:,0] < stop_t,:]

        ts = events[:,0]
        xs = events[:,1] 
        ys = events[:,2]
        ps = events[:,3]

        # print("index {} idx0 {} idx1 {} k {} \n time {}, {} start_t {} stop_t {}\n".format(
        #     index, idx0, idx1, self.k, ts[0], ts[-1], start_t, stop_t))
 
        spike_tensor = torch.zeros((2, self.height, self.width, self.num_time_bins))   
        ts = (self.num_time_bins-1) * (ts - start_t) /(stop_t - start_t)
        
        spike_tensor = torch.zeros((2, self.height, self.width, self.num_time_bins))   
        spike_tensor[ps, ys, xs, ts] = 1

        full_mask_tensor = torch.zeros((2, self.height, self.width, self.num_time_bins))   

        for i in range(0, self.k):
            curr_start = int((self.num_time_bins) * (i)/self.k)
            curr_end = int((self.num_time_bins) * (i+1)/self.k)

            currfile_nm = os.path.join(self.maskDir, "depth_mask_{:d}.png".format(
                int(self.data['timeframes'][index][2])))

            if (not os.path.isfile(currfile_nm)):
                print("mask file not found", currfile_nm)
                return None
            fullmask = np.asarray(Image.open(currfile_nm))[:,:,0]
            fullmask = fullmask.astype(bool).astype(float)

            if (np.sum(fullmask) < self.min_events):
                print("not enough events", np.sum(fullmask), " < min events: ", self.min_events)
                return None

            kernel = np.ones((5, 5), 'uint8')
            fullmask = cv2.dilate(fullmask, kernel, iterations=1)

            fullmask = np.expand_dims(fullmask, axis=(0,3))
            tiled = torch.from_numpy(np.tile(fullmask, (2,1,1, curr_end-curr_start)))      

            full_mask_tensor[:,:,:,curr_start:curr_end] = tiled  

        masked_spike_tensor = ((spike_tensor + full_mask_tensor) > 1).float()
        background_spikes = (spike_tensor + torch.logical_not(masked_spike_tensor).float()) > 1


        if (self.crop):
            summed = torch.sum(masked_spike_tensor, axis = (0,3))
            max_v = torch.argmax(summed)
            
            center_x = int(max_v % self.width)
            center_y = int(max_v / self.width)

            crop_x_min  = int(max(center_x - int(self.width_c/2), 0))
            crop_y_min  = int(max(center_y - int(self.height_c/2), 0))

            if ((center_x + int(self.width_c/2)) > self.width - 1):
                crop_x_min = self.width - 1 - self.width_c

            if ((center_y + int(self.height_c/2)) > self.height - 1):
                crop_y_min = self.height - 1 - self.height_c


            spike_tensor = spike_tensor[:,
                        int(crop_y_min):int(crop_y_min+self.height_c), 
                        int(crop_x_min):int(crop_x_min+self.width_c), 
                        :]
            masked_spike_tensor = masked_spike_tensor[:,
                                int(crop_y_min):int(crop_y_min+self.height_c),
                                int(crop_x_min):int(crop_x_min+self.width_c), :100]
            full_mask_tensor = full_mask_tensor[:,
                                int(crop_y_min):int(crop_y_min+self.height_c),
                                int(crop_x_min):int(crop_x_min+self.width_c), :100]

        if (torch.sum(background_spikes)/torch.sum(masked_spike_tensor) > self.maxBackgroundRatio):
            return None

        assert not torch.isnan(spike_tensor).any()
        assert not torch.isnan(masked_spike_tensor).any()
        out = {
            'file_number': index,
            'spike_tensor': spike_tensor,
            'masked_spike_tensor': masked_spike_tensor,
            'full_mask_tensor': full_mask_tensor,
            'time_start': start_t,
            'time_per_index': (stop_t - start_t)/self.num_time_bins,
            'ratio': torch.sum(background_spikes)/torch.sum(masked_spike_tensor)
        }
        return out

