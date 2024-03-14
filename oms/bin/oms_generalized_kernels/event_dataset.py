import os

import tonic
import h5py
import numpy as np

from version_2.bin.dvs_methods.dvs_v2e import dvs_v2e


class IrisDVSDataset(tonic.Dataset):
    sensor_size: tuple = None  # size of event camera sensor (height, width, channels)

    ordering = (
        "txyp"  # the order in which your event channels are provided in your recordings
    )

    def __init__(
            self,
            sensor_size: tuple,
            train: bool = True,
            transform: bool = None,
            target_transform=None,
            data_dir: str = None,
    ):
        """
        Tonic Dataset for Iris DVS VSim_dataset. Lazy-loads the VSim_dataset when needed due to size constraints).

        :param sensor_size: size of event camera sensor (height, width, channels)
        :param train: if True, will load training VSim_dataset. If False, will load testing VSim_dataset
        :param transform: transform to be applied to the events
        :param target_transform: transform to be applied to the target
        :param data_dir: directory of the VSim_dataset
        """

        super(IrisDVSDataset, self).__init__(
            save_to='./', transform=transform, target_transform=target_transform
        )

        self.train: bool = train
        self.sensor_size: tuple = sensor_size
        self.ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        # Check if data_dir is absolute or relative and make it relative
        if os.path.isabs(data_dir):
            self.data_dir: str = os.path.relpath(data_dir, self.ROOT_DIR)
            print(f"converted `data_dir` to relative path: {self.data_dir}")
        else:
            self.data_dir = data_dir

        # replace the strings with your training/testing file locations or pass as an argument
        if self.train and (self.data_dir is not None):
            print("Loading training VSim_dataset from events.h5 and ref-events.h5 files, "
                  "or from .npy files if already converted")
            # Once for the input events, and once for the target events
            IrisDVSDataset.convert_hdf5_to_event_frames(
                self.data_dir, event_frame_name='dvs-events.npy'
            )
            print("All input-events are converted to event-frames")

            IrisDVSDataset.convert_hdf5_to_event_frames(
                os.path.join(self.data_dir), dvs_events_fname='ref-events.h5', event_frame_name='ref-events.npy',
                mode='ref'
            )
            print("All reference-events are converted to event-frames")

            # remove all files that are not data_1, data_2, etc. from the directory list
            directory = os.listdir(os.path.join(self.ROOT_DIR, self.data_dir))
            for i in directory:
                if not i.startswith('data_'):
                    directory.remove(i)

            # list comprehension to save paths to the training_events in a list
            self.train_events: list = [
                os.path.join(self.ROOT_DIR, self.data_dir, i, 'dvs-events.npy')
                for i in directory
                           ]

            # list comprehension to save paths to the target_events in a list
            self.target_events: list = [
                os.path.join(self.ROOT_DIR, self.data_dir, i, 'ref/', 'ref-events.npy')
                for i in directory
            ]

            print("All events-frame files are ready")
        else:
            raise NotImplementedError

        print("Dataset is ready to be used", "\n")

    def __getitem__(self, index):
        events = np.load(self.train_events[index], allow_pickle=True)
        target = np.load(self.target_events[index], allow_pickle=True)

        print(events.shape, target.shape)

        # Shape here is (num_time_steps, polarity=2, height, width)

        # We don't need polarity for this experiment
        events = events[:, 0] + events[:, 1]
        target = target[:, 0] + target[:, 1]

        # Shape here is (num_time_steps, height, width)

        if self.transform is not None:
            events = self.transform(events)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return events, target

    def __len__(self):
        assert len(self.train_events) == len(self.target_events), \
            "Number of training events and target events must be equal"
        return len(self.train_events)

    @staticmethod
    def convert_hdf5_to_event_frames(
            results_dir: str,
            dvs_events_fname: str = 'events.h5',
            event_frame_name: str = 'dvs-events.npy',
            mode: str = None,
    ):
        """
        Converts events.h5 file (from v2e output) to event frames saved as dvs-events.npy files
        :param results_dir: directory of the results folder in which each set of VSim_dataset is saved.
        :param dvs_events_fname: file name of the h5py file (ends in .h5); e.g. 'events.h5'
        :param event_frame_name: name of the event frame file (ends in .npy); e.g. 'dvs-events.npy'
        :param mode: mode to run v2e in. Options are 'ref' or 'main'
        :return: None
        """

        # PARAMETERS
        # print(results_dir)
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        directory = os.listdir(os.path.normpath(os.path.join(ROOT_DIR, results_dir)))
        for i in directory:
            if not i.startswith('data_'):
                directory.remove(i)

        match mode:
            case 'ref':
                dvs_events_fname = 'ref/' + dvs_events_fname
                event_frame_name = 'ref/' + event_frame_name
            case _:
                pass

        # Check if results_dir is absolute or relative and make it relative
        if os.path.isabs(results_dir):
            results_dir = os.path.relpath(results_dir, ROOT_DIR)

        # ASSERTIONS
        # make sure results directory exists, and it is ready for processing
        assert os.path.exists(os.path.join(ROOT_DIR, results_dir)), f"Results directory {results_dir} does not exist."
        for i in directory:
            assert os.path.exists(os.path.join(ROOT_DIR, results_dir, i, dvs_events_fname)), \
                f"events.h5 file does not exist in {results_dir}/{i}. Most likely you have not run v2e yet. " \
                f"{os.path.join(ROOT_DIR, results_dir, i, dvs_events_fname)}"

        # Construct tonic transform, ToFrame for converting events into frames
        to_frame = tonic.transforms.ToFrame(
            sensor_size=(240, 180, 2),
            time_window=(1 / 120) * 1000000,  # distributes events into time bins of 1000 microseconds each
            include_incomplete=False,
        )

        # CODE LOOP
        # Loop through each directory in results_dir
        for i in directory:

            # Check that for the given directory, the dvs-events.npy file does not already exist
            if not os.path.exists(os.path.join(ROOT_DIR, results_dir, i, event_frame_name)):
                # Load Parameters and Results Directories
                results_dir = os.path.join(ROOT_DIR, 'VSim_dataset', f'results/{i}/')

                # print(os.path.join(results_dir, 'events.h5'))
                # Check that the events.h5 file exists
                assert os.path.exists(os.path.join(results_dir, 'events.h5')), \
                    f"v2e has not been run in {results_dir} yet. Please run it before proceeding."

                # Load h5py Dataset
                dvs_v2e_file = h5py.File(os.path.join(results_dir, dvs_events_fname), "r")
                dataset = dvs_v2e_file["events"]

                events = IrisDVSDataset.create_input(
                    dataset,
                    n_events=dataset.shape[0]
                )

                # Convert events to dvs event-frames
                convert_to_frame = to_frame(events)

                np.save(os.path.join(ROOT_DIR, results_dir, event_frame_name), convert_to_frame, allow_pickle=True)

    @staticmethod
    def create_input(
            dataset,
            n_events,
            dtype=np.dtype([("x", int), ("y", int), ("t", int), ("p", int)]),
    ):
        """
        Reformat h5py dataset into a compatible format for the tonic dataset.
        Format using DVS standards [x, y, time stamp, polarity]

        :param dataset: h5py dataset
        :param n_events: number of events to be used
        :param dtype: VSim_dataset type of the output array

        :return: numpy array of events
        """

        events = np.zeros(n_events, dtype=dtype)
        events["x"] = dataset[:, 1]
        events["y"] = dataset[:, 2]
        events["p"] = dataset[:, 3]
        events["t"] = dataset[:, 0]
        return events

    @staticmethod
    def apply_v2e(
            results_dir: str,
            v2e_dir: str,
            mode: str = None,
    ):
        """
        Applies v2e to the VSim_dataset in the results directory. Requires that you have created params.yaml files for each
        folder in results/.

        :param results_dir: directory of the results folder in which each set of VSim_dataset is saved.
        :param v2e_dir: absolute path to the v2e folder
        :param mode: mode to run v2e in. Options are 'ref' or 'main'

        :return: None
        """

        # TODO: NEEDS TESTING

        ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

        directory = os.listdir(os.path.join(ROOT_DIR, 'results/'))
        for i in directory:
            if not i.startswith('data_'):
                directory.remove(i)

        for i in range(len(directory)):
            params_dir = rf"params/data_{i + 1}.yaml"
            results_dir = os.path.join(ROOT_DIR, f'results/data_{i + 1}/')

            dvs_v2e(
                params_dir=params_dir,
                path_to_v2e=v2e_dir,
                output_dir=results_dir,
                mode=mode
            )
