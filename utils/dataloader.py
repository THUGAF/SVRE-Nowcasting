import os
from typing import Tuple, List
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset


class RadarDataset(Dataset):
    """Customized dataset.

    Args:
        root (str): Root directory for the dataset.
        start (int): Start point of the dataset.
        end (int): End point of the dataset.
        lon_range (List[int]): Longitude range for images.
        lat_range (List[int]): Latitude range for images.
        seed (int): Random seed for initializing workers.
    """
    
    def __init__(self, root: str, start: int, end: int, lon_range: List[int], lat_range: List[int], seed: int):
        assert end >= start
        self.root = root
        self.start = start
        self.end = end
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.seed = seed

    def __getitem__(self, index: int):
        self.data, self.seconds = self._load_ith_npz(self.start + index)
        self.data = self.data[:, :, self.lat_range[0]: self.lat_range[1], self.lon_range[0]: self.lon_range[1]]
        return self.data, self.seconds

    def __len__(self):
        return len(range(self.start, self.end))

    def _load_ith_npz(self, index: int):
        npz_file = np.load(os.path.join(self.root, str(index) + '.npz'))
        return torch.from_numpy(npz_file['DBZ']).float(), torch.from_numpy(npz_file['UNIX_Time']).int()


def load_data(root: str, start: int, end: int, batch_size: int, num_workers: int, train_ratio: float, valid_ratio: float, 
              lon_range: List[int], lat_range: List[int], seed: int) -> Tuple[DataLoader, DataLoader]:
    r"""Load training data and validation data.

    Args:
        root (str): Path to the dataset.
        start (int): Start point of the dataset.
        end (int): End point of the dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of processes.
        train_ratio (float): Training ratio of the whole dataset.
        valid_ratio (float): Validation ratio of the whole dataset.
        lon_range (List[int]): Longitude range for images.
        lat_range (List[int]): Latitude range for images.
        seed (int): Random seed for initializing workers.

    Returns:
        Dataloader: Dataloader for train-set.
        Dataloader: Dataloader for validation-set.
    """

    print('Loading data ...')
    dataset = RadarDataset(root, start, end, lon_range, lat_range, seed)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    train_node = round(train_ratio * dataset_size)
    val_node = round(valid_ratio * dataset_size)
    train_indices = indices[:train_node]
    val_indices = indices[train_node: train_node + val_node]
    test_indices = indices[train_node + val_node:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, 
                              shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, 
                            shuffle=False, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, 
                             shuffle=False, drop_last=True, pin_memory=True)
    
    print('\nTrain Loader')
    print('----Batch Num:', len(train_loader))

    print('\nVal Loader')
    print('----Batch Num:', len(val_loader))

    print('\nTest Loader')
    print('----Batch Num:', len(test_loader))
    
    return train_loader, val_loader, test_loader


def load_sample(root: str, num: int, lon_range: List[int], lat_range: List[int], seed: int):
    r"""Load sample data.

    Args:
        root (str): Path to the dataset.
        num (int): Position of sample data in the dataset.
        lon_range (List[int]): Longitude range for images.
        lat_range (List[int]): Latitude range for images.
        seed (int): Random seed for initializing workers.

    Returns:
        DataLoader: Dataloader for sample.
    """

    sample_loader = DataLoader(dataset=RadarDataset(root, num, num + 1, lon_range, lat_range, seed),
                               batch_size=1, drop_last=True)
    
    print('\nSample Loader')
    print('----Batch Num:', len(sample_loader))

    return sample_loader


# if __name__ == '__main__':
#     train, val, test = load_data('/data/gaf/SBandCRNpz', 0, 18016, 16, 1, 0.7, 0.1, [273, 529], [270, 526], 2021)
