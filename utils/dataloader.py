from typing import List, Tuple
import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np


class TrainingDataset(Dataset):
    """Customized dataset.

    Args:
        root (str): Path to the dataset.
        input_steps (int): Number of input steps.
        forecast_steps: (int): Number of forecast steps.
        x_range (List[int]): Longitude range for images.
        y_range (List[int]): Latitude range for images.
    """

    def __init__(self, root: str, input_steps: int, forecast_steps: int, x_range: List[int], y_range: List[int]):
        super().__init__()
        self.root = root
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.x_range = x_range
        self.y_range = y_range

        self.sample_num = []
        self.files = []
        self.total_steps = self.input_steps + self.forecast_steps
        self.date_list = sorted(os.listdir(self.root))
        for date in self.date_list:
            file_list = sorted(os.listdir(os.path.join(self.root, date)))
            self.sample_num.append(len(file_list) - self.total_steps + 1)
            for file_ in file_list:
                self.files.append(os.path.join(self.root, date, file_))
        
        self.sample_num = np.array(self.sample_num)
        self.sample_cumsum = np.cumsum(self.sample_num)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        files = self.locate_files(index)
        tensor, timestamp = self.load_pt(files)
        return tensor, timestamp
    
    def __len__(self) -> int:
        return sum(self.sample_num)
        
    def locate_files(self, index: int) -> list:
        date_order = np.where(index - self.sample_cumsum < 0)[0][0]
        if date_order == 0:
            file_anchor = index
        else:
            file_anchor = index + date_order * (self.total_steps - 1)
        files = self.files[file_anchor: file_anchor + self.total_steps]
        return files
    
    def load_pt(self, files: list) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor = []
        timestamp = []
        for file_ in files:
            tensor_single, timestamp_single = torch.load(file_)
            tensor_single = tensor_single[:, self.y_range[0]: self.y_range[1], self.x_range[0]: self.x_range[1]]
            tensor_single = tensor_single.transpose(2, 1)
            tensor.append(tensor_single)
            timestamp.append(timestamp_single)
        tensor = torch.stack(tensor)
        timestamp = torch.LongTensor(timestamp)
        return tensor, timestamp


class CaseDataset(TrainingDataset):
    """Customized dataset.

    Args:
        root (str): Path to the dataset.
        case_indices (List[int]): Indices of cases.
        input_steps (int): Number of input steps.
        forecast_steps (int): Number of input steps. 
        x_range (List[int]): X range.
        y_range (List[int]): Y range.
    """

    def __init__(self, root: str, case_indices: int, input_steps: int, forecast_steps: int, 
                 x_range: List[int], y_range: List[int]):
        super().__init__(root, input_steps, forecast_steps, x_range, y_range)
        self.case_indices = case_indices

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        files = self.locate_files(self.case_indices[index])
        tensor, timestamp = self.load_pt(files)
        return tensor, timestamp

    def __len__(self) -> int:
        return len(self.case_indices)


def load_data(root: str, input_steps: int, forecast_steps: int, batch_size: int, num_workers: int, 
              train_ratio: float, valid_ratio: float, x_range: List[int], y_range: List[int]) \
              -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load training and test data.

    Args:
        root (str): Path to the dataset.
        input_steps (int): Number of input steps.
        forecast_steps (int): Number of forecast steps.
        batch_size (int): Batch size.
        num_workers (int): Number of processes.
        train_ratio (float): Training ratio of the whole dataset.
        valid_ratio (float): Validation ratio of the whole dataset.
        x_range (List[int]): X range.
        y_range (List[int]): Y range.

    Returns:
        DataLoader: Dataloader for training and test.
    """

    dataset = TrainingDataset(root, input_steps, forecast_steps, x_range, y_range)
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
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers,
                             shuffle=False, pin_memory=True)

    print('\nDataset Length:', len(dataset))
    print('Train Set Length:', len(train_set))
    print('Val Set Length:', len(val_set))
    print('Test Set Length:', len(test_set))
    print('Train Loader Batch Num:', len(train_loader))
    print('Val Loader Batch Num:', len(val_loader))
    print('Test Loader Batch Num:', len(test_loader))

    return train_loader, val_loader, test_loader


def load_case(root: str, case_indices: List[int], input_steps: int, forecast_steps: int, 
              x_range: List[int], y_range: List[int]) -> DataLoader:
    """Load case data.

    Args:
        root (str): Path to the dataset.
        case_indices (List[int]): Indices of cases.
        input_steps (int): Number of input steps.
        forecast_steps (int): Number of forecast steps.
        x_range (List[int]): X range.
        y_range (List[int]): Y range.

    Returns:
        DataLoader: Dataloader for case.
    """

    case_set = CaseDataset(root, case_indices, input_steps, forecast_steps, x_range, y_range)
    case_loader = DataLoader(case_set, batch_size=1)
    return case_loader
