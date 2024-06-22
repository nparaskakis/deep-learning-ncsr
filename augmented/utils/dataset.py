import os
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data import DataLoader
import pandas as pd


try:
    
    if __name__ == "main":
        sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

    from train import *
    from test import *
    
except ImportError:
    
    from .train import *
    from .test import *





class FSC22Dataset(Dataset):

    def __init__(self, annotations_file: str, data_dir: str, device):
        self.annotations = pd.read_csv(annotations_file)
        self.data_dir = data_dir
        self.class_mapping = self.annotations[['Class ID', 'Class Name']].drop_duplicates().set_index('Class ID')['Class Name'].to_dict()
        self.class_mapping = {k-1: self.class_mapping[k] for k in sorted(self.class_mapping)}
        self.device = device
        
    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item_name = self.annotations.iloc[index, 1].strip(".wav") + ".pt"
        item_path = os.path.join(self.data_dir, item_name)
        item = torch.load(item_path, map_location=self.device)
        label = self._get_audio_sample_label(index)
        return item, label

    def _get_audio_sample_label(self, index: int) -> int:
        return self.annotations.iloc[index, 2] - 1







def get_data_loaders(dataset: Dataset, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:

    train_indices = dataset.annotations[dataset.annotations['split'] == 'train'].index.tolist()
    val_indices = dataset.annotations[dataset.annotations['split'] == 'val'].index.tolist()
    test_indices = dataset.annotations[dataset.annotations['split'] == 'test'].index.tolist()

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, validation_loader, test_loader