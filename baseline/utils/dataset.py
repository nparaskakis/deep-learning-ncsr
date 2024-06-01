import os
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data import DataLoader
import pandas as pd

from sklearn.model_selection import train_test_split

try:
    
    if __name__ == "main":
        sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

    from train import *
    from test import *
    
except ImportError:
    
    from .train import *
    from .test import *





class FSC22Dataset(Dataset):

    def __init__(self, annotations_file: str, data_dir: str):
        self.annotations = pd.read_csv(annotations_file)
        self.data_dir = data_dir
        self.class_mapping = self.annotations[['Class ID', 'Class Name']].drop_duplicates().set_index('Class ID')['Class Name'].to_dict()
        self.class_mapping = {k-1: self.class_mapping[k] for k in sorted(self.class_mapping)}

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item_name = self.annotations.iloc[index, 1].strip(".wav") + ".pt"
        item_path = os.path.join(self.data_dir, item_name)
        item = torch.load(item_path)
        label = self._get_audio_sample_label(index)
        return item, label

    def _get_audio_sample_label(self, index: int) -> int:
        return self.annotations.iloc[index, 2] - 1







def get_data_loaders(dataset: Dataset, train_size: float, val_size: float, test_size: float, batch_size: int, random_seed: int = 42) -> tuple[DataLoader, DataLoader, DataLoader]:

    assert train_size + val_size + test_size == 1, "The sum of sizes must be 1."

    train_val, test_data = train_test_split(dataset.annotations, test_size=test_size, stratify=dataset.annotations['Class ID'], random_state=random_seed)
    train_data, val_data = train_test_split(train_val, test_size=val_size/(train_size + val_size), stratify=train_val['Class ID'], random_state=random_seed)

    train_indices = train_data.index.tolist()
    val_indices = val_data.index.tolist()
    test_indices = test_data.index.tolist()

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, validation_loader, test_loader