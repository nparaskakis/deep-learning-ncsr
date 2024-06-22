import os
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data import DataLoader
import pandas as pd

from torchvision import transforms

from sklearn.model_selection import train_test_split

try:
    
    if __name__ == "main":
        sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

    from train import *
    from test import *
    
except ImportError:
    
    from .train import *
    from .test import *





class FSD50KDataset(Dataset):
    def __init__(self, annotations_file: str, vocabulary_file: str, data_dir: str, device, model_str):
        
        self.annotations = pd.read_csv(annotations_file)
        self.vocabulary = pd.read_csv(vocabulary_file, header=None, names=['Class Number', 'Class Name', 'Mid'])
        self.data_dir = data_dir
        self.device = device
        self.num_classes = 200 # len(self.vocabulary)
        # Create a mapping from Class Number to Class Name
        self.class_mapping = self.vocabulary.set_index('Class Number')['Class Name'].to_dict()

        self.model_str = model_str
        
    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item_name = f"{self.annotations.iloc[index, 0]}.pt"
        item_path = os.path.join(self.data_dir, item_name)
        item = torch.load(item_path, map_location=self.device)
        
        if self.model_str == "mobilenet":
            # tr = transforms.Resize((224, 224)).to("cpu")
            # item = tr(item.to("cpu")).to(self.device)
            item = item.repeat(3, 1, 1)
            
        classes = self.annotations.iloc[index, 2].split(",")
        classes_int = list(map(int, classes))
        
        # Convert to one-hot encoding
        labels = self.to_one_hot(classes_int).to(self.device)
        return item, labels
    
    def to_one_hot(self, class_numbers: list[int]) -> torch.Tensor:
        """
        Transforms a list of class numbers to one-hot encoded tensor.
        """
        labels_one_hot = torch.zeros(self.num_classes, device=self.device)
        labels_one_hot[class_numbers] = 1
        return labels_one_hot

    def from_one_hot(self, one_hot_tensor: torch.Tensor) -> list[int]:
        """
        Transforms a one-hot encoded tensor back to a list of class numbers.
        """
        class_numbers = torch.where(one_hot_tensor == 1)[0].tolist()
        return class_numbers
    
    def _get_audio_sample_labels(self, index: int) -> torch.Tensor:
        classes = self.annotations.iloc[index, 2].split(",")
        classes_int = list(map(int, classes))
        
        # Convert to one-hot encoding
        labels = self.to_one_hot(classes_int).to(self.device)
        
        return labels






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