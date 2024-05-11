# Import necessary libraries

import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data import DataLoader

import torchaudio

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from model import *
from training import *
from testing import *



# Class for the ESC50 Dataset

class FSC22Dataset(Dataset):
    
    def __init__(self, annotations_file: str, audio_dir: str, transformation: callable, target_sample_rate: int, num_samples: int, device: str | torch.device):
        
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        if self.device == "mps":
            self.transformation = transformation.to("cpu")
        else:
            self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        # Mapping from numeric labels to categorical labels
        self.class_mapping = self.annotations[['Class ID', 'Class Name']].drop_duplicates().set_index('Class ID')['Class Name'].to_dict()
        self.class_mapping = {k-1: self.class_mapping[k] for k in sorted(self.class_mapping)}

    def __len__(self) -> int:
        
        return len(self.annotations)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._preprocess_signal(signal, sr)
        if self.device == "mps":
            signal = self.transformation(signal.to("cpu")).to(self.device)
        else:
            signal = self.transformation(signal)
        return signal, label

    def _preprocess_signal(self, signal: torch.Tensor, sr: int) -> torch.Tensor:
        
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        return signal

    def _resample_if_necessary(self, signal: torch.Tensor, sr: int) -> torch.Tensor:
        
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        
        if (signal.shape[0] > 1):
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(self, signal: torch.Tensor):
        
        if (signal.shape[1] > self.num_samples):
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        
        length_signal = signal.shape[1]
        if (length_signal < self.num_samples):
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _get_audio_sample_path(self, index: int) -> str:
        
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 1])
        return path

    def _get_audio_sample_label(self, index: int) -> int:
        
        return self.annotations.iloc[index, 2]-1





# Functions to create DataLoader instances for training, validation, and testing.

def get_data_loaders(dataset: FSC22Dataset, train_size: float, val_size: float, test_size: float, batch_size: int, random_seed: int = 42) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    assert train_size + val_size + test_size == 1, "The sum of sizes must be 1."

    # Split data into train, validation, and test sets
    train_val, test_data = train_test_split(dataset.annotations, test_size=test_size, stratify=dataset.annotations['Class ID'], random_state=random_seed)
    train_data, val_data = train_test_split(train_val, test_size=val_size/(train_size + val_size), stratify=train_val['Class ID'], random_state=random_seed)

    # Creating data indices for training and validation splits
    train_indices = train_data.index.tolist()
    val_indices = val_data.index.tolist()
    test_indices = test_data.index.tolist()

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, validation_loader, test_loader




# Test the functions for data loading

if __name__ == "__main__":
    
    ANNOTATIONS_FILE = "data/metadata/metadata_FSC22.csv"
    AUDIO_DIR = "data/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050*5
    
    BATCH_SIZE = 16
    EPOCHS = 2
    LEARNING_RATE = 1e-5

    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )

    fsc22 = FSC22Dataset(
        annotations_file=ANNOTATIONS_FILE,
        audio_dir=AUDIO_DIR,
        transformation=mel_spectrogram,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        device=device
    )
    
    print(fsc22[0][0].shape)
    
    
    
    # Test function get_data_loaders_by_percentage
    
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset=fsc22,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        batch_size=BATCH_SIZE
    )
    
    train_sum = 0
    
    for inputs, labels in train_loader:
        train_sum += len(labels)
        
        pass
    
    
    
    val_sum = 0
    labelss = torch.tensor([])
    for inputs, labels in val_loader:
        val_sum += len(labels)
        labelss = torch.cat((labelss, labels), axis=0)
        pass
    # print(labelss)
    
    # Since `torch.bincount` works with integers, ensure your tensor is of integer type
    labelss = labelss.to(torch.int64)

    # Count occurrences
    counts = torch.bincount(labelss)

    # To display the counts along with their corresponding numbers
    for number, count in enumerate(counts):
        if count > 0:
            print(f"Number {number}: {count} times")
            
    test_sum = 0
    for inputs, labels in test_loader:
        test_sum += len(labels)
        pass
        
    print(f"Train: {train_sum}\t Validation: {val_sum}\t Test: {test_sum}")