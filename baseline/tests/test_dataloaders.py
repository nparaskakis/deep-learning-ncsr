import torch
import sys
import os

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_directory)

from utils.train import *
from utils.dataset import FSC22Dataset, get_data_loaders




def check_dataloader(data_loader):
    
    samples = 0
    all_labels = torch.tensor([])
    for _, labels in data_loader:
        samples += len(labels)
        all_labels = torch.cat((all_labels, labels), axis=0)
        pass
    
    all_labels = all_labels.to(torch.int64)
    labels_counts = torch.bincount(all_labels)
    
    return samples, labels_counts







if __name__ == "__main__":
    
    
    ANNOTATIONS_FILE = "../data/raw/metadata/metadata_FSC22.csv"
    AUDIO_DIR = "../data/preprocessed/audio"

    BATCH_SIZE = 64

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    print(f"\nUsing device {device}")

    fsc22 = FSC22Dataset(annotations_file=ANNOTATIONS_FILE, data_dir=AUDIO_DIR)
    
    train_loader, val_loader, test_loader = get_data_loaders(dataset=fsc22, train_size=0.7, val_size=0.15, test_size=0.15, batch_size=BATCH_SIZE)
    
    train_samples, train_labels_counts = check_dataloader(train_loader)
    val_samples, val_labels_counts = check_dataloader(val_loader)
    test_samples, test_labels_counts = check_dataloader(test_loader)
    
    print("\nTraining Set")
    for number, count in enumerate(train_labels_counts):
        print(f"Label {number}: {count} samples")
    
    print("\nValidation Set")
    for number, count in enumerate(val_labels_counts):
        print(f"Label {number}: {count} samples")
    
    print("\nTest Set")
    for number, count in enumerate(test_labels_counts):
        print(f"Label {number}: {count} samples")
    
    print(f"\nTraining Set: {train_samples} samples")
    print(f"Validation Set: {val_samples} samples")
    print(f"Test Set: {test_samples} samples")