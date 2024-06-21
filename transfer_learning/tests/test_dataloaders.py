import torch
import sys
import os

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_directory)

from utils.train import *
from utils.dataset import FSD50KDataset, get_data_loaders




def check_dataloader(data_loader, device):
    
    samples = 0
    all_labels = torch.tensor([]).to(device)
    for _, labels in data_loader:
        # print("labels shape", labels.shape)
        samples += len(labels)
        all_labels = torch.cat((all_labels, labels), axis=0)
        pass
    
    all_labels = all_labels.to(torch.int64)
    labels_counts = torch.bincount(all_labels)
    
    return samples, labels_counts







if __name__ == "__main__":
    
    
    ANNOTATIONS_FILE = "../../fsd50k_data/raw/metadata/metadata.csv"
    VOCABULARY_FILE = "../../fsd50k_data/raw/metadata/vocabulary.csv"
    AUDIO_DIR = "../../fsd50k_data/preprocessed/melspectrograms"

    BATCH_SIZE = 128

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    print(f"\nUsing device {device}")

    fsd50k = FSD50KDataset(annotations_file=ANNOTATIONS_FILE, vocabulary_file=VOCABULARY_FILE, data_dir=AUDIO_DIR, device=device)
    
    train_loader, val_loader, test_loader = get_data_loaders(dataset=fsd50k, batch_size=BATCH_SIZE)
    
    train_samples, train_labels_counts = check_dataloader(train_loader, device)
    val_samples, val_labels_counts = check_dataloader(val_loader, device)
    test_samples, test_labels_counts = check_dataloader(test_loader, device)
    
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