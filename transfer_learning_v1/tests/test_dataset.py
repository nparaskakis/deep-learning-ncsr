import torch
import sys
import os

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_directory)

from utils.train import *
from utils.dataset import FSD50KDataset, get_data_loaders


def check_dataset_get_item(dataset, index):
    
    print(f"\nGet the item with index {index} from the dataset")
    print(f"Item's shape: {dataset[index][0].shape}")
    print(f"Item's label: {dataset[index][1]}")    



def check_dataset_annotations(dataset):
    
    print("\nGet the dataframe of the annotations:")
    print(dataset.annotations.head())
    
    


def check_class_mapping(dataset):
    
    print("\nGet the dictionary of the labels' mapping to the classes:")
    print(dataset.class_mapping)






if __name__ == "__main__":
    
    
    ANNOTATIONS_FILE = "../../fsd50k_data/raw/metadata/metadata.csv"
    VOCABULARY_FILE = "../../fsd50k_data/raw/metadata/vocabulary.csv"
    AUDIO_DIR = "../../fsd50k_data/preprocessed/melspectrograms"

    BATCH_SIZE = 64

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    print(f"\nUsing device {device}")

    fsd50k = FSD50KDataset(annotations_file=ANNOTATIONS_FILE, vocabulary_file=VOCABULARY_FILE, data_dir=AUDIO_DIR, device=device)
    
    check_dataset_get_item(fsd50k, 0)
    
    check_dataset_get_item(fsd50k, 299)
    
    print("\n")
    check_dataset_annotations(fsd50k)
    
    print("\n")
    check_class_mapping(fsd50k)