import torchaudio
import torch
import sys
import os

parent_directory = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_directory)

from utils.train import *
from utils.dataset import FSC22Dataset, get_data_loaders


def test_dataloader():
    ANNOTATIONS_FILE = "../data/raw/metadata/metadata_FSC22.csv"
    AUDIO_DIR = "../data/preprocessed/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050*5

    BATCH_SIZE = 16
    EPOCHS = 2
    LEARNING_RATE = 1e-5

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device {device}")

    # mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=SAMPLE_RATE,
    #     n_fft=1024,
    #     hop_length=512,
    #     n_mels=128
    # )

    fsc22 = FSC22Dataset(
        annotations_file=ANNOTATIONS_FILE,
        audio_dir=AUDIO_DIR,
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

    assert train_sum > 0, "Train dataloader is empty"
    assert val_sum > 0, "Validation dataloader is empty"
    assert test_sum > 0, "Test dataloader is empty"


if __name__ == "__main__":
    test_dataloader()
