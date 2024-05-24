import torch
import sys

from torchsummary import summary
from torch import nn
from models import *


if __name__ == "__main__":

    if len(sys.argv) < 2:
        # Default model choice when no command-line argument is provided
        model_choice = "CNN1"
    else:
        model_choice = sys.argv[1]

    model_choice = sys.argv[1]

    if model_choice not in ["CNN1", "CNN2", "VGG16", "VGG19"]:
        raise ValueError(
            "Invalid model choice. Please choose 'CNN', 'CNN2', 'VGG16', or 'VGG19'.")

    if model_choice == "CNN1":
        model = CNNNetwork1()
    elif model_choice == "CNN2":
        model = CNNNetwork2()
    elif model_choice == "VGG16":
        model = VGG16Network()
    elif model_choice == "VGG19":
        model = VGG19Network()

    # Print a summary of the network architecture, specifying input size (channels, height, width)
    # Example input size (1 channel, 128x216 image/spectrogram)
    summary(model.cpu(), (1, 128, 216))
    # summary(model, (3, 224, 224))
