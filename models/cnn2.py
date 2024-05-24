import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F
from torch.nn import Module


class CNNNetwork2(Module):

    def __init__(self):

        super().__init__()

        self.layers12 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layers34 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layers567 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layers8910 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Flatten layer to convert matrix into a vector for the fully connected layer
        self.flatten = nn.Flatten()

        with torch.no_grad():
            self.flattened_size = self.flatten(self.layers8910(self.layers567(
                self.layers34(self.layers12(torch.zeros(1, 1, 128, 216)))))).shape[1]

        self.classifier = nn.Sequential(
            # Adjust the size according to your mel-spectrogram dimensions after pooling
            nn.Linear(in_features=self.flattened_size, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=27)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:

        x = self.layers12(input_data)
        x = self.layers34(x)
        x = self.layers567(x)
        x = self.layers8910(x)
        x = self.flatten(x)
        x = self.classifier(x)
        # predictions = self.softmax(x)
        return x
