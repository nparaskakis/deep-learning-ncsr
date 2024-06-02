import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F
from torch.nn import Module


class FCNNNetwork1(Module):

    def __init__(self, dim1, dim2, num_classes):

        super().__init__()
        
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, dim1, dim2)
            x = self.flatten(dummy_input)
            self.flattened_size = x.shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.flattened_size, out_features=8192),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=8192, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        x = self.flatten(input_data)
        x = self.classifier(x)
        return x