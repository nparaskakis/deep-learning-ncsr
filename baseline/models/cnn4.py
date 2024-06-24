import torch
from torch import nn
from torch.nn import Module

class CNNNetwork4(Module):
    def __init__(self, dim1, dim2, num_classes):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, dim1, dim2)
            x = self.layer1(dummy_input)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.flatten(x)
            self.flattened_size = x.shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.flattened_size, out_features=4096),
            nn.BatchNorm1d(num_features=4096),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p= 0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.BatchNorm1d(num_features=4096),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        x = self.layer1(input_data)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)      
        x = self.classifier(x)
        return x
