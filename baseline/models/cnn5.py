import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 32, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class CNNNetwork5(Module):
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

        self.inception1 = Inception(in_channels=256)
        self.conv1x1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512,
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
            x = self.inception1(x)
            x = self.conv1x1(x)
            x = self.layer4(x)
            x = self.flatten(x)
            self.flattened_size = x.shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.flattened_size, out_features=4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p= 0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        x = self.layer1(input_data)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.inception1(x)
        x = self.conv1x1(x)
        x = self.layer4(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
