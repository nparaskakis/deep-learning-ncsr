import torch
from torch import nn
import torch.nn.functional as F


class VGG16Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64,
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64,
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128,
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128,
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256,
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256,
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256,
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512,
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512,
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512,
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512,
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512,
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512,
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        ])

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 4 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 50)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        x = input_data
        for block in self.conv_blocks:
            x = block(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)
        logits = self.fc3(x)
        predictions = self.softmax(logits)
        return predictions
