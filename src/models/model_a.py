"""Matched-width standard-convolution baseline with global average pooling."""
from torch import nn

class ModelA(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        blocks = []
        channels = [3, 24, 48, 96, 128]
        for cin, cout in zip(channels[:-1], channels[1:]):
            blocks.append(nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                          nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2)))
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.flatten(self.pool(self.features(x))))
