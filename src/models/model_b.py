"""A stem convolution followed by reusable depthwise/pointwise blocks."""
from torch import nn

class DepthwiseSeparableConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
                         nn.BatchNorm2d(cin), nn.ReLU(),
                         nn.Conv2d(cin, cout, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU())

class ModelB(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Sequential(nn.Conv2d(3, 24, 3, padding=1, bias=False),
                          nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(2)),
            nn.Sequential(DepthwiseSeparableConv(24, 48), nn.MaxPool2d(2)),
            nn.Sequential(DepthwiseSeparableConv(48, 96), nn.MaxPool2d(2)),
            nn.Sequential(DepthwiseSeparableConv(96, 128), nn.MaxPool2d(2)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(128, num_classes)
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) < 100000

    def forward(self, x):
        return self.classifier(self.flatten(self.pool(self.features(x))))
