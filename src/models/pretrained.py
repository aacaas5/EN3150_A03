"""Explicit torchvision ImageNet weight versions; never silently fall back."""
from torch import nn
from torchvision.models import (mobilenet_v2, efficientnet_b0,
                                MobileNet_V2_Weights, EfficientNet_B0_Weights)

def pretrained_model(name, weights=True, num_classes=10):
    if name == "mobilenet_v2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2 if weights else None)
    elif name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if weights else None)
    else:
        raise ValueError(name)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model

def set_stage(model, head_only):
    for p in model.features.parameters():
        p.requires_grad_(not head_only)
    for p in model.classifier.parameters():
        p.requires_grad_(True)
