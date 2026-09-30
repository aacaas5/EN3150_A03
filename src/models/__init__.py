from .model_a import ModelA
from .model_b import ModelB
from .pretrained import pretrained_model

def build_model(name, weights=True):
    if name == "model_a":
        return ModelA()
    if name == "model_b":
        return ModelB()
    return pretrained_model(name, weights=weights)
