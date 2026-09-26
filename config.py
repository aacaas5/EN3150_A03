"""Central, deliberately small experiment configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 3150
BATCH_SIZE = 128
CUSTOM_EPOCHS = 25
OPTIMIZER_EPOCHS = 10
HEAD_EPOCHS = 2
FINETUNE_EPOCHS = 8
IMAGE_SIZE = 64
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
OPTIMIZERS = {"sgd": {"lr": 0.01, "momentum": 0.0},
              "momentum": {"lr": 0.01, "momentum": 0.9},
              "adam": {"lr": 0.001, "momentum": 0.0}}
