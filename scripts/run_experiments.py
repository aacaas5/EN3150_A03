"""Sequential GPU jobs; reruns preserve completed experiments."""
from scripts.inspect_models import inspect
from scripts.smoke_test import run as smoke
from scripts.optimizer_experiment import run as optimizers
from scripts.train_custom import run as custom
from scripts.train_pretrained import run as pretrained
from scripts.evaluate_all import run as evaluate

if __name__ == "__main__":
    inspect()
    smoke()
    optimizers()
    custom()
    pretrained()
    evaluate()
