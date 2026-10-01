import torch
from src.dataset import loaders, preprocess
from src.training import train_run
from src.metrics import classification_metrics
from src.plotting import curves, confusion_figure
from src.utils import read_json, write_json
from config import ROOT

def run():
    x, y = next(iter(loaders(splits=("train",))["train"]))
    assert x.shape == (128, 3, 64, 64) and y.shape == (128,)
    assert torch.equal(preprocess(x, "cpu"), preprocess(x, "cpu"))
    metrics, cm, _ = classification_metrics([0, 1], [0, 1], ["a", "b"])
    assert all(v == 1 for v in metrics.values())
    # Synthetic sanity check is isolated from all reported experiment results.
    confusion_figure(cm, ["a", "b"], "smoke_synthetic")
    train_run("model_b", "smoke", epochs=1, smoke=True)
    curves("smoke")
    write_json(ROOT / "results/raw/smoke/sanity.json", {"passed": True, "batch_shape": list(x.shape)})

if __name__ == "__main__":
    run()
