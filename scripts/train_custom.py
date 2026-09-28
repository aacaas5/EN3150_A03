import argparse
from config import ROOT, CUSTOM_EPOCHS, BATCH_SIZE
from src.training import train_run
from src.utils import read_json

def run(epochs=CUSTOM_EPOCHS, batch_size=BATCH_SIZE):
    if epochs < 20:
        raise ValueError("Final custom experiments require at least 20 epochs")
    selected = read_json(ROOT / "results/raw/selected_optimizer.json")["optimizer"]
    for name in ("model_a", "model_b"):
        path = ROOT / f"results/raw/{name}/run.json"
        if not path.exists():
            train_run(name, name, selected, epochs, batch_size)
        else:
            meta = read_json(path)
            assert meta["epochs"] == epochs and meta["batch_size"] == batch_size and meta["optimizer"] == selected

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=CUSTOM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    a = p.parse_args()
    run(a.epochs, a.batch_size)
