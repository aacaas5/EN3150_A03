import argparse
from config import ROOT, HEAD_EPOCHS, FINETUNE_EPOCHS, BATCH_SIZE
from src.training import train_run
from src.utils import read_json

def run(head_epochs=HEAD_EPOCHS, finetune_epochs=FINETUNE_EPOCHS, batch_size=BATCH_SIZE):
    assert head_epochs >= 1 and finetune_epochs >= 1
    for name in ("mobilenet_v2", "efficientnet_b0"):
        path = ROOT / f"results/raw/{name}/run.json"
        if not path.exists():
            train_run(name, name, "adam", batch_size=batch_size,
                      head_epochs=head_epochs, finetune_epochs=finetune_epochs)
        else:
            meta = read_json(path)
            assert meta["head_epochs"] == head_epochs and meta["finetune_epochs"] == finetune_epochs and meta["batch_size"] == batch_size

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--head-epochs", type=int, default=HEAD_EPOCHS)
    p.add_argument("--finetune-epochs", type=int, default=FINETUNE_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    a = p.parse_args()
    run(a.head_epochs, a.finetune_epochs, a.batch_size)
