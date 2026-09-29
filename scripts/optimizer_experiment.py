import argparse
import pandas as pd
from config import ROOT, OPTIMIZERS, OPTIMIZER_EPOCHS, BATCH_SIZE
from src.training import train_run
from src.utils import read_json, write_json, log
from src.plotting import curves, optimizer_plot

def run(epochs=OPTIMIZER_EPOCHS, batch_size=BATCH_SIZE):
    rows = []
    for name in OPTIMIZERS:
        run_name = f"optimizer_{name}"
        path = ROOT / f"results/raw/{run_name}/run.json"
        meta = read_json(path) if path.exists() else train_run("model_b", run_name, name, epochs, batch_size)
        assert meta["epochs"] == epochs and meta["batch_size"] == batch_size
        h = pd.read_csv(path.parent / "history.csv")
        rows.append({"optimizer": name, **OPTIMIZERS[name], "epochs": epochs,
                     "best_val_accuracy": meta["best_val_accuracy"], "best_epoch": meta["best_epoch"],
                     "best_val_loss": meta["best_val_loss"], "final_val_accuracy": h.val_accuracy.iloc[-1],
                     "final_train_accuracy": h.train_accuracy.iloc[-1],
                     "final_val_loss": h.val_loss.iloc[-1],
                     "avg_epoch_time_seconds": meta["avg_epoch_time_seconds"],
                     "initial_state_sha256": meta["initial_state_sha256"]})
        curves(run_name)
    assert len({r["initial_state_sha256"] for r in rows}) == 1
    df = pd.DataFrame(rows).sort_values(["best_val_accuracy", "best_val_loss"], ascending=[False, True])
    df.to_csv(ROOT / "results/tables/optimizer_comparison.csv", index=False)
    selected = str(df.iloc[0].optimizer)
    write_json(ROOT / "results/raw/selected_optimizer.json", {"optimizer": selected,
               "criterion": "highest validation accuracy; ties resolved by lower validation loss", "epochs": epochs})
    optimizer_plot()
    log(f"Selected optimizer: {selected}; selection excludes test data.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=OPTIMIZER_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = p.parse_args()
    run(args.epochs, args.batch_size)
