"""Test evaluation is separate from optimizer selection and training."""
import pandas as pd
import torch
from config import ROOT
from src.dataset import loaders, preprocess
from src.models import build_model
from src.metrics import classification_metrics
from src.profiling import architecture_table, inference_latency
from src.plotting import confusion_figure, curves
from src.utils import read_json, write_json, sha256, seed_everything

def evaluate(run):
    seed_everything()
    meta = read_json(ROOT / f"results/raw/{run}/run.json")
    assert not meta["smoke"]
    ckpt = ROOT / meta["checkpoint"]
    assert sha256(ckpt) == meta["checkpoint_sha256"]
    info = read_json(ROOT / "data/splits/dataset_info.json")
    assert meta["split_sha256"] == info["split_sha256"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(meta["model"], weights=False).to(device)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    truth, predicted = [], []
    with torch.inference_mode():
        for x, y in loaders(splits=("test",))["test"]:
            predicted.extend(model(preprocess(x, device)).argmax(1).cpu().tolist())
            truth.extend(y.tolist())
    metrics, matrix, report = classification_metrics(truth, predicted, info["classes"])
    out = ROOT / f"results/raw/{run}"
    pd.DataFrame({"sample_index": read_json(ROOT / "data/splits/test_indices.json"),
                  "true_label": truth, "predicted_label": predicted}).to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame(report).T.to_csv(out / "classification_report.csv")
    pd.DataFrame(matrix, index=info["classes"], columns=info["classes"]).to_csv(out / "confusion_matrix.csv")
    row = {"model": meta["model"], **architecture_table(model, meta["model"]),
           "model_size_bytes": ckpt.stat().st_size, "model_size_kb": ckpt.stat().st_size / 1024,
           "model_size_mb": ckpt.stat().st_size / 1024**2,
           "avg_epoch_time_seconds": meta["avg_epoch_time_seconds"],
           "best_epoch": meta["best_epoch"], "epochs": meta["epochs"],
           **metrics, **inference_latency(model)}
    write_json(out / "metrics.json", row)
    confusion_figure(matrix, info["classes"], run)
    curves(run)
    return row
