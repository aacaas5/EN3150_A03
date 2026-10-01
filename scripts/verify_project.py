"""Fail-fast audit of saved evidence; independent metric recomputation."""
import argparse
import math
import pandas as pd
import torch
from config import ROOT
from src.utils import read_json, write_json, sha256
from src.models import build_model
from src.metrics import classification_metrics
from src.profiling import parameter_counts

def verify(require_report=False):
    checks = []
    def check(name, passed):
        if not passed:
            raise AssertionError(name)
        checks.append({"check": name, "passed": True})
    info = read_json(ROOT / "data/splits/dataset_info.json")
    splits = {s: read_json(ROOT / f"data/splits/{s}_indices.json") for s in ("train", "val", "test")}
    check("Exact 70/15/15 sample sizes", [len(v) for v in splits.values()] == [18900, 4050, 4050])
    check("Disjoint and exhaustive sample indices", sorted(sum(splits.values(), [])) == list(range(27000)))
    manifest = read_json(ROOT / "data/splits/sample_manifest.json")
    check("Manifest hash", sha256(ROOT / "data/splits/sample_manifest.json") == info["manifest_sha256"])
    check("No pixel-identical images across splits", read_json(ROOT / "data/splits/content_audit.json")["cross_split_identical_images"] == 0)
    for i, name in enumerate(info["classes"]):
        c = info["class_counts"][name]
        check(f"Class {name} stratification", all(sum(manifest[j]["label"] == i for j in splits[s]) == c[s] for s in splits))
    selected = read_json(ROOT / "results/raw/selected_optimizer.json")["optimizer"]
    opts = pd.read_csv(ROOT / "results/tables/optimizer_comparison.csv")
    check("Three optimizers compared with identical initialization", set(opts.optimizer) == {"sgd", "momentum", "adam"} and opts.initial_state_sha256.nunique() == 1 and opts.epochs.nunique() == 1)
    winner = opts.sort_values(["best_val_accuracy", "best_val_loss"], ascending=[False, True]).iloc[0].optimizer
    check("Selection uses best validation result", selected == winner)
    for opt in opts.optimizer:
        meta = read_json(ROOT / f"results/raw/optimizer_{opt}/run.json")
        check(f"Optimizer {opt} same samples", meta["split_sha256"] == info["split_sha256"])
        check(f"Optimizer {opt} controlled configuration", meta["seed"] == info["seed"] and meta["model"] == "model_b" and meta["batch_size"] == read_json(ROOT / "results/raw/optimizer_sgd/run.json")["batch_size"])
        h = pd.read_csv(ROOT / f"results/raw/optimizer_{opt}/history.csv")
        check(f"Optimizer {opt} complete history", len(h) == meta["epochs"] and h.isna().sum().sum() == 0)
    all_results = pd.read_csv(ROOT / "results/tables/final_comparison.csv").set_index("model")
    for name in ("model_a", "model_b", "mobilenet_v2", "efficientnet_b0"):
        raw = ROOT / f"results/raw/{name}"
        meta = read_json(raw / "run.json")
        h = pd.read_csv(raw / "history.csv")
        check(f"{name}: identical split hashes", meta["split_sha256"] == info["split_sha256"])
        check(f"{name}: full finite history", len(h) == meta["epochs"] and h.isna().sum().sum() == 0 and (h.epoch_seconds > 0).all())
        check(f"{name}: validation checkpoint selection", math.isclose(h.val_accuracy.max(), meta["best_val_accuracy"]))
        check(f"{name}: checkpoint hash", sha256(ROOT / meta["checkpoint"]) == meta["checkpoint_sha256"])
        check(f"{name}: resolution cap", meta["input_size"] == 64)
        if name in ("model_a", "model_b"):
            check(f"{name}: at least 20 epochs with chosen optimizer", meta["epochs"] >= 20 and meta["optimizer"] == selected)
        else:
            check(f"{name}: two-stage pretrained training", meta["pretrained_weights"] is not None and set(h.stage) == {"head", "finetune"} and meta["stage_trainable_parameters"]["head"] < meta["stage_trainable_parameters"]["finetune"])
        model = build_model(name, weights=False)
        model.load_state_dict(torch.load(ROOT / meta["checkpoint"], weights_only=True))
        counts = parameter_counts(model)
        check(f"{name}: parameter count", counts["total_parameters"] == all_results.loc[name, "total_parameters"])
        if name == "model_b":
            check("Model B below 100,000 trainable parameters", counts["trainable_parameters"] < 100000)
            pilot = pd.read_csv(ROOT / f"results/raw/optimizer_{selected}/history.csv")
            columns = ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]
            check("Model B reproduces selected optimizer pilot prefix", (h[columns].iloc[:len(pilot)].reset_index(drop=True) - pilot[columns]).abs().to_numpy().max() < 1e-9)
        p = pd.read_csv(raw / "test_predictions.csv")
        check(f"{name}: predictions on exact test indices", p.sample_index.tolist() == splits["test"])
        check(f"{name}: test truth from manifest", p.true_label.tolist() == [manifest[j]["label"] for j in splits["test"]])
        metrics, cm, _ = classification_metrics(p.true_label, p.predicted_label, info["classes"])
        for metric, value in metrics.items():
            check(f"{name}: recomputed {metric}", math.isclose(value, all_results.loc[name, metric], abs_tol=1e-12))
        check(f"{name}: confusion matrix", (cm == pd.read_csv(raw / "confusion_matrix.csv", index_col=0).values).all() and cm.sum() == 4050)
        check(f"{name}: measured file size", (ROOT / meta["checkpoint"]).stat().st_size == all_results.loc[name, "model_size_bytes"])
        check(f"{name}: measured epoch time", math.isclose(h.epoch_seconds.mean(), all_results.loc[name, "avg_epoch_time_seconds"]))
        for folder, suffix in (("figures", "_curves"), ("confusion_matrices", "")):
            check(f"{name}: {folder} figure", (ROOT / f"results/{folder}/{name}{suffix}.pdf").exists())
    for name, subset in (("custom_models", ["model_a", "model_b"]), ("pretrained_models", ["mobilenet_v2", "efficientnet_b0"])):
        table = pd.read_csv(ROOT / f"results/tables/{name}.csv").set_index("model")
        check(f"{name} consistent with final comparison", table.equals(all_results.loc[subset]))
    check("No Git repository created", not (ROOT / ".git").exists())
    check("README present", (ROOT / "README.md").exists())
    if require_report:
        provenance = read_json(ROOT / "report/result_provenance.json")
        for relative, digest in provenance.items():
            check(f"Report source {relative} unchanged", sha256(ROOT / relative) == digest)
        check("LaTeX source and compiled PDF exist", (ROOT / "report/main.tex").exists() and (ROOT / "report/main.pdf").exists())
        latex_log = (ROOT / "report/main.log").read_text(errors="replace")
        check("No unresolved LaTeX errors or references", "! " not in latex_log and "undefined" not in latex_log.lower() and "Overfull" not in latex_log)
    write_json(ROOT / "results/verification.json", {"passed": True, "report_required": require_report, "checks": checks})
    print(f"PASS: {len(checks)} checks (report={require_report})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--require-report", action="store_true")
    verify(p.parse_args().require_report)
