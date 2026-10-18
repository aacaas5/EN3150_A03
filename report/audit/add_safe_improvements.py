"""Create no-retraining evidence additions from the completed experiment files."""
from __future__ import annotations

import json
import math
import platform
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from src.models import build_model
from src.utils import read_json, write_json, sha256

MODELS = ("model_a", "model_b", "mobilenet_v2", "efficientnet_b0")
DISPLAY = {
    "model_a": "Model A",
    "model_b": "Model B",
    "mobilenet_v2": "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0",
}


def extended_metrics():
    final_path = ROOT / "results/tables/final_comparison.csv"
    final = pd.read_csv(final_path).set_index("model")
    for name in MODELS:
        prediction_path = ROOT / f"results/raw/{name}/test_predictions.csv"
        predictions = pd.read_csv(prediction_path)
        truth = predictions.true_label.to_numpy()
        predicted = predictions.predicted_label.to_numpy()
        macro = precision_recall_fscore_support(
            truth, predicted, average="macro", zero_division=0
        )
        weighted = precision_recall_fscore_support(
            truth, predicted, average="weighted", zero_division=0
        )
        derived = {
            "macro_f1": float(macro[2]),
            "weighted_precision": float(weighted[0]),
            "weighted_recall": float(weighted[1]),
            "weighted_f1": float(weighted[2]),
        }
        for key, value in derived.items():
            final.loc[name, key] = value
        metrics_path = ROOT / f"results/raw/{name}/metrics.json"
        metrics = read_json(metrics_path)
        metrics.update(derived)
        write_json(metrics_path, metrics)

    final = final.reset_index()
    final.to_csv(final_path, index=False)
    final[final.model.isin(("model_a", "model_b"))].to_csv(
        ROOT / "results/tables/custom_models.csv", index=False
    )
    final[final.model.isin(("mobilenet_v2", "efficientnet_b0"))].to_csv(
        ROOT / "results/tables/pretrained_models.csv", index=False
    )
    master_columns = [
        "model", "test_accuracy", "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1", "total_parameters",
        "model_size_bytes", "model_size_mb", "conv_linear_macs",
        "cpu_latency_ms_mean", "cpu_latency_ms_p95",
    ]
    final[master_columns].to_csv(ROOT / "results/tables/master_comparison.csv", index=False)
    return final


def activation_estimate():
    rows = []
    for name in MODELS:
        meta = read_json(ROOT / f"results/raw/{name}/run.json")
        model = build_model(name, weights=False)
        model.load_state_dict(torch.load(ROOT / meta["checkpoint"], weights_only=True))
        model.eval()
        activations = []
        handles = []

        def make_hook(label):
            def hook(_module, _inputs, output):
                if isinstance(output, torch.Tensor):
                    activations.append((label, list(output.shape), output.numel()))
            return hook

        for label, layer in model.named_modules():
            if label and not list(layer.children()):
                handles.append(layer.register_forward_hook(make_hook(label)))
        with torch.inference_mode():
            model(torch.zeros(1, 3, 64, 64))
        for handle in handles:
            handle.remove()
        layer, shape, elements = max(activations, key=lambda item: item[2])
        rows.append({
            "model": name,
            "layer": layer,
            "shape": "x".join(map(str, shape)),
            "elements": elements,
            "fp32_bytes": elements * 4,
            "fp32_kib": elements * 4 / 1024,
            "fp32_mib": elements * 4 / 1024**2,
            "scope": "largest single leaf-layer output tensor; not total peak runtime RAM",
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results/tables/activation_memory_estimate.csv", index=False)
    return frame


def latency_rebenchmark():
    torch.set_num_threads(4)
    rows = []
    for name in MODELS:
        meta = read_json(ROOT / f"results/raw/{name}/run.json")
        model = build_model(name, weights=False)
        model.load_state_dict(torch.load(ROOT / meta["checkpoint"], weights_only=True))
        model.cpu().eval()
        sample = torch.zeros(1, 3, 64, 64)
        values = []
        with torch.inference_mode():
            for _ in range(20):
                model(sample)
            for _ in range(100):
                start = time.perf_counter_ns()
                model(sample)
                values.append((time.perf_counter_ns() - start) / 1e6)
        series = pd.Series(values)
        rows.append({
            "model": name,
            "mean_ms": float(series.mean()),
            "median_ms": float(series.median()),
            "p95_ms": float(series.quantile(0.95)),
            "std_ms": float(statistics.stdev(values)),
            "warmups": 20,
            "repeats": 100,
            "batch_size": 1,
            "dtype": "FP32",
            "cpu_threads": 4,
            "host": platform.processor() or "Windows host CPU",
            "checkpoint_sha256": sha256(ROOT / meta["checkpoint"]),
            "note": "fresh no-retraining CPU benchmark; inputs pre-normalized; forward pass only",
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results/tables/cpu_latency_rebenchmark.csv", index=False)
    return frame


def resource_figure(final):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    x_columns = ("total_parameters", "conv_linear_macs")
    x_labels = ("Parameters (log scale)", "Conv/Linear MACs per image (log scale)")
    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756")
    for axis, column, label in zip(axes, x_columns, x_labels):
        for color, (_, row) in zip(colors, final.iterrows()):
            axis.scatter(row[column], row.test_accuracy * 100, s=65, color=color, zorder=3)
            axis.annotate(DISPLAY[row.model], (row[column], row.test_accuracy * 100),
                          xytext=(5, 5), textcoords="offset points", fontsize=8)
        axis.set_xscale("log")
        axis.set_xlabel(label)
        axis.set_ylabel("Test accuracy (%)")
        axis.grid(True, which="both", alpha=0.25)
    fig.suptitle("Accuracy and measured resource cost")
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"results/figures/accuracy_resource_pareto.{suffix}", dpi=220)
        fig.savefig(ROOT / f"report/figures/accuracy_resource_pareto.{suffix}", dpi=220)
    plt.close(fig)


def representative_errors():
    info = read_json(ROOT / "data/splits/dataset_info.json")
    classes = info["classes"]
    predictions = pd.read_csv(ROOT / "results/raw/model_a/test_predictions.csv")
    cache = torch.load(ROOT / "data/rgb_uint8.pt", weights_only=True)
    preferred = [
        ("HerbaceousVegetation", "PermanentCrop"),
        ("Highway", "Industrial"),
        ("PermanentCrop", "HerbaceousVegetation"),
        ("River", "Highway"),
    ]
    chosen = []
    for true_name, predicted_name in preferred:
        true_id, predicted_id = classes.index(true_name), classes.index(predicted_name)
        matches = predictions[(predictions.true_label == true_id) &
                              (predictions.predicted_label == predicted_id)]
        chosen.extend(matches.head(2).to_dict("records"))
    if len(chosen) < 6:
        extras = predictions[predictions.true_label != predictions.predicted_label]
        used = {row["sample_index"] for row in chosen}
        chosen.extend(row for row in extras.to_dict("records")
                      if row["sample_index"] not in used) 
    chosen = chosen[:8]
    if not chosen:
        return
    fig, axes = plt.subplots(2, 4, figsize=(9.2, 5.0), constrained_layout=True)
    for axis, row in zip(axes.flat, chosen):
        image = cache["images"][int(row["sample_index"])].permute(1, 2, 0).numpy()
        axis.imshow(image)
        axis.set_title(
            f"True: {classes[int(row['true_label'])]}\nPredicted: {classes[int(row['predicted_label'])]}",
            fontsize=8,
        )
        axis.axis("off")
    for axis in axes.flat[len(chosen):]:
        axis.axis("off")
    fig.suptitle("Representative Model A errors from measured confusion pairs", fontsize=11)
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"results/figures/representative_errors.{suffix}", dpi=220)
        fig.savefig(ROOT / f"report/figures/representative_errors.{suffix}", dpi=220)
    plt.close(fig)
    pd.DataFrame(chosen).to_csv(ROOT / "results/tables/representative_errors.csv", index=False)


def latex_tables(final, activation, latency):
    master_lines = [
        r"\begin{landscape}",
        r"\begin{table}[p]",
        r"\centering\small",
        r"\caption{Master held-out test and resource comparison. Size is the measured FP32 state-dictionary file. Latency is the original batch-one host benchmark.}",
        r"\label{tab:master-comparison}",
        r"\begin{tabular}{lrrrrrrrrr}\toprule",
        r"Model & Acc. (\%) & Macro P & Macro R & Macro F1 & Params & Size (MiB) & MACs (M) & Mean (ms) & p95 (ms)\\\midrule",
    ]
    for _, row in final.iterrows():
        master_lines.append(
            f"{DISPLAY[row.model]} & {row.test_accuracy*100:.2f} & {row.macro_precision*100:.2f} & "
            f"{row.macro_recall*100:.2f} & {row.macro_f1*100:.2f} & {int(row.total_parameters):,} & "
            f"{row.model_size_mb:.3f} & {row.conv_linear_macs/1e6:.3f} & "
            f"{row.cpu_latency_ms_mean:.3f} & {row.cpu_latency_ms_p95:.3f}\\\\"
        )
    master_lines += [r"\bottomrule\end{tabular}", r"\end{table}", r"\end{landscape}"]
    (ROOT / "report/tables/master_comparison.tex").write_text("\n".join(master_lines) + "\n", encoding="utf-8")

    activation_lines = [
        r"\begin{table}[t]", r"\centering\small",
        r"\caption{Largest observed single intermediate activation tensor for batch-one FP32 inference. This is not total peak runtime RAM.}",
        r"\label{tab:activation-memory}",
        r"\begin{tabular}{lrrr}\toprule",
        r"Model & Shape & Elements & FP32 (KiB)\\\midrule",
    ]
    for _, row in activation.iterrows():
        activation_lines.append(
            f"{DISPLAY[row.model]} & {row['shape'].replace('x', r'$\times$')} & {int(row.elements):,} & {row.fp32_kib:.1f}\\\\"
        )
    activation_lines += [r"\bottomrule\end{tabular}", r"\end{table}"]
    (ROOT / "report/tables/activation_memory.tex").write_text("\n".join(activation_lines) + "\n", encoding="utf-8")

    latency_lines = [
        r"\begin{table}[t]", r"\centering\small",
        r"\caption{Fresh CPU-only repeat of the original forward-pass benchmark. It is additional timing evidence and does not replace the original reported latency.}",
        r"\label{tab:latency-repeat}",
        r"\begin{tabular}{lrrrr}\toprule",
        r"Model & Mean & Median & p95 & Std. dev. (ms)\\\midrule",
    ]
    for _, row in latency.iterrows():
        latency_lines.append(
            f"{DISPLAY[row.model]} & {row.mean_ms:.3f} & {row.median_ms:.3f} & {row.p95_ms:.3f} & {row.std_ms:.3f}\\\\"
        )
    latency_lines += [r"\bottomrule\end{tabular}", r"\end{table}"]
    (ROOT / "report/tables/latency_rebenchmark.tex").write_text("\n".join(latency_lines) + "\n", encoding="utf-8")


def main():
    final = extended_metrics()
    activation = activation_estimate()
    latency = latency_rebenchmark()
    resource_figure(final)
    representative_errors()
    latex_tables(final, activation, latency)
    write_json(ROOT / "results/safe_improvements.json", {
        "no_retraining": True,
        "source_predictions": {name: sha256(ROOT / f"results/raw/{name}/test_predictions.csv") for name in MODELS},
        "source_checkpoints": {name: sha256(ROOT / read_json(ROOT / f"results/raw/{name}/run.json")["checkpoint"]) for name in MODELS},
        "outputs": [
            "results/tables/master_comparison.csv",
            "results/tables/activation_memory_estimate.csv",
            "results/tables/cpu_latency_rebenchmark.csv",
            "results/tables/representative_errors.csv",
            "results/figures/accuracy_resource_pareto.pdf",
            "results/figures/representative_errors.pdf",
        ],
    })
    print("Safe no-retraining additions generated.")


if __name__ == "__main__":
    main()
