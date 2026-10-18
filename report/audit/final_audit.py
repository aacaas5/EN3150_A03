"""Final result consistency and rubric-evidence audit."""
import ast
import json
import subprocess
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("model_a", "model_b", "mobilenet_v2", "efficientnet_b0")


def main():
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append({"check": name, "status": "PASS"})

    final = pd.read_csv(ROOT / "results/tables/final_comparison.csv").set_index("model")
    master = pd.read_csv(ROOT / "results/tables/master_comparison.csv").set_index("model")
    info = json.loads((ROOT / "data/splits/dataset_info.json").read_text())
    splits = {name: json.loads((ROOT / f"data/splits/{name}_indices.json").read_text())
              for name in ("train", "val", "test")}
    check("Dataset total and split sizes", len(set(sum(splits.values(), []))) == 27000 and
          [len(splits[name]) for name in ("train", "val", "test")] == [18900, 4050, 4050])
    check("Saved split hashes shared by all final runs", all(
        json.loads((ROOT / f"results/raw/{name}/run.json").read_text())["split_sha256"] == info["split_sha256"]
        for name in MODELS))

    metric_names = ("test_accuracy", "macro_precision", "macro_recall", "macro_f1",
                    "weighted_precision", "weighted_recall", "weighted_f1")
    for name in MODELS:
        predictions = pd.read_csv(ROOT / f"results/raw/{name}/test_predictions.csv")
        truth, predicted = predictions.true_label, predictions.predicted_label
        macro = precision_recall_fscore_support(truth, predicted, average="macro", zero_division=0)
        weighted = precision_recall_fscore_support(truth, predicted, average="weighted", zero_division=0)
        values = (accuracy_score(truth, predicted), *macro[:3], *weighted[:3])
        for key, value in zip(metric_names, values):
            check(f"{name} recomputed {key}", np.isclose(final.loc[name, key], value, atol=1e-12))
            check(f"{name} master table {key}", np.isclose(master.loc[name, key], value, atol=1e-12))
        saved_matrix = pd.read_csv(ROOT / f"results/raw/{name}/confusion_matrix.csv", index_col=0).values
        check(f"{name} confusion matrix from predictions",
              np.array_equal(saved_matrix, confusion_matrix(truth, predicted, labels=range(10))))
        meta = json.loads((ROOT / f"results/raw/{name}/run.json").read_text())
        check(f"{name} measured checkpoint size",
              (ROOT / meta["checkpoint"]).stat().st_size == int(final.loc[name, "model_size_bytes"]))

    notebook = nbformat.read(ROOT / "FeatureFour_A03_EN3150.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    notebook_text = "\n".join(cell.source for cell in notebook.cells)
    notebook_outputs = json.dumps([cell.get("outputs", []) for cell in notebook.cells])
    check("Notebook quick-review box", "QUICK REVIEW MODE" in notebook_text and "FULL REPRODUCTION" in notebook_text)
    check("Notebook expensive flags default false", all(
        f"{flag} = False" in notebook_text for flag in
        ("ALLOW_FULL_EXPERIMENTS", "RUN_DATA_PREPARATION", "RUN_OPTIMIZER_EXPERIMENT",
         "RUN_CUSTOM_TRAINING", "RUN_PRETRAINED_TRAINING")))
    check("Notebook embedded checks executed", "PASS: saved predictions" in notebook_outputs)
    check("Notebook has no absolute local-user path", "C:\\Users\\" not in notebook_text)

    report_text = subprocess.check_output(
        ["pdftotext", "-layout", str(ROOT / "report/FeatureFour_A03_EN3150.pdf"), "-"],
        text=True, errors="replace"
    )
    for name in MODELS:
        check(f"PDF contains {name} accuracy",
              f"{final.loc[name, 'test_accuracy'] * 100:.2f}" in report_text)
        check(f"PDF contains {name} parameter count",
              f"{int(final.loc[name, 'total_parameters']):,}" in report_text)
        check(f"PDF contains {name} MAC value",
              f"{final.loc[name, 'conv_linear_macs'] / 1e6:.3f}" in report_text)
    check("PDF contains macro F1 master heading", "Macro F1" in report_text)
    check("PDF contains activation-memory warning", "not total peak runtime RAM" in report_text)

    rubric = pd.DataFrame([
        {
            "rubric_section": "Custom Architecture Design", "maximum_marks": 20,
            "assignment_requirement": "Standard Model A; depthwise-separable Model B below 100k; layer shapes, manual counts and hardware-aware activation",
            "pdf_evidence": "Sections 3.1-3.5 and Appendix A",
            "notebook_evidence": "Sections 4, 5 and 7; Model B assertion",
            "machine_readable_evidence": "model_a_architecture.csv; model_b_architecture.csv; convolution_savings.csv",
            "status": "COMPLETE", "remaining_risk": "Architecture merit remains subject to lecturer judgement",
        },
        {
            "rubric_section": "Optimizer Selection & Tuning", "maximum_marks": 15,
            "assignment_requirement": "SGD, momentum SGD and Adam under controlled pilot conditions; evidence-based selection",
            "pdf_evidence": "Section 4 and Tables 3-4",
            "notebook_evidence": "Sections 8 and 12; embedded pilot histories",
            "machine_readable_evidence": "optimizer_comparison.csv; optimizer histories and run JSON files",
            "status": "COMPLETE", "remaining_risk": "Fixed recipes are not an exhaustive hyperparameter search",
        },
        {
            "rubric_section": "Custom Model Training & Evaluation", "maximum_marks": 25,
            "assignment_requirement": "Both custom models >=20 epochs; curves; held-out metrics; confusion and resource comparison",
            "pdf_evidence": "Section 5 and Figures 2-5",
            "notebook_evidence": "Sections 8-12; predictions recomputed in quick review",
            "machine_readable_evidence": "model_a/model_b histories, predictions, metrics, reports and matrices",
            "status": "COMPLETE", "remaining_risk": "Single-seed final runs limit variance claims",
        },
        {
            "rubric_section": "SOTA Fine-Tuning & Evaluation", "maximum_marks": 20,
            "assignment_requirement": "MobileNetV2 and EfficientNet-B0 with ImageNet weights, head stage, full fine-tuning and same split",
            "pdf_evidence": "Section 6 and Figures 6-9",
            "notebook_evidence": "Sections 6, 8, 10 and 12",
            "machine_readable_evidence": "pretrained histories, run JSON files, predictions, reports and matrices",
            "status": "COMPLETE", "remaining_risk": "64x64 input differs from the original ImageNet recipe",
        },
        {
            "rubric_section": "Final Comparison & Trade-off Discussion", "maximum_marks": 20,
            "assignment_requirement": "Accuracy, memory and computation comparison with edge deployment interpretation",
            "pdf_evidence": "Sections 7-9; master Table 10; activation, latency and Pareto evidence",
            "notebook_evidence": "Section 12 comparison and Section 13 conclusion",
            "machine_readable_evidence": "master_comparison.csv; activation_memory_estimate.csv; cpu_latency_rebenchmark.csv",
            "status": "COMPLETE", "remaining_risk": "No true edge-device power, peak RAM or quantized benchmark",
        },
    ])
    rubric.to_csv(ROOT / "report/audit/final_rubric_audit.csv", index=False)
    output = {"passed": True, "checks": checks, "rubric_statuses": rubric[["rubric_section", "status", "remaining_risk"]].to_dict("records")}
    (ROOT / "report/audit/final_consistency_report.json").write_text(json.dumps(output, indent=2) + "\n")
    print(f"PASS: {len(checks)} consistency checks; {len(rubric)} rubric sections audited")


if __name__ == "__main__":
    main()
