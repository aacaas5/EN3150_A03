"""Headless publication figures, PDF and 220-dpi PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import ROOT

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def curves(run):
    df = pd.read_csv(ROOT / f"results/raw/{run}/history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, metric, title in zip(axes, ("loss", "accuracy"), ("Cross-entropy loss", "Accuracy")):
        for split in ("train", "val"):
            ax.plot(df.epoch, df[f"{split}_{metric}"], label=split)
        ax.set(xlabel="Epoch", ylabel=title, title=run.replace("_", " "))
        ax.legend()
        ax.grid(alpha=.2)
    save(fig, ROOT / "results/figures", f"{run}_curves")

def confusion_figure(matrix, classes, name):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes, xlabel="Predicted class", ylabel="True class",
           title=name.replace("_", " "))
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=.046)
    save(fig, ROOT / "results/confusion_matrices", name)

def optimizer_plot():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for name in ("sgd", "momentum", "adam"):
        df = pd.read_csv(ROOT / f"results/raw/optimizer_{name}/history.csv")
        axes[0].plot(df.epoch, df.val_loss, label=name)
        axes[1].plot(df.epoch, df.val_accuracy, label=name)
    for ax, ylabel in zip(axes, ("Validation loss", "Validation accuracy")):
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.legend()
        ax.grid(alpha=.2)
    save(fig, ROOT / "results/figures", "optimizer_comparison")

def final_plots(df):
    labels = [s.replace("_", " ") for s in df.model]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(labels, df.test_accuracy * 100, color=["#4c78a8", "#54a24b", "#eeca3b", "#f58518"])
    ax.set(ylabel="Test accuracy (%)", ylim=(0, 100))
    save(fig, ROOT / "results/figures", "final_accuracy")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].bar(labels, df.total_parameters)
    axes[0].set(ylabel="Parameters (log scale)", yscale="log")
    axes[1].bar(labels, df.model_size_mb)
    axes[1].set(ylabel="FP32 state dictionary (MiB)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    save(fig, ROOT / "results/figures", "final_resources")
