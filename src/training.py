"""Shared FP32 training loop. Best checkpoint is chosen only by validation.

Epoch timings cover training and validation, including batch preprocessing,
and exclude checkpoint/CSV writes. CUDA is synchronized at timing boundaries.
No scheduler or early stopping is used, so optimizer settings remain explicit.
"""
import time
import platform
import pandas as pd
import torch
import torchvision
from torch import nn
from config import ROOT, SEED, BATCH_SIZE, OPTIMIZERS
from src.dataset import loaders, preprocess
from src.models import build_model
from src.models.pretrained import set_stage
from src.profiling import parameter_counts
from src.utils import seed_everything, read_json, write_json, log, sha256

def make_optimizer(model, name, lr=None):
    settings = OPTIMIZERS[name]
    lr = settings["lr"] if lr is None else lr
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=0)
    return torch.optim.SGD(params, lr=lr, momentum=settings["momentum"], weight_decay=0)

def epoch_pass(model, loader, device, optimizer=None, head_only=False, max_batches=None):
    training = optimizer is not None
    model.train(training)
    if training and head_only:
        # Frozen feature BN running statistics must also stay fixed.
        model.features.eval()
    loss_sum, correct, count = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(training):
        for batch, (images, labels) in enumerate(loader):
            if max_batches is not None and batch >= max_batches:
                break
            x = preprocess(images, device, training)
            y = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.detach().item() * len(y)
            correct += (logits.detach().argmax(1) == y).sum().item()
            count += len(y)
    return loss_sum / count, correct / count

def train_run(model_name, run_name, optimizer_name="adam", epochs=25,
              batch_size=BATCH_SIZE, head_epochs=0, finetune_epochs=0,
              smoke=False):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = ROOT / f"results/raw/{run_name}"
    raw.mkdir(parents=True, exist_ok=True)
    ckpt = ROOT / f"checkpoints/{run_name}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    if (raw / "run.json").exists() and not smoke:
        raise FileExistsError(f"Completed run exists: {run_name}; preserve it or choose another run name")
    model = build_model(model_name).to(device)
    initial_hash = __import__('hashlib').sha256(b"".join(
        t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
    data = loaders(batch_size, splits=("train", "val"))
    transfer = head_epochs > 0
    total_epochs = head_epochs + finetune_epochs if transfer else epochs
    history, stage_counts = [], {}
    best_acc, best_loss, best_epoch = -1, float("inf"), 0
    optimizer = None
    for epoch in range(1, total_epochs + 1):
        head_only = transfer and epoch <= head_epochs
        stage = "head" if head_only else ("finetune" if transfer else "full")
        if epoch == 1 or (transfer and epoch == head_epochs + 1):
            if transfer:
                set_stage(model, head_only)
            optimizer = make_optimizer(model, optimizer_name,
                                       (0.001 if head_only else 0.0001) if transfer else None)
            stage_counts[stage] = parameter_counts(model)["trainable_parameters"]
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        train_loss, train_acc = epoch_pass(model, data["train"], device, optimizer, head_only,
                                            2 if smoke else None)
        val_loss, val_acc = epoch_pass(model, data["val"], device, max_batches=2 if smoke else None)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        history.append({"epoch": epoch, "stage": stage, "optimizer": optimizer_name,
                        "lr": optimizer.param_groups[0]["lr"],
                        "momentum": OPTIMIZERS[optimizer_name]["momentum"],
                        "train_loss": train_loss, "val_loss": val_loss,
                        "train_accuracy": train_acc, "val_accuracy": val_acc,
                        "epoch_seconds": elapsed})
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc, best_loss, best_epoch = val_acc, val_loss, epoch
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ckpt)
        pd.DataFrame(history).to_csv(raw / "history.csv", index=False)
        print(f"{run_name} {epoch}/{total_epochs} {stage}: train={train_acc:.4f} val={val_acc:.4f} loss={val_loss:.4f} time={elapsed:.1f}s", flush=True)
    info = read_json(ROOT / "data/splits/dataset_info.json")
    metadata = {"model": model_name, "run": run_name, "seed": SEED, "epochs": total_epochs,
                "batch_size": batch_size, "optimizer": optimizer_name,
                "optimizer_settings": OPTIMIZERS[optimizer_name],
                "head_epochs": head_epochs, "finetune_epochs": finetune_epochs,
                "stage_trainable_parameters": stage_counts, "best_epoch": best_epoch,
                "best_val_accuracy": best_acc, "best_val_loss": best_loss,
                "avg_epoch_time_seconds": sum(r["epoch_seconds"] for r in history) / len(history),
                "checkpoint": str(ckpt.relative_to(ROOT)), "checkpoint_sha256": sha256(ckpt),
                "initial_state_sha256": initial_hash, "split_sha256": info["split_sha256"],
                "manifest_sha256": info["manifest_sha256"], "input_size": 64,
                "normalization": "ImageNet", "augmentation": "independent horizontal flip p=0.5",
                "device": str(device), "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
                "torch": torch.__version__, "torchvision": torchvision.__version__,
                "python": platform.python_version(), "precision": "FP32", "smoke": smoke,
                "pretrained_weights": {"mobilenet_v2": "IMAGENET1K_V2", "efficientnet_b0": "IMAGENET1K_V1"}.get(model_name),
                **parameter_counts(model)}
    write_json(raw / "run.json", metadata)
    log(f"Completed {run_name}: {total_epochs} epochs; best validation accuracy {best_acc:.6f} at epoch {best_epoch}; checkpoint `{metadata['checkpoint']}`.")
    return metadata
