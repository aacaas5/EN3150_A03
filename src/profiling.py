"""Leaf-layer shapes and exact Conv2d/Linear MAC accounting.

MAC counts exclude normalization, activations, pooling, elementwise operations,
and data movement; 2*MAC is an arithmetic FLOP estimate, not full operator FLOPs.
"""
import time
import torch
from torch import nn
import pandas as pd
from config import ROOT

def parameter_counts(model):
    return {"total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}

def architecture_table(model, name):
    rows, hooks = [], []
    device = next(model.parameters()).device
    def hook(label):
        def record(layer, args, out):
            macs = 0
            if isinstance(layer, nn.Conv2d):
                macs = out.numel() * (layer.in_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1]
            elif isinstance(layer, nn.Linear):
                macs = out.numel() * layer.in_features
            rows.append({"layer": label, "type": type(layer).__name__,
                "input_shape": str(list(args[0].shape)), "output_shape": str(list(out.shape)),
                "kernel": str(getattr(layer, "kernel_size", "-")),
                "in_channels": getattr(layer, "in_channels", getattr(layer, "in_features", "-")),
                "out_channels": getattr(layer, "out_channels", getattr(layer, "out_features", "-")),
                "stride": str(getattr(layer, "stride", "-")), "padding": str(getattr(layer, "padding", "-")),
                "groups": getattr(layer, "groups", "-"),
                "activation": type(layer).__name__ if isinstance(layer, (nn.ReLU, nn.ReLU6, nn.SiLU)) else "-",
                "parameters": sum(p.numel() for p in layer.parameters(recurse=False)),
                "trainable_parameters": sum(p.numel() for p in layer.parameters(recurse=False) if p.requires_grad),
                "conv_linear_macs": int(macs)})
        return record
    for label, layer in model.named_modules():
        if not list(layer.children()):
            hooks.append(layer.register_forward_hook(hook(label)))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 64, device=device))
    for h in hooks:
        h.remove()
    model.train(was_training)
    assert output.shape == (1, 10)
    df = pd.DataFrame(rows)
    assert int(df.parameters.sum()) == parameter_counts(model)["total_parameters"]
    path = ROOT / f"results/tables/{name}_architecture.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return {**parameter_counts(model), "conv_linear_macs": int(df.conv_linear_macs.sum()),
            "conv_linear_flops_2mac": int(df.conv_linear_macs.sum()) * 2}

def inference_latency(model, repeats=100):
    """Batch-one FP32, warm model, host CPU only, four PyTorch threads.

    Input is already normalized. Excludes preprocessing and file I/O. This is
    a host benchmark, not a measurement on a microcontroller/edge accelerator.
    """
    original = next(model.parameters()).device
    model.cpu().eval()
    x = torch.zeros(1, 3, 64, 64)
    with torch.inference_mode():
        for _ in range(20):
            model(x)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            samples.append((time.perf_counter() - start) * 1000)
    model.to(original)
    return {"cpu_latency_ms_mean": sum(samples) / len(samples),
            "cpu_latency_ms_median": float(pd.Series(samples).median()),
            "cpu_latency_ms_p95": float(pd.Series(samples).quantile(.95)),
            "latency_repeats": repeats, "cpu_threads": torch.get_num_threads()}

def convolution_savings():
    rows = []
    for cin, cout in [(24, 48), (48, 96), (96, 128)]:
        standard, dsc = 9 * cin * cout, 9 * cin + cin * cout
        rows.append({"cin": cin, "cout": cout, "standard_weights": standard,
                     "depthwise_weights": 9 * cin, "pointwise_weights": cin * cout,
                     "dsc_weights": dsc, "reduction_percent": 100 * (1 - dsc / standard)})
    pd.DataFrame(rows).to_csv(ROOT / "results/tables/convolution_savings.csv", index=False)
