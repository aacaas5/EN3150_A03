"""Build the report only after completed experiments pass the evidence audit.

All experimental numbers are read from saved outputs. Regeneration does not
train models, evaluate test data again, or alter the measured results.
"""
import ast
import shutil
import pandas as pd
from config import ROOT
from src.utils import read_json, write_json, sha256
from src.plotting import curves, optimizer_plot, final_plots, confusion_figure
from scripts.verify_project import verify

REPORT = ROOT / "report"
NAMES = {"model_a": "Model A", "model_b": "Model B", "mobilenet_v2": "MobileNetV2", "efficientnet_b0": "EfficientNet-B0"}

def esc(value):
    return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

def table(name, headers, rows, caption):
    text = r"\begin{table}[H]\centering\small" + "\n"
    text += r"\caption{" + caption + "}\n"
    text += r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}\n\\toprule\n"
    text += " & ".join(headers) + r" \\ \midrule" + "\n"
    text += "\n".join(" & ".join(map(esc, row)) + r" \\" for row in rows)
    text += "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    (REPORT / f"tables/{name}.tex").write_text(text, encoding="utf-8")
    return r"\input{tables/" + name + "}\n"

def figure(name, caption, width="0.96"):
    return (r"\begin{figure}[H]\centering" + "\n" +
            rf"\includegraphics[width={width}\textwidth]{{figures/{name}.pdf}}" + "\n" +
            r"\caption{" + caption + "}\n\\end{figure}\n")

def architecture(name):
    df = pd.read_csv(ROOT / f"results/tables/{name}_architecture.csv", keep_default_na=False)
    rows = []
    for _, r in df.iterrows():
        shape = lambda s: r"$" + r"\times".join(map(str, ast.literal_eval(s))) + "$"
        label = r.layer.replace("features", "f").replace("classifier", "head")
        op = {"BatchNorm2d": "BN", "AdaptiveAvgPool2d": "GAP", "MaxPool2d": "MaxPool"}.get(r.type, r.type)
        ksp = "/".join(str(r[c]).replace("(", "").replace(")", "").replace(", ", "x") for c in ("kernel", "stride", "padding"))
        rows.append([esc(label), op, shape(r.input_shape), shape(r.output_shape), esc(ksp), str(r.parameters)])
    text = r"\begin{center}\scriptsize\setlength{\tabcolsep}{3pt}" + "\n"
    text += r"\begin{longtable}{lllllr}" + "\n"
    text += r"\toprule Layer & Operation & Input (NCHW) & Output & K/S/P & Params \\ \midrule\endhead" + "\n"
    text += "\n".join(" & ".join(row) + r" \\" for row in rows)
    text += "\n\\bottomrule\n\\end{longtable}\n\\end{center}\n"
    (REPORT / f"tables/{name}_layers.tex").write_text(text, encoding="utf-8")
    return r"\input{tables/" + name + "_layers}\n"

def run():
    verify()
    for folder in ("tables", "figures", "sections"):
        (REPORT / folder).mkdir(parents=True, exist_ok=True)
    final = pd.read_csv(ROOT / "results/tables/final_comparison.csv")
    df = final.set_index("model")
    opt = pd.read_csv(ROOT / "results/tables/optimizer_comparison.csv").set_index("optimizer")
    info = read_json(ROOT / "data/splits/dataset_info.json")
    meta = {n: read_json(ROOT / f"results/raw/{n}/run.json") for n in NAMES}
    histories = {n: pd.read_csv(ROOT / f"results/raw/{n}/history.csv") for n in NAMES}
    selected = read_json(ROOT / "results/raw/selected_optimizer.json")["optimizer"]
    a, b = df.loc["model_a"], df.loc["model_b"]
    # Regenerate required figures strictly from stored evidence.
    for n in NAMES:
        curves(n)
        matrix = pd.read_csv(ROOT / f"results/raw/{n}/confusion_matrix.csv", index_col=0).values
        confusion_figure(matrix, info["classes"], n)
    optimizer_plot()
    final_plots(final)
    for folder in ("figures", "confusion_matrices"):
        for p in (ROOT / f"results/{folder}").glob("*.pdf"):
            if "smoke" not in p.name:
                shutil.copy2(p, REPORT / "figures" / p.name)

    def result_table(name, models, caption):
        return table(name, ["Model", "Accuracy (\\%)", "Macro P (\\%)", "Macro R (\\%)"],
                     [[NAMES[n], *[f"{df.loc[n, c] * 100:.2f}" for c in ("test_accuracy", "macro_precision", "macro_recall")]] for n in models], caption)

    sections = {}
    sections["01_introduction"] = r"""\section{Introduction}
CNNs learn local spatial features using shared convolution kernels \cite{lecun1998}.
Edge image classification requires a balance between recognition quality,
stored weights, computation and measured execution time. This study compares
a standard CNN (Model A), a custom depthwise-separable CNN (Model B), and two
ImageNet-pretrained lightweight networks on one fixed EuroSAT split. The central
constraint is fewer than 100,000 trainable parameters for Model B.
Experiments use PyTorch \cite{pytorch2019}; no test samples influence optimizer
selection, training, or checkpoint selection.
"""
    sections["02_dataset"] = r"""\section{Dataset and Data Preparation}
\subsection{EuroSAT Dataset}
EuroSAT comprises Sentinel-2 satellite scenes for land-use and land-cover
classification \cite{helber2019}. We use the RGB version exposed by
\texttt{torchvision.datasets.EuroSAT}, preserving its original class labels.
\subsection{Dataset Classes}
""" + table("dataset", ["Class", "Total", "Train", "Validation", "Test"],
            [[n, *[info["class_counts"][n][c] for c in ("total", "train", "val", "test")]] for n in info["classes"]]
            + [["Total", info["total_images"], *info["split_sizes"].values()]], "Class counts and the shared stratified allocation.") + r"""
\subsection{64x64 Input}
All networks receive original $64\times64$ RGB images. The pretrained networks
support this size through adaptive pooling; their usual larger ImageNet input
recipe is not used. No images are upscaled above the assignment resolution cap.
\subsection{70/15/15 Split}
One stratified split is constructed with seed 3150: training receives 70\%,
and the held-out 30\% is divided equally into validation and test sets.
Actual sample indices and a path/label manifest are saved in \texttt{data/splits/}.
Hash checks bind every run to exactly the same split and sample order.
An exhaustive content audit found no pixel-identical images across sets.
This is an image-level split; geographic independence is not established.
\subsection{Preprocessing and Augmentation}
Images are decoded into a reusable unsigned-byte cache, converted to floating
point and divided by 255. Fixed ImageNet normalization uses means
$(0.485,0.456,0.406)$ and standard deviations $(0.229,0.224,0.225)$.
Independent horizontal flips with probability 0.5 apply only during training.
Validation and test processing is deterministic. Custom and pretrained models
use the same pipeline here; neither normalization nor augmentation changes
sample membership or estimates statistics from held-out data.
"""
    savings = pd.read_csv(ROOT / "results/tables/convolution_savings.csv")
    sections["03_architectures"] = r"""\section{Custom CNN Architectures}
\subsection{Model A}
Four standard $3\times3$ convolutions use channel widths 24, 48, 96 and 128.
Each is followed by batch normalization, ReLU and $2\times2$ max-pooling.
Padding preserves convolutional spatial size. Global average pooling and a
small linear head avoid a large flattened dense layer. Batch normalization
provides learned channel scaling and stabilizes intermediate feature scales.
""" + f"The executed parameter count is {int(a.total_parameters):,}.\n" + r"""
\subsection{Model B}
The stem, channel widths, pooling schedule and head match Model A. Each later
standard convolution is replaced by a depthwise $3\times3$ convolution followed
by a pointwise $1\times1$ convolution. Depthwise groups equal input channels.
Each component has batch normalization and ReLU. These extra operations mean
this is not a strict single-operator ablation. Full leaf-layer tables, including
input/output shapes, activations and parameter counts, appear in the appendix;
the CSV tables additionally expose channels and groups explicitly.
""" + f"Model B has {int(b.trainable_parameters):,} trainable parameters; the constructor and final audit assert that this is below 100,000.\n" + r"""
\subsection{Depthwise Separable Convolution}
Depthwise filtering processes channels independently; pointwise convolution
learns cross-channel combinations \cite{howard2017}. For output spatial size
$H_o\times W_o$, multiplying the weight expressions below by $H_oW_o$ gives
the corresponding convolution MAC counts for one image.
\subsection{Parameter Calculation}
With convolution biases disabled,
\begin{align}
P_{\mathrm{standard}} &= K_hK_wC_{\mathrm{in}}C_{\mathrm{out}},\\
P_{\mathrm{DSC}} &= K_hK_wC_{\mathrm{in}} + C_{\mathrm{in}}C_{\mathrm{out}}.
\end{align}
Their ratio is $1/C_{\mathrm{out}}+1/(K_hK_w)$. Batch normalization adds two
trainable scalars per output channel; running statistics are non-trainable
buffers. The classification head includes bias. Thus whole-network counts
include more than the convolution weights in the following table.
""" + table("savings", ["Channels", "Standard", "Depthwise", "Pointwise", "Reduction (\\%)"],
            [[f"{int(r.cin)} to {int(r.cout)}", int(r.standard_weights), int(r.depthwise_weights), int(r.pointwise_weights), f"{r.reduction_percent:.2f}"] for _, r in savings.iterrows()], "Analytical convolution weight counts, also generated programmatically.") + r"""
\subsection{Hardware-Aware Activation Selection}
ReLU computes $\max(0,x)$ using a simple comparison and avoids exponentials.
It is widely supported by inference kernels and is compatible with common
integer inference workflows. No quantization or activation-specific energy
measurement is performed here, so hardware savings are not claimed empirically.
"""
    sections["04_optimizer"] = r"""\section{Optimizer Selection and Tuning}
\subsection{SGD}
Standard SGD updates parameters with the current minibatch gradient. The
baseline uses learning rate 0.01 and no momentum.
\subsection{SGD with Momentum}
Momentum accumulates a velocity from earlier gradients; this can accelerate
consistent descent directions \cite{sutskever2013}. The experiment uses
learning rate 0.01 and momentum 0.9, without Nesterov acceleration.
\subsection{Adam}
Adam adapts updates using first and second gradient moments \cite{kingma2015}.
The tested learning rate is 0.001, with default betas $(0.9,0.999)$ and
$\epsilon=10^{-8}$.
\subsection{Experimental Comparison}
Model B is used for its low compute requirement. Each recipe uses ten epochs,
batch size 128, identical initial weights, seed, data order and augmentation
strategy. No weight decay, scheduler or early stopping is used. These are
controlled recipe comparisons, not exhaustive learning-rate searches.
""" + table("optimizers", ["Optimizer", "LR", "Momentum", "Best val. (\\%)", "Epoch", "Final val. (\\%)"],
           [[n, f"{r.lr:g}", f"{r.momentum:g}", f"{100*r.best_val_accuracy:.2f}", int(r.best_epoch), f"{100*r.final_val_accuracy:.2f}"] for n,r in opt.iterrows()], "Validation-based optimizer comparison; all runs use ten epochs.") + figure("optimizer_comparison", "Validation loss and accuracy; unsmoothed observations.") + r"""\subsection{Selected Optimizer}
""" + f"{selected.capitalize()} is selected by highest validation accuracy, with lower validation loss breaking ties. " + f"Momentum improves best validation accuracy over plain SGD by {100*(opt.loc['momentum','best_val_accuracy']-opt.loc['sgd','best_val_accuracy']):.2f} percentage points under these settings. " + r"""The curves show faster early progress with momentum and Adam, but neither
is monotonic. The fixed budget does not establish asymptotic convergence.
No optimizer setting was changed after viewing test data.
"""
    custom = r"""\section{Custom CNN Training and Evaluation}
\subsection{Training Setup}
Both custom models are initialized afresh and trained for 25 epochs using the
selected optimizer, batch size 128 and cross-entropy loss. Computation uses
FP32 on an NVIDIA GeForce RTX 5060 Laptop GPU. Deterministic algorithms and
seed 3150 are enabled. The best validation-accuracy checkpoint, with validation
loss breaking ties, is retained. There is no early stopping. The first ten
epochs of the final Model B run exactly reproduce its Adam pilot losses and
accuracies. Software versions are recorded in each run's JSON and the README.
\subsection{Training Curves}
"""
    for n in ("model_a", "model_b"):
        h, m = histories[n], meta[n]
        delta = h.val_accuracy.diff()
        worst = int(delta.idxmin())
        custom += figure(n + "_curves", NAMES[n] + " training and validation history.")
        custom += (f"{NAMES[n]} selects epoch {m['best_epoch']} with {100*m['best_val_accuracy']:.2f}\\% validation accuracy. "
                   f"Final training/validation accuracies are {100*h.train_accuracy.iloc[-1]:.2f}\\%/{100*h.val_accuracy.iloc[-1]:.2f}\\%. "
                   f"Its largest epoch-to-epoch validation drop is {-100*delta.iloc[worst]:.2f} percentage points at epoch {int(h.epoch.iloc[worst])}. "
                   "Validation fluctuations motivate checkpoint selection; their precise cause was not isolated.\n")
    custom += r"""\subsection{Test Results}
Accuracy is the fraction of correct predictions. Precision and recall are
unweighted macro averages over all ten classes; undefined class precision is
set to zero. Each selected checkpoint is evaluated on the untouched test set.
""" + result_table("custom_accuracy", ["model_a", "model_b"], "Custom-model held-out metrics.")
    custom += r"\subsection{Confusion Matrices}" + "\n"
    for n in ("model_a", "model_b"):
        custom += figure(n, NAMES[n] + ": numeric counts, true classes on rows and predictions on columns.", ".72")
    custom += r"\subsection{Model A vs Model B Resource Comparison}" + "\n" + table("custom_resources", ["Model", "Parameters", "Size (KiB)", "Epoch (s)", "MACs (M)"],
           [[NAMES[n], f"{int(df.loc[n,'total_parameters']):,}", f"{df.loc[n,'model_size_kb']:.2f}", f"{df.loc[n,'avg_epoch_time_seconds']:.3f}", f"{df.loc[n,'conv_linear_macs']/1e6:.3f}"] for n in ("model_a","model_b")], "Trainable and total parameters coincide for custom models.")
    custom += (f"Model B reduces whole-network parameters by {100*(1-b.total_parameters/a.total_parameters):.2f}\\%, "
               f"serialized size by {100*(1-b.model_size_bytes/a.model_size_bytes):.2f}\\%, and counted MACs by {100*(1-b.conv_linear_macs/a.conv_linear_macs):.2f}\\%. "
               f"Its test accuracy is {100*(a.test_accuracy-b.test_accuracy):.2f} percentage points lower. "
               "The measured training time does not decrease: depthwise operators, extra normalization/activation layers and launch or memory overhead can limit speedups. These explanations are plausible, not isolated by the measurements.\n")
    sections["05_custom"] = custom
    transfer = r"""\section{Lightweight State-of-the-Art Models}
\subsection{MobileNetV2}
MobileNetV2 uses inverted residual blocks and linear bottlenecks
\cite{sandler2018}. We load torchvision's explicit ImageNet
\texttt{IMAGENET1K\_V2} weights and replace its classifier with a ten-class head.
\subsection{EfficientNet-B0}
EfficientNet balances network depth, width and resolution \cite{tan2019}.
We use torchvision's B0 \texttt{IMAGENET1K\_V1} weights and replace its final
classifier. No fallback to random initialization is permitted.
\subsection{Transfer Learning Strategy}
Stage one trains the head for two epochs using Adam at 0.001; feature weights
and feature batch-normalization running statistics remain frozen. Stage two
unfreezes the entire network and uses a new Adam optimizer at 0.0001 for eight
epochs. Validation selects the checkpoint across both stages. Both models use
exactly the saved custom-model split and the same 64x64 preprocessing.
"""
    stage_rows = []
    for n in ("mobilenet_v2", "efficientnet_b0"):
        for stage in ("head", "finetune"):
            h = histories[n].query("stage == @stage")
            stage_rows.append([NAMES[n], stage, len(h), f"{h.lr.iloc[0]:g}", f"{meta[n]['stage_trainable_parameters'][stage]:,}", f"{h.epoch_seconds.mean():.3f}"])
    transfer += table("transfer_stages", ["Model", "Stage", "Epochs", "LR", "Trainable", "Epoch (s)"], stage_rows, "Measured stages; final fine-tuning trains all parameters.")
    transfer += r"\subsection{Results}" + "\n" + result_table("pretrained_accuracy", ["mobilenet_v2", "efficientnet_b0"], "Pretrained-model held-out metrics.")
    transfer += table("pretrained_resources", ["Model", "Total/trainable", "Size (MiB)", "Mean epoch (s)"],
                      [[NAMES[n], f"{int(df.loc[n,'total_parameters']):,}", f"{df.loc[n,'model_size_mb']:.3f}", f"{df.loc[n,'avg_epoch_time_seconds']:.3f}"] for n in ("mobilenet_v2", "efficientnet_b0")], "Total parameters equal fine-tuning trainable parameters. Timing averages both stages.")
    for n in ("mobilenet_v2", "efficientnet_b0"):
        transfer += f"{NAMES[n]} selects epoch {meta[n]['best_epoch']} with {100*meta[n]['best_val_accuracy']:.2f}\\% validation accuracy.\n"
        transfer += figure(n + "_curves", NAMES[n] + " two-stage training history.")
        transfer += figure(n, NAMES[n] + " test confusion matrix (counts).", ".72")
    sections["06_transfer"] = transfer
    comparison = r"""\section{Final Comparison}
\subsection{Accuracy}
""" + figure("final_accuracy", "Held-out test accuracy using validation-selected checkpoints.")
    for n in ("mobilenet_v2", "efficientnet_b0"):
        r = df.loc[n]
        comparison += (f"Relative to {NAMES[n]}, Model B saves {100*(1-b.total_parameters/r.total_parameters):.2f}\\% of parameters "
                       f"and {100*(1-b.model_size_bytes/r.model_size_bytes):.2f}\\% of serialized bytes, "
                       f"with {100*(r.test_accuracy-b.test_accuracy):.2f} percentage points lower test accuracy.\n")
    comparison += r"""\subsection{Parameter Count}
The parameter totals count learned tensors only, while trainable totals reflect
the active training stage. Frozen pretraining parameters still occupy storage.
\subsection{Memory Footprint}
""" + figure("final_resources", "Parameter counts and measured FP32 state-dictionary sizes.") + r"""
Sizes include batch-normalization buffers and serialization overhead but exclude
optimizer state. KiB and MiB use powers of 1024. The CSV column suffixes
\texttt{kb/mb} use these binary units. Disk size is not peak inference RAM;
activation and workspace memory are not measured.
\subsection{Computational Cost}
""" + table("compute", ["Model", "MACs (M)", "2xMAC (MFLOP)", "CPU mean (ms)", "CPU p95 (ms)"],
            [[NAMES[n], f"{r.conv_linear_macs/1e6:.3f}", f"{r.conv_linear_flops_2mac/1e6:.3f}", f"{r.cpu_latency_ms_mean:.3f}", f"{r.cpu_latency_ms_p95:.3f}"] for n,r in df.iterrows()], "One 64x64 image: analytical Conv/Linear cost and measured host latency.") + r"""
MAC accounting uses executed layer output shapes, kernel sizes and group counts.
It excludes normalization, activations, pooling, residual additions and data
movement; twice MACs is only an arithmetic FLOP estimate. CPU timing uses the
Intel Core i7-13620H host, four PyTorch threads, FP32, batch one, 20 warm-ups
and 100 timed forwards. It excludes preprocessing and file I/O.
\subsection{Edge Deployment Trade-offs}
Model B offers the smallest storage and counted arithmetic cost, but it is not
faster than Model A in these host measurements. Depthwise speed depends on
operator support, memory movement and workload size. The pretrained models
improve accuracy but require far more weights and more training time per epoch.
The shorter transfer-learning schedule does not include the original ImageNet
pretraining cost. Device-specific quantization, peak RAM, latency and power
measurements would be needed before choosing a deployment target.
"""
    sections["07_comparison"] = comparison
    discussion = r"""\section{Discussion}
The custom comparison holds channel widths, pooling schedule, training budget
and optimizer fixed, while Model B introduces depthwise/pointwise factorization
and additional normalization and ReLU operations. Thus the observed difference
cannot be attributed solely to the convolution operator. Both custom models
show validation fluctuations, making the saved best checkpoint preferable to
blindly using the final epoch.
"""
    for n in NAMES:
        cm = pd.read_csv(ROOT / f"results/raw/{n}/confusion_matrix.csv", index_col=0)
        values = cm.to_numpy().copy()
        __import__('numpy').fill_diagonal(values, 0)
        i,j = __import__('numpy').unravel_index(values.argmax(), values.shape)
        discussion += f"For {NAMES[n]}, the largest directed confusion is {esc(cm.index[i])} predicted as {esc(cm.columns[j])}, with {int(values[i,j])} images.\n"
    discussion += r"""
These are single-seed observations with no confidence intervals or repeated-run
variance estimate. Small accuracy differences should not be treated as proven
general advantages. The split is stratified but not geographic; satellite
scene correlation can make generalization to new regions harder than this
test suggests. ImageNet transfer supplies additional prior information, and
the 64x64 input differs from the pretrained recipe. The optimizer pilot compares
three fixed configurations for a short budget; it does not prove universal
optimizer superiority. No unseen edge device or energy benefit is asserted.
"""
    sections["08_discussion"] = discussion
    best = df.test_accuracy.idxmax()
    sections["09_conclusion"] = r"\section{Conclusion}" + "\n" + (
        f"The required resource-constrained model is achieved with {int(b.trainable_parameters):,} trainable parameters "
        f"and {100*b.test_accuracy:.2f}\\% test accuracy. Compared with Model A, it retains similar observed accuracy "
        "while substantially reducing weights and counted MACs, although host timing does not improve. "
        f"{NAMES[best]} achieves the highest observed test accuracy, {100*df.loc[best,'test_accuracy']:.2f}\\%, at a larger storage cost. "
        "The appropriate choice therefore depends on the available memory, acceptable accuracy loss and measured target-device performance.\n")
    appendix = r"""\appendix
\section{Layer-wise Architecture Tables}
The alias \texttt{f} means \texttt{features}; \texttt{head} means
\texttt{classifier}. Shapes include batch size one. Input/output shapes expose
all channel counts. K/S/P denotes kernel/stride/padding; symmetric pairs use
\texttt{x}. A dash denotes not applicable. All listed parameters are trainable.
ReLU entries explicitly show activation placement. BN denotes batch normalization
and GAP denotes adaptive global average pooling. Model B depthwise convolutions
use groups equal to input channels; its pointwise and all standard convolutions
use one group. Full names and groups are in the machine-readable architecture CSVs.
\subsection{Model A}
""" + architecture("model_a") + r"\subsection{Model B}" + "\n" + architecture("model_b")
    sections["10_appendix"] = appendix
    for name, content in sections.items():
        (REPORT / f"sections/{name}.tex").write_text(content, encoding="utf-8")
    abstract = (f"Four CNNs are evaluated on a deterministic stratified split of {info['total_images']:,} EuroSAT RGB images. "
                f"Model B uses {int(b.trainable_parameters):,} trainable parameters and achieves {100*b.test_accuracy:.2f}\\% test accuracy, "
                f"compared with {100*a.test_accuracy:.2f}\\% for the standard CNN. "
                f"MobileNetV2 and EfficientNet-B0 achieve {100*df.loc['mobilenet_v2','test_accuracy']:.2f}\\% and {100*df.loc['efficientnet_b0','test_accuracy']:.2f}\\%, respectively. "
                "The compact model substantially reduces stored weights and counted arithmetic but does not improve measured host latency over the standard CNN. "
                "All numerical results derive from executed experiments and saved predictions.")
    main = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=24mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{graphicx,booktabs,amsmath,float,longtable,array}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{2em}
\begin{document}
\begin{titlepage}
\centering
{\large University of Moratuwa\par}
\vspace{2cm}
{\Large EN3150 Assignment 03\par}
\vspace{0.8cm}
{\huge\bfseries Resource-Constrained CNN\\[0.3cm]for Edge Image Classification\par}
\vspace{1.8cm}
{\large Module: EN3150\par}
{\large Group number: [Fill in]\par}
\vspace{1cm}
\begin{tabular}{ll}\toprule
Group member & Index number\\\midrule
{[Member 1 name]} & {[Index 1]}\\
{[Member 2 name]} & {[Index 2]}\\
{[Member 3 name]} & {[Index 3]}\\
{[Member 4 name]} & {[Index 4]}\\\bottomrule
\end{tabular}
\vfill
{\large Experimental study and reproducible implementation\par}
\end{titlepage}
\begin{abstract}
""" + abstract + "\n\\end{abstract}\n\\tableofcontents\n\\clearpage\n"
    main += "\n".join(r"\input{sections/" + n + "}" for n in sections if n != "10_appendix")
    main += "\n\\clearpage\n\\bibliographystyle{plain}\n\\bibliography{references}\n\\clearpage\n\\input{sections/10_appendix}\n\\end{document}\n"
    # Preserve manually entered title information when regenerating existing report.
    main_path = REPORT / "main.tex"
    if not main_path.exists():
        main_path.write_text(main, encoding="utf-8")
    else:
        old = main_path.read_text(encoding="utf-8")
        start, end = old.index(r"\begin{abstract}"), old.index(r"\end{abstract}")
        old = old[:start] + "\\begin{abstract}\n" + abstract + "\n" + old[end:]
        main_path.write_text(old, encoding="utf-8")
    tracked = list((ROOT / "results/tables").glob("*.csv"))
    tracked += [ROOT / "data/splits/dataset_info.json", ROOT / "results/raw/selected_optimizer.json"]
    for n in NAMES:
        tracked += list((ROOT / f"results/raw/{n}").glob("*.json"))
        tracked += list((ROOT / f"results/raw/{n}").glob("*.csv"))
    tracked += list((REPORT / "tables").glob("*.tex")) + list((REPORT / "sections").glob("*.tex"))
    tracked += list((REPORT / "figures").glob("*.pdf"))
    write_json(REPORT / "result_provenance.json", {p.relative_to(ROOT).as_posix(): sha256(p) for p in tracked})
    print("Generated report/main.tex and result-derived tables, sections and figures.")

if __name__ == "__main__":
    run()
