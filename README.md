# EN3150 Assignment 03

Resource-constrained CNN classification of EuroSAT RGB. This project compares a
standard CNN, a custom depthwise-separable CNN below 100,000 parameters,
torchvision MobileNetV2, and torchvision EfficientNet-B0. No Git repository is
created by any script.

## Environment (Windows PowerShell)

The recorded execution uses Python 3.14.5, PyTorch 2.13.0+cu130,
torchvision 0.28.0+cu130, and an NVIDIA GeForce RTX 5060 Laptop GPU.
Use a virtual environment for a clean reproduction:

```powershell
cd C:\Users\aacaa\Documents\project\EN3150_A03
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

For CPU execution, replace `cu130` with `cpu`; scripts automatically fall back to
CPU when CUDA is unavailable. Training will take longer. Internet access is
needed for the original dataset and official pretrained weight downloads.

## Reproduce the experiments

Run from the project root, using module syntax. Default settings are in
`config.py`. The complete sequence is:

```powershell
python -m scripts.prepare_data
python -m scripts.run_experiments
python -m scripts.verify_project
python -m scripts.generate_report_assets
```

Or execute the stages separately:

```powershell
python -m scripts.inspect_models
python -m scripts.smoke_test
python -m scripts.optimizer_experiment --epochs 10 --batch-size 128
python -m scripts.train_custom --epochs 25 --batch-size 128
python -m scripts.train_pretrained --head-epochs 2 --finetune-epochs 8 --batch-size 128
python -m scripts.evaluate_all
python -m scripts.verify_project
python -m scripts.generate_report_assets
```

Completed training runs are preserved and skipped after their configuration is
checked. Interrupted runs have no `run.json` and restart from the fixed seed;
there is no mid-epoch resume. To retrain every run, first archive/move the existing
`checkpoints/` and `results/` directories. Keep `data/splits/` unchanged. Never
mix histories from different settings. `evaluate_all` regenerates test metrics,
figures, tables and host latency measurements from saved best checkpoints;
regenerate the report afterward because latency values may change.

## Data and leakage controls

`torchvision.datasets.EuroSAT` supplies 27,000 original 64x64 RGB images and ten
original class names. Seed 3150 produces a stratified 18,900/4,050/4,050 split.
Actual indices, a sample-path/label manifest, class counts, and SHA-256 hashes
are stored under `data/splits/`. Membership is checked on every loader creation.
All models use these exact samples. Initial preparation also checks for
pixel-identical images across the three sets. The random image-level split does
not establish geographic independence; nearby scenes can remain correlated.

Images are cached as uint8 tensors for efficient Windows loading. Every model
uses RGB values divided by 255 and ImageNet mean/std normalization. Training
alone applies independent horizontal flips with probability 0.5. Validation and
test processing is deterministic. No fitted normalization or test statistics are
used. Pretrained networks accept 64x64 through adaptive pooling, so images are
not upscaled to their usual 224x224 ImageNet recipe. This preserves the assignment
cap and must be considered when interpreting transfer performance.

## Models and experiments

- Model A: four Conv-BatchNorm-ReLU-MaxPool blocks, channels 24/48/96/128,
  global average pooling and a linear ten-class head.
- Model B: same stem, widths and pooling; later standard convolutions become
  depthwise 3x3 plus pointwise 1x1 convolutions. Extra batch normalization/ReLU
  follow each component. The constructor asserts fewer than 100,000 trainable
  parameters. See generated architecture CSVs for exact counts.
- Optimizer experiment: identical Model B initialization, split, batch size,
  seed and ten-epoch budget. SGD lr=0.01; momentum SGD lr=0.01, momentum=0.9;
  Adam lr=0.001 (default betas 0.9/0.999, eps=1e-8). No weight decay, scheduler,
  or early stopping. Validation accuracy selects the optimizer, with validation
  loss breaking ties. This compares stated recipes, not exhaustive tuning.
- Custom final training: fresh initialization, selected optimizer, 25 epochs.
- Transfer: official MobileNetV2 IMAGENET1K_V2 and EfficientNet-B0 IMAGENET1K_V1
  weights; replace the head. Train head for two epochs with Adam lr=0.001,
  keeping frozen features and their BatchNorm statistics in evaluation mode.
  Then unfreeze all layers for eight epochs with a new Adam optimizer lr=0.0001.
  Best validation checkpoint across both stages is retained.

Training uses FP32, CrossEntropyLoss, fixed RNG seeds, deterministic PyTorch
algorithms, four CPU threads, and no early stopping. Reproducibility is intended
within the recorded software/hardware environment; cross-version/device bitwise
identity is not promised. Timing includes training, validation and preprocessing,
with CUDA synchronization, and excludes checkpoint/CSV writes.

## Results and interpretation

- `results/raw/<run>/`: history, run metadata, predictions, class report, numeric
  confusion matrix and summary metrics.
- `results/tables/custom_models.csv`, `pretrained_models.csv`,
  `final_comparison.csv`: measured final comparisons.
- `results/tables/optimizer_comparison.csv`: validation-based selection.
- `results/tables/*_architecture.csv`: leaf names, shapes, kernels, channels,
  stride, padding, groups, activations, layer parameters and MACs.
- `results/figures/` and `results/confusion_matrices/`: PDF and PNG figures.
- `results/verification.json`: final machine-readable audit.

Precision and recall use **macro averaging across all ten classes**, with
`zero_division=0`. Test data is first evaluated after optimizer selection and
all training. FP32 `state_dict` sizes include buffers and serialization overhead,
but exclude optimizer state. Columns named `model_size_kb/mb` use binary units
(KiB/MiB), stated explicitly in the report. Disk size is not peak runtime RAM.
MACs count Conv2d/Linear only; approximate arithmetic FLOPs are twice that count.
Normalization, nonlinearities, pooling, residual additions and memory traffic
are excluded. Batch-one CPU latency uses 20 warm-ups and 100 timed forwards,
four threads, and pre-normalized inputs. It is a host benchmark, not a direct
edge-device power/latency measurement. No edge hardware energy claims are made.

## LaTeX report and VS Code

The report generator refuses to run without completed, audited experiments.
Its numerical content is derived from saved CSV/JSON outputs, with source hashes
in `report/result_provenance.json`. After generating assets:

```powershell
Push-Location report
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
Pop-Location
python -m scripts.verify_project --require-report
```

Use MiKTeX or TeX Live with standard packages. If MiKTeX lacks `float.sty`, run
`miktex packages install float`. In VS Code, open `report/main.tex`
with LaTeX Workshop and use its pdflatex/bibtex recipe, or `latexmk -pdf main.tex`.
The final title page contains the group name, four member names and index numbers. `TODO.md` and `RUN_LOG.md` record project status and
execution details. Never submit smoke-test figures as experiment results.

## Team

| Member | GitHub | Primary Contributions |
| --- | --- | --- |
| Rajaretnam Aruniya (ARUNIYA R.) | [Aruniya23](https://github.com/Aruniya23) | EuroSAT preparation, class inspection, stratified splits, preprocessing and data/reproducibility checks. |
| MUHAMATH M.A.A.A. | [aacaas5](https://github.com/aacaas5) | Model A, Model B, depthwise separable design, parameter and architecture analysis, integration and coordination. |
| Piriyatharsan M. | [Piriyatharsan23](https://github.com/Piriyatharsan23) | Shared training, SGD/momentum/Adam comparison, evaluation metrics and confusion matrices. |
| Snekan S. | [Snekan19](https://github.com/Snekan19) | MobileNetV2, EfficientNet-B0, transfer learning, MAC/resource profiling, final comparison and report/notebook integration. |

The contribution areas above are supplied and confirmed by the team.

## Development Note

The team reports collaborative offline development on a University department
computer from 26 September through 1 October 2026, followed by transfer to an
Internet-connected computer for repository synchronization. Git was not used
for the earlier offline work. Date prefixes in commit subjects identify the
team-reported offline work dates; they are not historical Git timestamps.
Git author and committer timestamps record the actual import and are not
backdated. Existing execution logs and file metadata are retained as recorded;
they do not independently verify each team-reported daily milestone.

## Final submission artifacts

- `FeatureFour_A03_EN3150.ipynb`: self-contained notebook with embedded recorded
  evidence; full training and downloads are disabled by default.
- `report/FeatureFour_A03_EN3150.pdf`: current final report, including group/member
  details and the title-page repository link.
- `report/main.tex`: final source. Historical log/TODO notes about placeholders
  and earlier page counts describe earlier versions, not this final PDF.

Do not rerun the report generator merely to compile the final edited report:
it can replace editorial/title-page edits. Compile the existing source directly.
