# EN3150 Assignment 03 — Feature Four

## Group Members

| Name | Index Number | Main Contribution |
| --- | --- | --- |
| Rajaretnam Aruniya | 230059T | EuroSAT preparation, stratified data split, preprocessing and reproducibility checks |
| Muhamath M.A.A.A. | 230416L | Model A, Model B, depthwise-separable design and architecture analysis |
| Piriyatharsan M. | 230491J | Training functions, optimizer comparison, evaluation metrics and confusion matrices |
| Snekan S. | 230620G | MobileNetV2, EfficientNet-B0, transfer learning, resource profiling and final comparison |

## Project Overview

This project studies image classification on the EuroSAT RGB dataset. We compare two custom CNNs with two lightweight pretrained models:

- Model A uses standard convolution blocks.
- Model B uses depthwise-separable convolutions and stays below the assignment limit of 100,000 parameters.
- MobileNetV2 and EfficientNet-B0 use pretrained ImageNet weights and are fine-tuned for EuroSAT.

The aim is to compare classification performance with parameter count, model size, computation and CPU inference latency.

## What We Did

The EuroSAT dataset contains 27,000 RGB images from ten classes. We used the original 64 × 64 image size and created one stratified split with seed 3150:

- Training: 18,900 images (70%)
- Validation: 4,050 images (15%)
- Test: 4,050 images (15%)

All four models use the same saved split. Training images use random horizontal flipping, while validation and test preprocessing remain deterministic.

We implemented and evaluated:

- Model A with four standard Conv–BatchNorm–ReLU–MaxPool blocks.
- Model B with depthwise and pointwise convolutions in the later blocks.
- An optimizer study comparing SGD, SGD with momentum and Adam.
- MobileNetV2 and EfficientNet-B0 with two-stage transfer learning.
- Accuracy, macro precision, macro recall, macro F1 and confusion-matrix evaluation.
- Parameter, model-size, MAC and CPU latency measurements.

Adam gave the best validation result in the controlled optimizer study, so it was used for the final custom-model training.

## Final Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Parameters | Size (MiB) | MACs (M) | CPU Mean (ms) | CPU p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Model A | 94.30% | 94.25% | 94.26% | 94.17% | 164,962 | 0.639 | 30.967 | 1.358 | 1.802 |
| Model B | 94.15% | 94.00% | 93.88% | 93.89% | 22,426 | 0.102 | 6.188 | 1.723 | 2.459 |
| MobileNetV2 | 97.31% | 97.20% | 97.21% | 97.20% | 2,236,682 | 8.754 | 24.461 | 8.452 | 13.143 |
| EfficientNet-B0 | 96.81% | 96.72% | 96.71% | 96.71% | 4,020,358 | 15.604 | 31.979 | 12.454 | 27.092 |

Model B satisfies the parameter requirement with 22,426 parameters. Compared with Model A, it reduces parameters by 86.41% and counted MACs by 80.02%, while its test accuracy is only 0.15 percentage points lower. However, Model B was slightly slower than Model A in the recorded CPU test, showing that fewer MACs do not always produce lower latency.

MobileNetV2 achieved the highest accuracy at 97.31%. EfficientNet-B0 achieved 96.81%, but it used more parameters, storage and CPU time than MobileNetV2 in this experiment.

## Results and Figures

The main results are stored in the `results/` directory:

- `results/tables/final_comparison.csv` contains the complete four-model comparison.
- `results/tables/optimizer_comparison.csv` contains the optimizer-study results.
- `results/raw/` contains the saved histories, predictions, metrics and run information.
- `results/figures/` contains the learning curves and resource-comparison figures.
- `results/confusion_matrices/` contains the test confusion matrices for all four models.

The figures show:

- Training and validation loss and accuracy across epochs.
- The prediction errors made by each model on the held-out test set.
- Accuracy compared with parameter count and MACs.
- The difference between theoretical computation and measured CPU latency.

## Notebook

`FeatureFour_A03_EN3150.ipynb` contains the model definitions, optimizer and training functions, evaluation code, resource profiling, learning curves, confusion matrices and final comparison.

Running all cells with the default settings uses Quick Review Mode. It verifies the embedded experiment evidence without downloading data or starting training. Full reproduction is available through the explicit opt-in flags and requires EuroSAT plus the pretrained ImageNet weights.
