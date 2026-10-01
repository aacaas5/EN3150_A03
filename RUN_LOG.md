# Run log

## 2026-10-01 — initial inspection

- Empty project directory; no Git initialized.
- Python 3.14.5; torch 2.13.0+cu130; torchvision 0.28.0+cu130.
- CUDA available: NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM.
- MiKTeX pdflatex is available.
- Plan: seed 3150; 64x64 for all models; ImageNet normalization for transfer learning (also used for custom models for consistency); horizontal flips only in training. Pretrained CNNs support 64x64 despite their ImageNet recipe using larger inputs. No resampling above the assignment resolution cap.
- `python -m pip install pandas matplotlib scikit-learn pillow`: pandas was missing; installed pandas 3.0.6 and tzdata. Other dependencies were already available.
- `python -m scripts.prepare_data > prepare_data.log 2>&1`: downloaded original torchvision EuroSAT archive and decoded images for a reusable uint8 cache.
- `python -m scripts.inspect_models > inspect_models.log 2>&1`: all four [2,3,64,64] forward passes produced [2,10]; downloaded explicit official torchvision ImageNet weights. Model A: 164,962 parameters, 30,967,040 Conv/Linear MACs. Model B: 22,426 parameters, 6,188,288 MACs. Windows PowerShell marked the redirected weight-download stderr progress as a native-command error, despite model inspection completing and saving all outputs.
- Host CPU: Intel Core i7-13620H, 10 physical / 16 logical cores. PyTorch constrained to four threads for controlled host profiling.
- `python -m pip freeze > environment-lock.txt`: captured installed environment. Minimal project dependencies are pinned separately in requirements.txt.
- `python -m scripts.run_experiments` queued after dataset completion, output in experiments.log; GPU jobs run sequentially to avoid timing interference.
- Primary references checked: EuroSAT https://arxiv.org/abs/1709.00029 ; MobileNets https://arxiv.org/abs/1704.04861 ; MobileNetV2 https://arxiv.org/abs/1801.04381 ; EfficientNet https://proceedings.mlr.press/v97/tan19a.html ; Adam https://arxiv.org/abs/1412.6980 ; momentum https://proceedings.mlr.press/v28/sutskever13.html ; PyTorch https://papers.nips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html ; CNN background https://leon.bottou.org/papers/lecun-98h .
- Data audit passed: 27,000 distinct decoded image contents; zero identical images crossing split boundaries; exact per-class 70/15/15 allocation. First-time Windows JPEG loading was substantially slower than cached training; the one-time tensor cache avoids repeating it.
- Smoke: correct [128,3,64,64] batch, deterministic validation transform, perfect toy-label metric sanity check, two training/two validation batches on Model B, saved checkpoint/history and generated curves. Synthetic smoke outputs are explicitly excluded from final comparisons.
- Optimizer comparison: 10 epochs each; final custom runs: 25 epochs each; pretrained: 2 head-only + 8 all-layer fine-tuning epochs. Selection uses validation accuracy only.

- EuroSAT prepared: 27,000 RGB images; stratified 18,900/4,050/4,050 split; saved index and manifest hashes.

- Completed smoke: 1 epochs; best validation accuracy 1.000000 at epoch 1; checkpoint `checkpoints\smoke.pt`.

- Completed optimizer_sgd: 10 epochs; best validation accuracy 0.816296 at epoch 9; checkpoint `checkpoints\optimizer_sgd.pt`.

- Completed optimizer_momentum: 10 epochs; best validation accuracy 0.905679 at epoch 8; checkpoint `checkpoints\optimizer_momentum.pt`.

- Completed optimizer_adam: 10 epochs; best validation accuracy 0.914568 at epoch 10; checkpoint `checkpoints\optimizer_adam.pt`.

- Selected optimizer: adam; selection excludes test data.

- Completed model_a: 25 epochs; best validation accuracy 0.940000 at epoch 18; checkpoint `checkpoints\model_a.pt`.

- Completed model_b: 25 epochs; best validation accuracy 0.930864 at epoch 23; checkpoint `checkpoints\model_b.pt`.

- Completed mobilenet_v2: 10 epochs; best validation accuracy 0.975062 at epoch 10; checkpoint `checkpoints\mobilenet_v2.pt`.

- Completed efficientnet_b0: 10 epochs; best validation accuracy 0.968642 at epoch 8; checkpoint `checkpoints\efficientnet_b0.pt`.

- Full experiment runner completed successfully (exit 0). Both custom models ran 25 epochs; all three optimizer pilots ran 10 epochs; each pretrained model ran 2 head-only and 8 fine-tuning epochs.
- `python -m scripts.verify_project`: PASS, 99 checks. Independently recomputed accuracy/macro precision/macro recall from all 4,050 saved test predictions per model. Model B's final-run first ten epochs exactly match its selected Adam pilot (maximum loss/accuracy difference 0.0).
- Test accuracies: Model A 0.942962962963; Model B 0.941481481481; MobileNetV2 0.973086419753; EfficientNet-B0 0.968148148148. Machine-readable measurements are in results/tables/.
- `python -m scripts.generate_report_assets`: generated result-derived report sections, tables, figures, main.tex and SHA-256 provenance only after the completed experiments passed audit.
- Report compilation initially found missing `float.sty`; fixed using `miktex packages install float`. A title-page placeholder beginning with `[` was interpreted as a LaTeX optional argument; braces around placeholder cells fixed the error in both source and generator.
- Compiled successfully with pdflatex, bibtex, pdflatex, pdflatex (exit 0). Final report: 18 A4 pages, `report/main.pdf`. No undefined citations/references, LaTeX errors, or overfull boxes in final main.log. MiKTeX's administrative update reminder did not affect compilation.
- `pdftoppm -scale-to 1000 -png report/main.pdf tmp/pdf-review/page`: rendered all 18 pages. Visually inspected all pages via contact sheets and detailed architecture/confusion-matrix pages; no clipping, overlapping content or missing figures. Review artifacts are under tmp/pdf-review/.
- `python -m scripts.verify_project --require-report`: PASS, 172 checks, including generated report assets/source-result hashes. `python -m compileall -q src scripts config.py` also passed. Final audit evidence: `results/verification.json`.
- All requested experiments completed; no fabricated values, no Git initialization, no GitHub activity. Remaining user information is limited to title-page group/member/index placeholders.
