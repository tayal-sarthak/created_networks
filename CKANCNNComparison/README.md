CKAN vs CNN Comparison

Overview
- Implements a clear, human-readable pipeline to compare CKANs (Kernel-Attention Convolutional Networks) against conventional CNNs (ResNet-like and ConvNeXt-like) across multiple datasets.
- Focuses on fixed-parameter comparisons, memory footprint estimates, calibration/uncertainty (via MC Dropout), and explainability (Grad-CAM and CKAN attention maps).

What is a CKAN (here)?
- We implement a practical CKAN block: a spatial depthwise convolution, followed by a mixture of K pointwise “expert” convolutions.
- The mixture weights are produced by an RBF-kernel attention between per-location features and K learned prototypes.
- This yields dynamic per-location channel mixing while keeping the module interpretable via prototype similarity maps.

Key Features
- Models: CKAN (configurable), SimpleResNet, SimpleConvNeXt.
- Datasets: CIFAR-10/100, MNIST, SVHN, STL10 (via torchvision).
- Fixed parameter budget: simple width search to align parameter counts across models within a tolerance.
- Metrics: accuracy, loss, ECE (Expected Calibration Error), NLL, parameter count, estimated memory.
- Explainability: Grad-CAM and CKAN prototype attention maps.
- Bayesian angle: MC Dropout inference to estimate uncertainty and calibration.

Quick Start
1) Create and activate a Python environment with PyTorch and torchvision.
2) One-click end-to-end run: open and run `run_all.py` (uses `/usr/local/bin/python3`).
   - It installs requirements (configurable), trains CKAN/ResNet/ConvNeXt under a shared param budget, evaluates (with MC Dropout), and saves explainability overlays plus a summary CSV.
3) Or install requirements manually: `/usr/local/bin/python3 -m pip install -r requirements.txt`
4) Optional smoke test (CPU-friendly): `bash scripts/smoke.sh`
5) Train CKAN on CIFAR-10:
   `/usr/local/bin/python3 -m ckancnn.cli.train --model ckan_small --dataset cifar10 --epochs 2 --batch-size 128`
6) Evaluate with MC Dropout and calibration metrics:
   `/usr/local/bin/python3 -m ckancnn.cli.evaluate --checkpoint outputs/ckan_small_cifar10/latest.pt --dataset cifar10 --mc-samples 20`
7) Generate explainability visualizations:
   `/usr/local/bin/python3 -m ckancnn.cli.explain --checkpoint outputs/ckan_small_cifar10/latest.pt --dataset cifar10 --num-samples 8`
7) Run a budgeted comparison (quick demo epochs):
   `/usr/local/bin/python3 -m ckancnn.cli.compare --dataset cifar10 --param-budget 1000000 --epochs 2`

Parameter Budget Matching
- Use `--param-budget` to target a parameter count.
- The CLI will tune a width multiplier for each model to reach the desired count (within a tolerance).

Data
- Datasets are loaded via torchvision. They will be downloaded automatically to `data/` on first use.
- If running offline, ensure datasets are already present in `data/`.

Outputs
- Checkpoints, logs, and CSV metrics are saved under `outputs/<run_name>/`.
- Explainability images are saved under `outputs/<run_name>/explain/`.

Notes
- This code is written to be explicit, legible, and easy to modify. No tricky one-liners; the control flow is straightforward and variables are named descriptively.
- The CKAN design here is practical and interpretable but not tied to a specific paper’s implementation; adjust K, prototype dimension, and blocks as needed.
