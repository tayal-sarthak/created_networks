ckan vs cnn comparison

overview
- clear readable pipeline for ckan vs cnn across datasets
- focus on fixed parameter budget, memory estimate, calibration via mc dropout, explain via grad-cam, ckan attention

what is ckan (here)
- practical ckan block: spatial depthwise conv then mixture of k pointwise expert conv
- mixture weights via rbf kernel attention between per location features and k learned prototypes
- dynamic per location channel mixing with prototype similarity maps for interpretability

key features
- models: ckan (configurable), simpleresnet, simpleconvnext
- datasets: cifar-10/100, mnist, svhn, stl10 via torchvision
- fixed parameter budget: width search aligns param counts across models within tolerance
- metrics: accuracy, loss, ece, nll, param count, memory estimate
- explainability: grad-cam, ckan prototype attention maps
- bayesian angle: mc dropout inference for uncertainty, calibration

# quick start
1) create python env with pytorch, torchvision
2) one click run: open, run `run_all.py` (uses `/usr/local/bin/python3`)
   - installs requirements (configurable), trains ckan/resnet/convnext under shared param budget, evaluates (mc dropout), saves explain overlays, summary csv
3) manual deps: `/usr/local/bin/python3 -m pip install -r requirements.txt`
4) optional smoke test (cpu friendly): `bash scripts/smoke.sh`
5) train ckan on cifar-10:
   `/usr/local/bin/python3 -m ckancnn.cli.train --model ckan_small --dataset cifar10 --epochs 2 --batch-size 128`
6) evaluate with mc dropout, calibration metrics:
   `/usr/local/bin/python3 -m ckancnn.cli.evaluate --checkpoint outputs/ckan_small_cifar10/latest.pt --dataset cifar10 --mc-samples 20`
7) explainability visuals:
   `/usr/local/bin/python3 -m ckancnn.cli.explain --checkpoint outputs/ckan_small_cifar10/latest.pt --dataset cifar10 --num-samples 8`
7) budgeted comparison (quick demo epochs):
   `/usr/local/bin/python3 -m ckancnn.cli.compare --dataset cifar10 --param-budget 1000000 --epochs 2`

parameter budget matching
- use `--param-budget` for param count target
- cli tunes width multiplier per model for target within tolerance

data
- datasets via torchvision, auto download under `data/` on first use
- offline: ensure datasets present under `data/`

outputs
- checkpoints, logs, csv metrics under `outputs/summary.csv`

notes
- ckan design practical, interpretable, adjust k & prototype dim blocks when needed
