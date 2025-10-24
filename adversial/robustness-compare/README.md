# sarthak tayal - robustness-compare w/ blur.tar from ImageNet-C
some tools to evaluate ImageNet classifiers on ImageNet-C and summarize results.
please make sure to have torch and torchvision installed!
Quick start
-----------

Option A: one-click runner (drop-in blur.tar)

1) Place a `blur.tar` file in this folder. The script will extract it into `../ImageNet-C/` automatically.
2) Run:

   python3 run_compare.py

also manual invocation

   python3 compare_models.py --models resnet18 --imagenet-c ../ImageNet-C --device mps

Results
-------
- A summary prints to the terminal.
- Full JSON is written to `results.json`.
- Convert to CSVs with:

   python3 results_to_csv.py

This writes `results_flat.csv` (per-model/corruption/severity) and `summary.csv` (per-model aggregates).

Notes
-----
- See the ImageNet-C dataset and structure at https://github.com/hendrycks/robustness
- Clean ImageNet val is optional. if absent, it’s skipped in the summary.
