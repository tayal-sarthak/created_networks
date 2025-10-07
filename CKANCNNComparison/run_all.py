#!/usr/local/bin/python3
"""
one-click pipeline for ckan vs cnn under fixed parameter budget

does in one run:
- optional deps install < *IT INSTALLED requirements.txt. You can make it install nothing by simply removing requirements.txt!!!!
- per dataset, per model:
  - match budget via width search
  - train few epochs
  - evaluate: accuracy, loss, ece, nll
  - mc dropout eval for calibration
  - save grad-cam, ckan attention overlays
  - save checkpoints, epoch metrics
- write summary csv across runs

config section below controls datasets, models, epochs, budget
"""

import os
import sys
import csv
import time
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Tuple


# config

# install deps flag
INSTALL_REQUIREMENTS = True

DATASETS: List[str] = [
    "cifar10",
    # "mnist",      # enable when needed
    # "svhn",       # larger download
    # "cifar100",   # harder task
    # "stl10",      # large download, slower
]

MODELS: List[str] = [
    "ckan_small",
    "resnet18",
    "convnext_tiny",
]

PARAM_BUDGET = 1_000_000          # trainable parameters target (approx)
EPOCHS = 2                         # small for a quick demo; increase for better results
BATCH_SIZE = 128
SEED = 42
DATA_ROOT = "data"

# bayesian style eval
MC_DROPOUT = True
MC_SAMPLES = 20

# explainability
NUM_EXPLAIN_IMAGES = 8

# output location
ROOT_OUTPUT_DIR = os.path.join("outputs", "chain")


# helpers

def run(cmd: List[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def try_install_requirements():
    if not INSTALL_REQUIREMENTS:
        print("[install] Skipped requirements installation (INSTALL_REQUIREMENTS=False)")
        return
    req_path = os.path.join(os.getcwd(), "requirements.txt")
    if not os.path.exists(req_path):
        print("[install] requirements.txt not found; skipping installation")
        return
    print("[install] Installing dependencies from requirements.txt …")
    code = run(["/usr/local/bin/python3", "-m", "pip", "install", "-r", "requirements.txt"]) 
    if code != 0:
        print("[install] pip install failed (continuing anyway). You may need to install manually.")


def ensure_dirs():
    os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)


def main():
    start_time = time.time()
    ensure_dirs()
    try_install_requirements()

    # import heavy deps after optional install
    from ckancnn.utils import set_seed, get_device
    from ckancnn.data import build_dataloaders
    from ckancnn.models import create_model
    from ckancnn.trainers import Trainer, TrainConfig
    from ckancnn.utils.model_info import count_params, parameter_memory_bytes
    from ckancnn.explain import GradCAM, CKANAttentionMaps
    import torch
    import torch.nn as nn

    set_seed(SEED)
    device = get_device()
    print(f"[env] Device: {device}")

    summary_rows: List[Dict[str, object]] = []

    def find_width_for_budget(factory, num_classes: int, in_channels: int, target_params: int, tol: float = 0.05) -> float:
        lo, hi = 0.05, 4.0
        best = 1.0
        best_err = float("inf")
        for _ in range(24):
            mid = (lo + hi) / 2.0
            model = factory(num_classes=num_classes, width_mult=mid, in_channels=in_channels)
            p = count_params(model)
            err = abs(p - target_params)
            if err < best_err:
                best_err = err
                best = mid
            if target_params > 0 and (err / target_params) <= tol:
                best = mid
                break
            if p < target_params:
                lo = mid
            else:
                hi = mid
        return best

    for dataset in DATASETS:
        print(f"\n=== Dataset: {dataset} ===")
        train_loader, val_loader, num_classes, in_channels, image_size = build_dataloaders(
            dataset, batch_size=BATCH_SIZE, root=DATA_ROOT, download=True
        )

        for model_name in MODELS:
            print(f"\n--- Model: {model_name} ---")

            create_fn = lambda num_classes, width_mult, in_channels: create_model(
                model_name, num_classes=num_classes, width_mult=width_mult, in_channels=in_channels
            )
            width_mult = find_width_for_budget(create_fn, num_classes, in_channels, PARAM_BUDGET, tol=0.05)
            model = create_fn(num_classes=num_classes, width_mult=width_mult, in_channels=in_channels)
            params = count_params(model)
            param_mem_mb = parameter_memory_bytes(model) / 1e6

            out_dir = os.path.join(ROOT_OUTPUT_DIR, dataset, f"{model_name}_p{params}")
            os.makedirs(out_dir, exist_ok=True)
            print(f"[info] width_mult={width_mult:.3f}, params={params:,}, param_mem≈{param_mem_mb:.2f} MB")
            print(f"[out] {out_dir}")

            # train
            train_cfg = TrainConfig(
                epochs=EPOCHS,
                lr=0.1,
                weight_decay=5e-4,
                momentum=0.9,
                label_smoothing=0.0,
                mixed_precision=(device.type == "cuda"),
            )
            run_meta = {
                "model": model_name,
                "width_mult": float(width_mult),
                "dataset": dataset,
                "num_classes": int(num_classes),
                "in_channels": int(in_channels),
                "image_size": int(image_size),
                "params": int(params),
            }
            # save run.json for downstream tools
            with open(os.path.join(out_dir, "run.json"), "w") as f:
                import json as _json
                _json.dump(run_meta, f, indent=2)

            trainer = Trainer(model=model.to(device), device=device, num_classes=num_classes, log_dir=out_dir, config=train_cfg, run_meta=run_meta)
            trainer.fit(train_loader, val_loader)

            # standard eval
            std_metrics = trainer.evaluate(val_loader)

            # mc dropout eval
            eval_cfg = TrainConfig(epochs=1, mc_dropout=MC_DROPOUT, mc_samples=MC_SAMPLES)
            eval_trainer = Trainer(model=model, device=device, num_classes=num_classes, log_dir=out_dir, config=eval_cfg)
            mc_metrics = eval_trainer.evaluate(val_loader)

            # explain overlays
            try:
                # grad-cam over last conv
                def find_last_conv(m: nn.Module) -> nn.Module:
                    last = None
                    for sub in m.modules():
                        if isinstance(sub, nn.Conv2d):
                            last = sub
                    if last is None:
                        raise RuntimeError("No Conv2d layer found for Grad-CAM target.")
                    return last

                images, labels = next(iter(val_loader))
                images = images[: NUM_EXPLAIN_IMAGES].to(device)

                target_layer = find_last_conv(model)
                gradcam = GradCAM(model, target_layer)
                cams = gradcam(images)
                gradcam.remove_hooks()

                imgs = images.detach().cpu().clone()
                # unnormalize assume mean .5 std .5 in loaders
                imgs = (imgs * 0.5) + 0.5
                cams_resized = torch.nn.functional.interpolate(cams, size=imgs.shape[-2:], mode="bilinear", align_corners=False)
                overlay = (0.6 * imgs) + (0.4 * cams_resized.expand_as(imgs))

                from torchvision.utils import save_image
                exp_dir = os.path.join(out_dir, "explain")
                os.makedirs(exp_dir, exist_ok=True)
                save_image(imgs, os.path.join(exp_dir, "images.png"), nrow=min(4, NUM_EXPLAIN_IMAGES))
                save_image(overlay, os.path.join(exp_dir, "gradcam_overlay.png"), nrow=min(4, NUM_EXPLAIN_IMAGES))

                # ckan attention maps when model has ckan blocks
                try:
                    ckan_maps = CKANAttentionMaps(model)
                    maps = ckan_maps(images)
                    ckan_maps.remove_hooks()
                    if maps.numel() > 0:
                        maps_avg = maps.mean(dim=1, keepdim=True)
                        maps_resized = torch.nn.functional.interpolate(maps_avg, size=imgs.shape[-2:], mode="bilinear", align_corners=False)
                        overlay_ckan = (0.6 * imgs) + (0.4 * maps_resized.expand_as(imgs))
                        save_image(overlay_ckan, os.path.join(exp_dir, "ckan_attention_overlay.png"), nrow=min(4, NUM_EXPLAIN_IMAGES))
                except Exception:
                    pass
            except Exception as e:
                print(f"[explain] Skipped explainability for {model_name}: {e}")

            # collect summary
            row = {
                "dataset": dataset,
                "model": model_name,
                "width_mult": round(float(width_mult), 3),
                "params": int(params),
                "param_mem_mb": round(float(param_mem_mb), 3),
                "val_loss": round(float(std_metrics.get("loss", 0.0)), 4),
                "val_acc": round(float(std_metrics.get("acc", 0.0)), 4),
                "val_ece": round(float(std_metrics.get("ece", 0.0)), 4),
                "val_nll": round(float(std_metrics.get("nll", 0.0)), 4),
                "mc_ece": round(float(mc_metrics.get("ece", 0.0)), 4),
                "mc_nll": round(float(mc_metrics.get("nll", 0.0)), 4),
                "out_dir": out_dir,
            }
            summary_rows.append(row)
            print("[summary]", row)

    # write summary csv
    summary_csv = os.path.join(ROOT_OUTPUT_DIR, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset", "model", "width_mult", "params", "param_mem_mb",
                "val_loss", "val_acc", "val_ece", "val_nll", "mc_ece", "mc_nll", "out_dir",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    elapsed = time.time() - start_time
    print(f"\nAll done. Summary: {summary_csv}")
    print(f"Total elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
