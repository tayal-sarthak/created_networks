import argparse
import os
import json
from typing import Optional

import torch
import torch.nn as nn

from ..data import build_dataloaders
from ..models import create_model, list_models
from ..trainers import Trainer, TrainConfig
from ..utils import set_seed, get_device
from ..utils.model_info import count_params as count_params_util, parameter_memory_bytes, measure_peak_memory_forward


def _count_params(model: nn.Module) -> int:
    return count_params_util(model)


def _find_width_for_budget(create_fn, num_classes: int, in_channels: int, target_params: int, tol: float = 0.05):
    # search width, minimize abs error vs target; allow small widths
    lo = 0.05
    hi = 4.0
    best = 1.0
    best_err = float("inf")
    for _ in range(24):
        mid = (lo + hi) / 2.0
        model = create_fn(num_classes=num_classes, width_mult=mid, in_channels=in_channels)
        params = _count_params(model)
        err = abs(params - target_params)
        if err < best_err:
            best_err = err
            best = mid
        rel = err / max(1, target_params)
        if rel <= tol:
            best = mid
            break
        if params < target_params:
            lo = mid
        else:
            hi = mid
    return best


def parse_args():
    parser = argparse.ArgumentParser(description="Train CKAN/CNN models with fixed parameter budgets")
    parser.add_argument("--model", type=str, default="ckan_small", choices=list_models())
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "mnist", "svhn", "stl10"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--param-budget", type=int, default=0, help="Target number of trainable parameters (approximate). 0 disables matching.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--mc-dropout", action="store_true")
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--measure-memory", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    train_loader, val_loader, num_classes, in_channels, image_size = build_dataloaders(
        args.dataset, batch_size=args.batch_size, root=args.data_root, download=(not args.no_download)
    )

    create_fn = lambda num_classes, width_mult, in_channels: create_model(
        args.model, num_classes=num_classes, width_mult=width_mult, in_channels=in_channels
    )

    width_mult = 1.0
    if args.param_budget and args.param_budget > 0:
        width_mult = _find_width_for_budget(create_fn, num_classes, in_channels, args.param_budget, tol=0.05)

    model = create_fn(num_classes=num_classes, width_mult=width_mult, in_channels=in_channels)
    params = _count_params(model)

    run_name = args.run_name.strip()
    if run_name == "":
        run_name = f"{args.model}_{args.dataset}"
    out_dir = os.path.join("outputs", run_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model: {args.model}, width_mult={width_mult:.3f}, params={params:,}")
    print(f"Parameter memory (float32): {parameter_memory_bytes(model)/1e6:.2f} MB")
    print(f"Dataset: {args.dataset} ({num_classes} classes)")
    print(f"Outputs: {out_dir}")

    # save run metadata for robust evaluation later
    run_meta = {
        "model": args.model,
        "width_mult": float(width_mult),
        "dataset": args.dataset,
        "num_classes": int(num_classes),
        "in_channels": int(in_channels),
        "image_size": int(image_size),
        "params": int(params),
    }
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    if args.measure_memory and device.type == "cuda":
        sample = torch.randn(min(4, args.batch_size), in_channels, image_size, image_size)
        peak_bytes = measure_peak_memory_forward(model.to(device), sample, device)
        print(f"Peak forward memory (approx): {peak_bytes/1e6:.2f} MB")

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        label_smoothing=args.label_smoothing,
        mc_dropout=args.mc_dropout,
        mc_samples=args.mc_samples,
        mixed_precision=args.mixed_precision and device.type == "cuda",
    )
    trainer = Trainer(model=model, device=device, num_classes=num_classes, log_dir=out_dir, config=cfg, run_meta=run_meta)
    trainer.fit(train_loader, val_loader, initial_lr=args.lr)


if __name__ == "__main__":
    main()
#!/usr/local/bin/python3
