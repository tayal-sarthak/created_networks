import argparse
import os
import json

import torch

from ..data import build_dataloaders
from ..models import create_model
from ..trainers.trainer import Trainer, TrainConfig
from ..utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with calibration and MC Dropout")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="ckan_small")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "mnist", "svhn", "stl10"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mc-dropout", action="store_true")
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    train_loader, val_loader, num_classes, in_channels, image_size = build_dataloaders(
        args.dataset, batch_size=args.batch_size, root=args.data_root, download=(not args.no_download)
    )

    # try load meta for width multiplier
    width_mult = 1.0
    ckpt = torch.load(args.checkpoint, map_location=device)
    meta = ckpt.get("meta", None)
    if meta and isinstance(meta, dict):
        width_mult = float(meta.get("width_mult", width_mult))
    else:
        # fallback run.json in same folder
        try:
            with open(os.path.join(os.path.dirname(args.checkpoint), "run.json"), "r") as f:
                run_meta = json.load(f)
                width_mult = float(run_meta.get("width_mult", width_mult))
        except Exception:
            pass
    model = create_model(args.model, num_classes=num_classes, width_mult=width_mult, in_channels=in_channels)
    model.load_state_dict(ckpt["state_dict"])  # type: ignore

    cfg = TrainConfig(epochs=1, mc_dropout=args.mc_dropout, mc_samples=args.mc_samples)
    trainer = Trainer(model=model, device=device, num_classes=num_classes, log_dir=os.path.dirname(args.checkpoint), config=cfg)
    metrics = trainer.evaluate(val_loader)
    print("Evaluation:")
    print({k: float(v) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
#!/usr/local/bin/python3
