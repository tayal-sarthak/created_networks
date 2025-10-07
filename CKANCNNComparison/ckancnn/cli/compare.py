import argparse
import os

from ..data import build_dataloaders
from ..models import create_model
from ..trainers import Trainer, TrainConfig
from ..utils import get_device, set_seed
from ..utils.model_info import count_params


def parse_args():
    parser = argparse.ArgumentParser(description="Compare CKAN vs CNNs under fixed parameter budgets")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "mnist", "svhn", "stl10"])
    parser.add_argument("--models", type=str, nargs="*", default=["ckan_small", "resnet18", "convnext_tiny"])
    parser.add_argument("--param-budget", type=int, default=1_000_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    train_loader, val_loader, num_classes, in_channels, image_size = build_dataloaders(
        args.dataset, batch_size=args.batch_size, root=args.data_root, download=(not args.no_download)
    )

    results = []
    for model_name in args.models:
        # search width for budget, minimize abs error
        def create(width_mult):
            return create_model(model_name, num_classes=num_classes, width_mult=width_mult, in_channels=in_channels)

        lo, hi = 0.05, 4.0
        target = args.param_budget
        width = 1.0
        best_err = float("inf")
        for _ in range(24):
            mid = (lo + hi) / 2
            m = create(mid)
            p = count_params(m)
            err = abs(p - target)
            if err < best_err:
                best_err = err
                width = mid
            if target > 0 and (err / target) <= 0.05:
                width = mid
                break
            if p < target:
                lo = mid
            else:
                hi = mid
        model = create(width).to(device)
        params = count_params(model)

        out_dir = os.path.join("outputs", f"compare_{args.dataset}", f"{model_name}_p{params}")
        os.makedirs(out_dir, exist_ok=True)

        cfg = TrainConfig(epochs=args.epochs, lr=0.1, weight_decay=5e-4, momentum=0.9)
        run_meta = {
            "model": model_name,
            "width_mult": float(width),
            "dataset": args.dataset,
            "num_classes": int(num_classes),
            "in_channels": int(in_channels),
            "params": int(params),
        }
        # save run.json for consistency
        import json
        with open(os.path.join(out_dir, "run.json"), "w") as f:
            json.dump(run_meta, f, indent=2)

        trainer = Trainer(model=model, device=device, num_classes=num_classes, log_dir=out_dir, config=cfg, run_meta=run_meta)
        trainer.fit(train_loader, val_loader)
        results.append((model_name, params, out_dir))

    print("Finished. Results saved under:")
    for name, params, out_dir in results:
        print(f"- {name} (params={params}) -> {out_dir}")


if __name__ == "__main__":
    main()
#!/usr/local/bin/python3
