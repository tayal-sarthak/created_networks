
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

try:
    from torchvision.models import get_model, get_model_weights
    HAS_GET_MODEL = True
except Exception:
    import torchvision.models as models  # type: ignore
    HAS_GET_MODEL = False


CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "speckle_noise",
    "gaussian_blur",
    "spatter",
    "saturate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="evaluate models on imagenet-c")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--imagenet-c", type=Path, required=True)
    parser.add_argument("--imagenet-val", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--precision", choices=("fp32", "amp"), default="fp32")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--corruptions",
        nargs="*",
        default=None,
        help="optional list of corruption names to evaluate; default=all present",
    )
    parser.add_argument(
        "--severities",
        nargs="*",
        type=int,
        default=None,
        help="optional list of severities to evaluate; default=1 2 3 4 5",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="limit images per class per severity (0 means no limit)",
    )
    return parser.parse_args()


def resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    clean_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    corruption_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return clean_transform, corruption_transform


def load_model(name: str, device: torch.device) -> torch.nn.Module:
    if HAS_GET_MODEL:
        weights = None
        try:
            weights_enum = get_model_weights(name)
            if hasattr(weights_enum, "DEFAULT"):
                weights = weights_enum.DEFAULT
        except Exception:
            weights = None
        model = get_model(name, weights=weights)
    else:
        if not hasattr(models, name):  # type: ignore[name-defined]
            raise ValueError(f"unknown model: {name}")
        ctor = getattr(models, name)  # type: ignore[name-defined]
        try:
            model = ctor(pretrained=True)
        except TypeError:
            model = ctor()
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def evaluate_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> Dict[str, float]:
    total = 0
    top1 = 0
    top5 = 0
    if use_amp and device.type == "cuda":
        autocast = torch.cuda.amp.autocast
    else:
        class _NC:
            def __enter__(self):
                return None
            def __exit__(self, exc_type, exc, tb):
                return False
        autocast = _NC

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        with autocast():
            logits = model(images)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        max_k = min(5, logits.size(1))
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
        correct = pred.eq(targets.view(-1, 1))
        top1 += correct[:, :1].sum().item()
        top5 += correct[:, :max_k].sum().item()
        total += targets.size(0)

    top1_acc = top1 / total if total > 0 else float("nan")
    top5_acc = top5 / total if total > 0 else float("nan")
    return {"top1": top1_acc, "top5": top5_acc, "count": total}


def _limit_per_class(ds: datasets.ImageFolder, max_n: int) -> datasets.ImageFolder:
    if max_n <= 0:
        return ds
    # Build a mask that keeps at most max_n samples per target class
    kept = []
    per_class: Dict[int, int] = {}
    for path, target in ds.samples:  # type: ignore[attr-defined]
        count = per_class.get(target, 0)
        if count < max_n:
            kept.append((path, target))
            per_class[target] = count + 1
    ds.samples = kept  # type: ignore[attr-defined]
    ds.imgs = kept  # for older torchvision
    return ds


def build_imagenet_c(
    root: Path,
    transform: transforms.Compose,
    corruptions: List[str] | None,
    severities: List[int] | None,
    max_per_class: int,
) -> Dict[str, Dict[int, datasets.ImageFolder]]:
    if corruptions is None:
        # choose those present at root
        corruptions = [c for c in CORRUPTIONS if (root / c).exists()]
    if severities is None:
        severities = [1, 2, 3, 4, 5]
    out: Dict[str, Dict[int, datasets.ImageFolder]] = {}
    for name in corruptions:
        if not (root / name).exists():
            print(f"  skip missing corruption: {name}")
            continue
        per_sev: Dict[int, datasets.ImageFolder] = {}
        for sev in severities:
            folder = root / name / str(sev)
            if not folder.exists():
                print(f"  skip missing severity {sev} for {name}")
                continue
            ds = datasets.ImageFolder(folder.as_posix(), transform=transform)
            ds = _limit_per_class(ds, max_per_class)
            per_sev[sev] = ds
        if per_sev:
            out[name] = per_sev
    if not out:
        raise SystemExit("no valid corruptions found under imagenet-c path")
    return out


def print_summary(rows: Dict[str, Dict[str, float]]) -> None:
    print()
    # decide whether to show clean columns
    show_clean = any(
        not math.isnan(m.get("clean_top1", float("nan"))) for m in rows.values()
    )
    if show_clean:
        print("model           | clean@1 | clean@5 | mCE (unnorm)")
        print("--------------------------------------------------")
        for name, m in rows.items():
            c1 = m.get("clean_top1", float("nan"))
            c5 = m.get("clean_top5", float("nan"))
            mce = m.get("avg_corruption_top1_error", float("nan"))
            c1s = f"{c1 * 100:.2f}" if not math.isnan(c1) else "skipped"
            c5s = f"{c5 * 100:.2f}" if not math.isnan(c5) else "skipped"
            mces = f"{mce * 100:.2f}" if not math.isnan(mce) else "nan"
            print(f"{name:15s} | {c1s:>7} | {c5s:>7} | {mces:>11}")
    else:
        print("model           | mCE (unnorm)")
        print("------------------------------")
        for name, m in rows.items():
            mce = m.get("avg_corruption_top1_error", float("nan"))
            mces = f"{mce * 100:.2f}" if not math.isnan(mce) else "nan"
            print(f"{name:15s} | {mces:>11}")
    print()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.backends.cudnn.benchmark = device.type == "cuda"

    clean_t, corrupt_t = build_transforms()
    clean_loader = None
    if args.imagenet_val is not None and args.imagenet_val.exists():
        clean_ds = datasets.ImageFolder(args.imagenet_val.as_posix(), transform=clean_t)
        clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    csets = build_imagenet_c(
        args.imagenet_c,
        corrupt_t,
        corruptions=args.corruptions,
        severities=args.severities,
        max_per_class=args.max_per_class,
    )

    all_results: Dict[str, Dict] = {}
    summary: Dict[str, Dict[str, float]] = {}

    for model_name in args.models:
        print(f"\nmodel: {model_name}")
        model = load_model(model_name, device)

        if clean_loader is not None:
            clean = evaluate_loader(model, clean_loader, device, use_amp=(args.precision == "amp"))
            print(f"  clean top1 {clean['top1'] * 100:.2f}%  top5 {clean['top5'] * 100:.2f}%")
        else:
            # silently skip clean if not provided
            clean = {"top1": float("nan"), "top5": float("nan"), "count": 0}

        errs: Dict[str, List[float]] = {}
        for cname in csets.keys():
            sev_errors: List[float] = []
            per_sev = csets[cname]
            for sev in sorted(per_sev.keys()):
                ds = per_sev[sev]
                ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
                m = evaluate_loader(model, ld, device, use_amp=(args.precision == "amp"))
                sev_errors.append(1.0 - m["top1"])
            mean_err = sum(sev_errors) / len(sev_errors)
            print(f"  {cname:15s}: mean top1 error {mean_err * 100:.2f}%")
            errs[cname] = sev_errors

        corruption_means: List[float] = []
        for cname in errs.keys():
            values = errs[cname]
            corruption_means.append(sum(values) / len(values))
        avg_err = sum(corruption_means) / len(corruption_means)

        summary[model_name] = {
            "clean_top1": clean["top1"],
            "clean_top5": clean["top5"],
            "avg_corruption_top1_error": avg_err,
            "avg_corruption_top1_accuracy": 1.0 - avg_err,
        }

        all_results[model_name] = {
            "clean": clean,
            "corruptions": errs,
            "summary": summary[model_name],
        }

    print_summary(summary)

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(all_results, indent=2))
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
