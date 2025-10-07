import argparse
import os
import json

import torch
import torchvision
from torchvision.utils import save_image

from ..data import build_dataloaders
from ..models import create_model
from ..explain import GradCAM, CKANAttentionMaps
from ..utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM and CKAN attention maps")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="ckan_small")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "mnist", "svhn", "stl10"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def _find_last_conv(module: torch.nn.Module) -> torch.nn.Module:
    last = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM target.")
    return last


def main():
    args = parse_args()
    device = get_device()
    _, val_loader, num_classes, in_channels, image_size = build_dataloaders(
        args.dataset, batch_size=args.batch_size, root=args.data_root, download=(not args.no_download)
    )

    # load width multiplier from checkpoint meta or run.json
    width_mult = 1.0
    ckpt = torch.load(args.checkpoint, map_location=device)
    meta = ckpt.get("meta", None)
    if meta and isinstance(meta, dict):
        width_mult = float(meta.get("width_mult", width_mult))
    else:
        try:
            with open(os.path.join(os.path.dirname(args.checkpoint), "run.json"), "r") as f:
                run_meta = json.load(f)
                width_mult = float(run_meta.get("width_mult", width_mult))
        except Exception:
            pass

    model = create_model(args.model, num_classes=num_classes, width_mult=width_mult, in_channels=in_channels)
    model.load_state_dict(ckpt["state_dict"])  # type: ignore
    model.eval()
    model.to(device)

    out_dir = args.out_dir if args.out_dir else os.path.join(os.path.dirname(args.checkpoint), "explain")
    os.makedirs(out_dir, exist_ok=True)

    images, labels = next(iter(val_loader))
    images = images[: args.num_samples].to(device)
    labels = labels[: args.num_samples].to(device)

    target_layer = _find_last_conv(model)
    gradcam = GradCAM(model, target_layer)
    cams = gradcam(images)  # shape n 1 h w
    gradcam.remove_hooks()

    # save images and cam overlays
    imgs = images.detach().cpu().clone()
    imgs = (imgs * 0.5) + 0.5  # unnormalize assume mean .5 std .5
    cam_resized = torch.nn.functional.interpolate(cams, size=imgs.shape[-2:], mode="bilinear", align_corners=False)
    overlay = (0.6 * imgs) + (0.4 * cam_resized.expand_as(imgs))
    save_image(imgs, os.path.join(out_dir, "images.png"), nrow=min(4, args.num_samples))
    save_image(overlay, os.path.join(out_dir, "gradcam_overlay.png"), nrow=min(4, args.num_samples))

    # ckan attention maps when available
    try:
        ckan_maps = CKANAttentionMaps(model)
        maps = ckan_maps(images)
        ckan_maps.remove_hooks()
        if maps.numel() > 0:
            maps_avg = maps.mean(dim=1, keepdim=True)  # mean over all prototypes across blocks
            maps_avg = torch.nn.functional.interpolate(maps_avg, size=imgs.shape[-2:], mode="bilinear", align_corners=False)
            overlay_ckan = (0.6 * imgs) + (0.4 * maps_avg.expand_as(imgs))
            save_image(overlay_ckan, os.path.join(out_dir, "ckan_attention_overlay.png"), nrow=min(4, args.num_samples))
    except Exception:
        pass

    print(f"Saved explainability images to: {out_dir}")


if __name__ == "__main__":
    main()
#!/usr/local/bin/python3
