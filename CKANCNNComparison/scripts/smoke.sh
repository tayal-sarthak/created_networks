#!/usr/bin/env bash
set -euo pipefail

/usr/local/bin/python3 - << 'PY'
import torch
from ckancnn.models import create_model

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

for name in ["ckan_small", "ckan_base", "resnet18", "convnext_tiny"]:
    model = create_model(name, num_classes=10, width_mult=0.25, in_channels=3)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(name, y.shape, count_params(model))
print("Smoke OK")
PY
