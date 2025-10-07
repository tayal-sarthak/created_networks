from typing import Dict, Callable

from .ckan import CKANSmall, CKANBase
from .cnn_baselines import SimpleResNet18, SimpleConvNeXtTiny


_MODEL_FACTORIES: Dict[str, Callable] = {
    "ckan_small": CKANSmall,
    "ckan_base": CKANBase,
    "resnet18": SimpleResNet18,
    "convnext_tiny": SimpleConvNeXtTiny,
}


def create_model(name: str, num_classes: int, width_mult: float = 1.0, **kwargs):
    key = name.lower()
    if key not in _MODEL_FACTORIES:
        available = ", ".join(sorted(_MODEL_FACTORIES.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _MODEL_FACTORIES[key](num_classes=num_classes, width_mult=width_mult, **kwargs)


def list_models():
    return sorted(list(_MODEL_FACTORIES.keys()))
