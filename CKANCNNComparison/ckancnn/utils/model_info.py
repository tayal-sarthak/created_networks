import torch
import torch.nn as nn


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_memory_bytes(model: nn.Module, dtype_bytes: int = 4) -> int:
    params = count_params(model)
    return params * dtype_bytes


@torch.no_grad()
def measure_peak_memory_forward(model: nn.Module, sample: torch.Tensor, device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    _ = model(sample.to(device))
    peak = torch.cuda.max_memory_allocated(device)
    return int(peak)
