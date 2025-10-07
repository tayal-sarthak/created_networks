import torch
import torch.nn.functional as F


def ece_score(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> torch.Tensor:
    # probs shape n c, targets n
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(targets)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    n = probs.size(0)
    for i in range(n_bins):
        start = bin_boundaries[i]
        end = bin_boundaries[i + 1]
        mask = (confidences > start) & (confidences <= end)
        if mask.any():
            conf = confidences[mask].mean()
            acc = accuracies[mask].float().mean()
            ece += (mask.float().mean()) * (conf - acc).abs()
    return ece


def nll_loss(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    safe_probs = (probs + eps).clamp(max=1.0)
    logp = torch.log(safe_probs)
    nll = F.nll_loss(logp, targets, reduction="mean")
    return nll
