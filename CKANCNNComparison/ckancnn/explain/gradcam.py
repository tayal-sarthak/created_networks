from typing import List, Tuple

import torch
import torch.nn as nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.fwd_handle = target_layer.register_forward_hook(forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def __call__(self, inputs: torch.Tensor, class_idx: int = None) -> torch.Tensor:
        self.model.zero_grad()
        outputs = self.model(inputs)
        if class_idx is None:
            class_idx = outputs.argmax(dim=1)
            scores = outputs.gather(1, class_idx.view(-1, 1)).squeeze()
        else:
            if isinstance(class_idx, int):
                class_idx = torch.full((inputs.size(0),), class_idx, dtype=torch.long, device=outputs.device)
            scores = outputs.gather(1, class_idx.view(-1, 1)).squeeze()
        scores.sum().backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # weights mean over spatial dims shape n c 1 1
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # cam shape n 1 h w
        cam = torch.relu(cam)
        # normalize per sample in [0,1]
        n, _, h, w = cam.shape
        cam = cam.view(n, -1)
        mins = cam.min(dim=1, keepdim=True)[0]
        maxs = cam.max(dim=1, keepdim=True)[0]
        cam = (cam - mins) / (maxs - mins + 1e-8)
        cam = cam.view(n, 1, h, w)
        return cam
