from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from ..models.ckan import KernelAttention, CKANBlock


class CKANAttentionMaps:
    def __init__(self, model: nn.Module):
        self.model = model
        self.attn_outputs = []
        self.hooks = []

        def hook_attn(module: nn.Module, inp, out):
            # out shape n k h w
            self.attn_outputs.append(out.detach())

        # register hooks on kernelattention modules
        for m in self.model.modules():
            if isinstance(m, KernelAttention):
                self.hooks.append(m.register_forward_hook(hook_attn))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    @torch.no_grad()
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        self.attn_outputs.clear()
        _ = self.model(inputs)
        # concat attention maps across all ckan blocks on channel
        if len(self.attn_outputs) == 0:
            return torch.empty(0)
        # pick target spatial size as max h w among blocks
        sizes: List[Tuple[int, int]] = [(t.shape[-2], t.shape[-1]) for t in self.attn_outputs]
        Ht = max(h for h, w in sizes)
        Wt = max(w for h, w in sizes)
        # upsample each into (Ht, Wt) then concat on channel
        ups = [
            torch.nn.functional.interpolate(t, size=(Ht, Wt), mode="bilinear", align_corners=False)
            for t in self.attn_outputs
        ]
        maps = torch.cat(ups, dim=1)  # shape n sum(k) ht wt
        # normalize per sample
        n = maps.size(0)
        maps_flat = maps.view(n, -1)
        mins = maps_flat.min(dim=1, keepdim=True)[0]
        maxs = maps_flat.max(dim=1, keepdim=True)[0]
        maps_norm = (maps_flat - mins) / (maxs - mins + 1e-8)
        maps_norm = maps_norm.view_as(maps)
        return maps_norm
