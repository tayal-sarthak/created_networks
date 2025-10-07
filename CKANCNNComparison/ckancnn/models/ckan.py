from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(value: int, divisor: int) -> int:
    if value % divisor == 0:
        return value
    return value + (divisor - value % divisor)


class DepthwiseConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PointwiseExpert(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class KernelAttention(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, num_prototypes: int, gamma: float = 1.0):
        super().__init__()
        self.project = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=True)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embed_dim) * 0.02)
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape n c h w
        q = self.project(x)  # proj shape n d h w
        n, d, h, w = q.shape
        q_flat = q.permute(0, 2, 3, 1).reshape(n * h * w, d)  # flat shape n*h*w d
        p = self.prototypes  # proto shape k d
        # squared distance q p: |q|^2 + |p|^2 − 2 q p^t
        q_norm2 = (q_flat ** 2).sum(dim=1, keepdim=True)  # shape n*h*w 1
        p_norm2 = (p ** 2).sum(dim=1).unsqueeze(0)  # shape 1 k
        two_qp = 2.0 * q_flat @ p.t()  # shape n*h*w k
        dist2 = q_norm2 + p_norm2 - two_qp
        # rbf kernel weights
        scores = -self.gamma.abs() * dist2
        weights = F.softmax(scores, dim=1)  # shape n*h*w k
        weights = weights.view(n, h, w, -1).permute(0, 3, 1, 2).contiguous()  # shape n k h w
        return weights


class CKANBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_experts: int = 4,
        embed_dim: int = 64,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.num_experts = num_experts

        self.dw = DepthwiseConv(in_channels, kernel_size=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.attn = KernelAttention(in_channels, embed_dim, num_experts, gamma)
        self.experts = nn.ModuleList(
            [PointwiseExpert(in_channels, out_channels) for _ in range(num_experts)]
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        y = self.dw(x)
        y = self.bn1(y)
        y = self.act1(y)

        # attention align with expert output spatial size
        weights = self.attn(y)  # shape n k h_out w_out
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(y))  # list shape n c_out h w
        stacked = torch.stack(expert_outputs, dim=1)  # shape n k c_out h w
        # weights expand shape n k h w → n k 1 h w
        weights_expanded = weights.unsqueeze(2)
        mixed = (stacked * weights_expanded).sum(dim=1)  # shape n c_out h w

        mixed = self.bn2(mixed)
        out = self.act2(mixed + residual)
        return out


class CKANStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        num_experts: int,
        embed_dim: int,
        gamma: float,
    ):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            blocks.append(
                CKANBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    stride=s,
                    num_experts=num_experts,
                    embed_dim=embed_dim,
                    gamma=gamma,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class CKAN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        stage_channels: Tuple[int, int, int, int],
        stage_blocks: Tuple[int, int, int, int],
        num_experts: int = 4,
        embed_dim: int = 64,
        gamma: float = 1.0,
        in_channels: int = 3,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stage_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stage_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.stage1 = CKANStage(
            in_channels=stage_channels[0],
            out_channels=stage_channels[0],
            num_blocks=stage_blocks[0],
            stride=1,
            num_experts=num_experts,
            embed_dim=embed_dim,
            gamma=gamma,
        )
        self.stage2 = CKANStage(
            in_channels=stage_channels[0],
            out_channels=stage_channels[1],
            num_blocks=stage_blocks[1],
            stride=2,
            num_experts=num_experts,
            embed_dim=embed_dim,
            gamma=gamma,
        )
        self.stage3 = CKANStage(
            in_channels=stage_channels[1],
            out_channels=stage_channels[2],
            num_blocks=stage_blocks[2],
            stride=2,
            num_experts=num_experts,
            embed_dim=embed_dim,
            gamma=gamma,
        )
        self.stage4 = CKANStage(
            in_channels=stage_channels[2],
            out_channels=stage_channels[3],
            num_blocks=stage_blocks[3],
            stride=2,
            num_experts=num_experts,
            embed_dim=embed_dim,
            gamma=gamma,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(stage_channels[3], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x


class CKANSmall(nn.Module):
    def __init__(
        self,
        num_classes: int,
        width_mult: float = 1.0,
        num_experts: int = 4,
        embed_dim: int = 64,
        gamma: float = 1.0,
        in_channels: int = 3,
    ):
        super().__init__()
        base = [48, 96, 192, 384]
        stage_channels = tuple(_make_divisible(int(c * width_mult), 8) for c in base)
        stage_blocks = (2, 2, 3, 2)
        self.model = CKAN(
            num_classes=num_classes,
            stage_channels=stage_channels,
            stage_blocks=stage_blocks,
            num_experts=num_experts,
            embed_dim=embed_dim,
            gamma=gamma,
            in_channels=in_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CKANBase(nn.Module):
    def __init__(
        self,
        num_classes: int,
        width_mult: float = 1.0,
        num_experts: int = 4,
        embed_dim: int = 96,
        gamma: float = 1.0,
        in_channels: int = 3,
    ):
        super().__init__()
        base = [64, 128, 256, 512]
        stage_channels = tuple(_make_divisible(int(c * width_mult), 8) for c in base)
        stage_blocks = (2, 3, 6, 3)
        self.model = CKAN(
            num_classes=num_classes,
            stage_channels=stage_channels,
            stage_blocks=stage_blocks,
            num_experts=num_experts,
            embed_dim=embed_dim,
            gamma=gamma,
            in_channels=in_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
