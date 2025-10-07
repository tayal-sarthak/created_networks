from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(value: int, divisor: int) -> int:
    if value % divisor == 0:
        return value
    return value + (divisor - value % divisor)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out


class SimpleResNet(nn.Module):
    def __init__(self, block: nn.Module, num_blocks: Tuple[int, int, int, int], num_classes: int, width_mult: float = 1.0, in_channels: int = 3):
        super().__init__()
        base = [64, 128, 256, 512]
        channels = tuple(_make_divisible(int(c * width_mult), 8) for c in base)
        self.in_planes = channels[0]

        self.conv1 = nn.Conv2d(in_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.layer1 = self._make_layer(block, channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], num_blocks[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out).flatten(1)
        out = self.fc(out)
        return out


class SimpleResNet18(nn.Module):
    def __init__(self, num_classes: int, width_mult: float = 1.0, in_channels: int = 3):
        super().__init__()
        self.model = SimpleResNet(BasicBlock, (2, 2, 2, 2), num_classes=num_classes, width_mult=width_mult, in_channels=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dw(x)
        # nchw → nhwc for norm and linear
        n, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = shortcut + self.drop_path(x)
        return x


class SimpleConvNeXt(nn.Module):
    def __init__(self, depths: Tuple[int, int, int, int], dims: Tuple[int, int, int, int], num_classes: int, in_channels: int = 3):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        # groupnorm(1,c) as layernorm nchw
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0], eps=1e-6),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            down = nn.Sequential(
                nn.GroupNorm(1, dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(down)

        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = [ConvNeXtBlock(dims[i]) for _ in range(depths[i])]
            self.stages.append(nn.Sequential(*blocks))

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        x = self.downsample_layers[1](x)
        x = self.stages[1](x)
        x = self.downsample_layers[2](x)
        x = self.stages[2](x)
        x = self.downsample_layers[3](x)
        x = self.stages[3](x)
        # nhwc norm
        x = x.mean([-2, -1])  # global avg pool
        x = self.norm(x)
        x = self.head(x)
        return x


class SimpleConvNeXtTiny(nn.Module):
    def __init__(self, num_classes: int, width_mult: float = 1.0, in_channels: int = 3):
        super().__init__()
        base_dims = [96, 192, 384, 768]
        dims = tuple(_make_divisible(int(d * width_mult), 8) for d in base_dims)
        depths = (3, 3, 9, 3)
        self.model = SimpleConvNeXt(depths=depths, dims=dims, num_classes=num_classes, in_channels=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x.div(keep_prob) * random_tensor
