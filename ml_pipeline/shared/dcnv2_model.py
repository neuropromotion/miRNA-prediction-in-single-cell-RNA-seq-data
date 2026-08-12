"""Compact DCNv2 for tabular regression (cross + deep, parallel stack)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CrossLayerV2(nn.Module):
    def __init__(self, d_in: int, low_rank: int = 32) -> None:
        super().__init__()
        self.U = nn.Parameter(torch.empty(d_in, low_rank))
        self.V = nn.Parameter(torch.empty(d_in, low_rank))
        self.bias = nn.Parameter(torch.zeros(d_in))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)

    def forward(self, x0: Tensor, x: Tensor) -> Tensor:
        return x0 * (x @ self.U @ self.V.T) + self.bias + x


class DCNv2(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int = 1,
        n_cross_layers: int = 3,
        cross_low_rank: int = 32,
        d_deep: int = 256,
        n_deep_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cross = nn.ModuleList(
            CrossLayerV2(d_in, cross_low_rank) for _ in range(n_cross_layers)
        )
        deep_layers: list[nn.Module] = []
        din = d_in
        for _ in range(n_deep_layers):
            deep_layers.extend([nn.Linear(din, d_deep), nn.ReLU(), nn.Dropout(dropout)])
            din = d_deep
        self.deep = nn.Sequential(*deep_layers)
        self.head = nn.Linear(d_in + d_deep, d_out)

    def forward(self, x: Tensor) -> Tensor:
        x0 = x
        xc = x
        for layer in self.cross:
            xc = layer(x0, xc)
        xd = self.deep(x)
        return self.head(torch.cat([xc, xd], dim=1))
