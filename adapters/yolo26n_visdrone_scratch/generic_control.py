"""Generic matched-control blocks for P9 attribution (adapter-owned).

These deliberately contain NO cross-domain structure — they are the "boring"
alternatives a mechanism must beat for its functional form to be credited.
"""
from __future__ import annotations

import torch
from torch import nn


class GenericResidualConv(nn.Module):
    """Wraps an existing block and adds a residual generic conv branch whose
    parameter count is tuned to `target_params` (±10%).

    out = original(x) + branch(original(x))
    branch = 1x1 conv (c->h) -> SiLU -> 1x1 conv (h->c), zero-init last layer
    so training starts at the original function (fair start for both
    mechanism and control when the mechanism also starts near-neutral;
    the P9 comparison is between their TRAINED outcomes either way).
    """

    def __init__(self, original: nn.Module, target_params: int):
        super().__init__()
        self.original = original
        c = self._out_channels(original)
        # solve 2*c*h + h (biases) ~= target_params for hidden width h
        h = max(1, int(target_params / (2 * c + 1)))
        self.branch = nn.Sequential(
            nn.Conv2d(c, h, 1), nn.SiLU(), nn.Conv2d(h, c, 1),
        )
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)

    @staticmethod
    def _out_channels(module: nn.Module) -> int:
        for m in reversed(list(module.modules())):
            if isinstance(m, nn.Conv2d):
                return m.out_channels
            if isinstance(m, nn.BatchNorm2d):
                return m.num_features
        raise ValueError("cannot infer out_channels of wrapped block")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.original(x)
        return y + self.branch(y)
