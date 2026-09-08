"""Divisive normalization on detection class logits (exp002 mechanism).

Source domain: visual neuroscience — the canonical cortical computation
(Carandini & Heeger, "Normalization as a canonical neural computation",
Nat. Rev. Neurosci. 2012): a unit's response is divided by the pooled
activity of its neighbours, y_i = x_i / (sigma + sum_j w_j x_j)^n.

Applied here in the LOG domain to the one-to-one head's class logits
(B, nc, H, W) at each scale, so it composes with the sigmoid the loss and
inference already apply:

    y_i = x_i - n * log( sigma + sum_j w_j * sigmoid(x_j) )

- w: learned 3x3 depthwise (per-class) surround kernel, centre excluded
  (init 1/8 on the 8 neighbours);
- sigma > 0 (softplus-parameterised, init 1.0), n >= 0 (softplus, init 0.5).
Trained from step 0. Params per scale = 9*nc + 2 (nc=10 -> 92).

Nearest CV block: NMS (post-hoc, non-differentiable, absent in the NMS-free
YOLO26 head) / DETR query self-attention. Difference: a suppression FIELD
(pooled-energy division, local, no attention weights, no feature mixing),
inside the decision head, learned end-to-end.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn


class DivNorm(nn.Module):
    def __init__(self, nc: int, k: int = 3, sigma_init: float = 1.0, n_init: float = 0.5):
        super().__init__()
        self.nc, self.k = nc, k
        w = torch.full((nc, 1, k, k), 1.0 / (k * k - 1))
        w[:, :, k // 2, k // 2] = 0.0                      # surround only
        self.w = nn.Parameter(w)
        inv = lambda v: torch.log(torch.expm1(torch.tensor(float(v))))   # softplus^-1
        self.sigma_raw = nn.Parameter(inv(sigma_init))
        self.n_raw = nn.Parameter(inv(n_init))

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.sigma_raw)

    @property
    def n(self) -> torch.Tensor:
        return F.softplus(self.n_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.conv2d(torch.sigmoid(x), self.w, padding=self.k // 2, groups=self.nc)
        return x - self.n * torch.log(self.sigma + pooled)


def demo() -> None:
    """Self-check: identity when n->0; suppression grows with neighbour activity; grads flow."""
    torch.manual_seed(0)
    m = DivNorm(10)
    x = torch.randn(2, 10, 8, 8, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape
    # a strongly active neighbourhood must reduce the centre logit more than a quiet one
    hot = x.clone(); hot[:, :, 3:6, 3:6] = 6.0; hot[:, :, 4, 4] = x[:, :, 4, 4]        # surround hot, centre unchanged
    quiet = x.clone(); quiet[:, :, 3:6, 3:6] = -6.0; quiet[:, :, 4, 4] = x[:, :, 4, 4]  # surround quiet, centre unchanged
    assert (m(hot)[:, :, 4, 4] < m(quiet)[:, :, 4, 4]).all()
    m.n_raw.data.fill_(-20.0)                              # n -> ~0 => identity
    assert torch.allclose(m(x), x, atol=1e-6)
    m.n_raw.data.fill_(0.0)
    m(x).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    assert sum(p.numel() for p in m.parameters()) == 92
    print("divnorm demo OK")


if __name__ == "__main__":
    demo()
