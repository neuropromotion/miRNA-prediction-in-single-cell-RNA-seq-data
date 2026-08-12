"""Optimizer recipes for tabular DL: AdamW, AdamW+EMA, Muon+AdamW.

Follows the practical grouping from yandex-research/tabular-dl-optimizers:
Muon on 2D hidden weights; AdamW on biases / non-2D / output head.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn


def split_muon_adamw_params(
    model: nn.Module,
    *,
    exclude_name_substrings: tuple[str, ...] = ("output", "head", "bias"),
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return (muon_params, adamw_params).

    Muon: ndim==2 and name does not look like output/head/bias.
    Everything else (incl. 1D, embeddings, output matrices) → AdamW.
    """
    muon: list[nn.Parameter] = []
    adamw: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        exclude = any(s in lname for s in exclude_name_substrings)
        if p.ndim == 2 and not exclude:
            muon.append(p)
        else:
            adamw.append(p)
    if not muon:
        # Fallback: all 2D to Muon if naming heuristic excluded everything.
        for p in model.parameters():
            if p.requires_grad and p.ndim == 2:
                muon.append(p)
        adamw = [p for p in model.parameters() if p.requires_grad and p not in set(muon)]
    return muon, adamw


class DualOptimizer:
    """Thin wrapper so train loops can call zero_grad/step once."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = [o for o in optimizers if o is not None]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups


def build_optimizer(
    model: nn.Module,
    *,
    kind: str = "adamw",
    lr: float = 2e-3,
    weight_decay: float = 3e-4,
    muon_lr: float | None = None,
) -> DualOptimizer | torch.optim.Optimizer:
    """kind: 'adamw' | 'adamw_ema' | 'muon'.

    adamw_ema uses the same AdamW optimizer; EMA is handled by the train loop.
    """
    kind = kind.lower().strip()
    if kind in ("adamw", "adamw_ema"):
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if kind not in ("muon", "muon_adamw"):
        raise ValueError(f"Unknown optimizer kind: {kind}")

    muon_params, adamw_params = split_muon_adamw_params(model)
    m_lr = float(muon_lr if muon_lr is not None else max(lr, 0.02))

    if hasattr(torch.optim, "Muon"):
        muon_cls = torch.optim.Muon
    else:
        from muon_opt import Muon as muon_cls  # shared/ on sys.path

    opts: list[torch.optim.Optimizer] = []
    if muon_params:
        opts.append(
            muon_cls(
                muon_params,
                lr=m_lr,
                weight_decay=weight_decay,
                momentum=0.95,
            )
        )
    if adamw_params:
        opts.append(torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay))
    if not opts:
        raise RuntimeError("No trainable parameters for Muon/AdamW")
    return DualOptimizer(opts) if len(opts) > 1 else opts[0]


def make_ema(model: nn.Module, decay: float = 0.99):
    """PyTorch AveragedModel EMA (tabular-dl-optimizers recipe)."""
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))


def ema_state_dict(ema_model) -> dict[str, Any]:
    """Extract underlying module state from AveragedModel."""
    mod = getattr(ema_model, "module", ema_model)
    return {k: v.detach().cpu().clone() for k, v in mod.state_dict().items()}
