"""Named architectural sites for YOLO26n (built from yolo26.yaml, scale 'n').

Layer indices refer to model.model[idx] after YOLO("yolo26n.yaml") is built.
YOLO26 is end-to-end (NMS-free) with reg_max=1 (no DFL head).

Depth tags:
  early = backbone stem + first C3k2 stages (high-res, low-level features)
  mid   = deep backbone + SPPF + C2PSA (semantic aggregation, attention)
  late  = FPN head C3k2 blocks + Detect (multi-scale fusion, prediction)

This list enumerates where the DEFAULT injection path (module wrap/replace)
is wired. It is not a search-space boundary: the agent may propose training
objectives, connectivity changes, or interventions outside this list; those
go through mechanism_spec kinds other than "site_wrap" (see adapter.py).
"""
from __future__ import annotations

from core.adapter import Site

# (name, depth, model.model index) — single source for SITES and SITE_INDEX.
_DEFS: list[tuple[str, str, int]] = [
    # backbone
    ("backbone.c3k2_p2", "early", 2),    # 256ch stage
    ("backbone.c3k2_p3", "early", 4),    # 512ch stage
    ("backbone.c3k2_p4", "mid", 6),
    ("backbone.c3k2_p5", "mid", 8),
    ("backbone.sppf", "mid", 9),
    ("backbone.c2psa", "mid", 10),       # attention block
    # head
    ("head.c3k2_fuse_p4", "late", 13),
    ("head.c3k2_out_p3", "late", 16),    # small objects
    ("head.c3k2_out_p4", "late", 19),    # medium objects
    ("head.c3k2_out_p5", "late", 22),    # large objects
    ("head.detect", "late", 23),         # NMS-free head
]

SITES: list[Site] = [{"name": n, "depth": d} for n, d, _ in _DEFS]

# name -> model.model index
SITE_INDEX: dict[str, int] = {n: i for n, _, i in _DEFS}
