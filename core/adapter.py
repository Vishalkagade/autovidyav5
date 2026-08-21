"""Adapter contract. Every model-specific plug-in implements this Protocol.

Design rules:
- The harness treats everything as opaque strings, opaque numbers, opaque tensors.
- Model-specific decisions (what counts as a "unit", what tolerance is meaningful,
  what an "architectural site" is) live in the adapter.
- The harness only counts, compares, and gates.

Regime note (bold / from-scratch):
- Models are trained FROM SCRATCH. There is no pretrained optimum to preserve,
  so identity-init is a diagnostic aid, not the attribution gate.
- Attribution comes from MATCHED CONTROLS (P9): every mechanism experiment is
  paired with a parameter-matched generic control trained identically. The
  mechanism-specific effect is (mechanism - control), not (mechanism - baseline).
- Verification comes from PER-UNIT STATISTICS (P10): paired bootstrap and
  rank tests over test units, seed-averaged. Aggregate deltas alone never
  classify a Winner.
"""
from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

from torch import Tensor

StageName = Literal["S1", "S2"]
DepthTag = Literal["early", "mid", "late"]
ControlFamily = Literal["param_matched_generic", "frozen_random"]


class Site(TypedDict):
    """A named architectural location where a mechanism can be injected.

    `name` is opaque to the harness — used only as a string key for matching
    and as the P5 axis grouping unit.
    `depth` is the only semantic tag the harness uses (for proposal balance hints).
    """

    name: str
    depth: DepthTag


class StageMetrics(TypedDict):
    """Return shape of `train_and_eval`."""

    primary: float           # the headline metric value
    secondary: dict[str, float]
    seed: int
    stage: StageName
    epochs_or_seconds: float  # actual training budget consumed
    curve: list[float]       # per-epoch primary-metric trajectory from the
                             # TRAINING-TIME val split (a small locked subset,
                             # not the full val set) — feeds the P12 screen
                             # audit. Empty list if the trainer produced no
                             # per-epoch record.


@runtime_checkable
class Adapter(Protocol):
    """The contract. Implementing class needs every attribute / method below."""

    # ─── identity ────────────────────────────────────────────────────────
    name: str
    """Adapter identifier, e.g. 'yolo26n_coco2k_scratch'."""

    primary_metric_name: str
    """e.g. 'mAP50', 'dice'. The harness uses this as a label."""

    primary_metric_higher_is_better: bool

    param_budget_fraction: float
    """Max fraction of baseline params a mechanism may add (bold regime: ~0.10).
    P9 rejects mechanism specs exceeding this before any GPU time is spent."""

    # ─── construction ────────────────────────────────────────────────────
    def build_baseline(self) -> Any:
        """Return a fresh RANDOMLY-INITIALIZED baseline model (no pretrained
        weights, no mechanism). Must be deterministic given torch seed."""
        ...

    def inject_mechanism(self, baseline: Any, mechanism_spec: dict) -> Any:
        """Return a model with the mechanism built in.

        `mechanism_spec` is an adapter-defined dict; the harness passes it
        through. In the from-scratch regime the mechanism does NOT need to be
        identity at init — it is part of the architecture from step 0. The
        adapter must record `param_count_added` in the returned model's
        `.autovidya_meta` dict for P9's budget check.
        """
        ...

    def build_control(self, baseline: Any, mechanism_spec: dict,
                      family: ControlFamily) -> Any:
        """Return the matched control for `mechanism_spec` (P9).

        `param_matched_generic`: a generic block (adapter-defined default,
            e.g. extra conv / MLP) at the SAME site with param count within
            ±10% of the mechanism's.
        `frozen_random`: the mechanism's own structure with parameters frozen
            at random init (tests whether learning the mechanism matters).

        Further families (e.g. capacity-matched widening of the site block)
        are defined per-mechanism at Winner-promotion time, not here.
        """
        ...

    # ─── training ────────────────────────────────────────────────────────
    def train_and_eval(self, model: Any, *, stage: StageName, seed: int) -> StageMetrics:
        """Train FROM SCRATCH to the stage's budget and return seed-keyed metrics.

        Budget is adapter-owned (epoch count or time cap). The harness fixes
        Tier-3 parameters (split, batch, image size) before calling.
        """
        ...

    # ─── diagnostics (used by disciplines) ───────────────────────────────
    def probe_batch(self) -> Tensor:
        """Small fixed batch for structural checks. Stable across runs."""
        ...

    def gradient_flow_check(self, model: Any, probe: Tensor) -> tuple[bool, float]:
        """One forward+backward on the probe: do the mechanism's parameters
        receive nonzero gradients? Returns (passed, grad_norm). Replaces the
        old identity-init check as the mandatory pre-Stage-1 gate in the
        from-scratch regime (P6')."""
        ...

    def sites(self) -> list[Site]:
        """Enumerate named injection sites. NOT a fixed search space — the
        agent may propose interventions elsewhere; see `describe_model()`."""
        ...

    def describe_model(self) -> dict:
        """Structured, non-prescriptive dump of the baseline architecture."""
        ...

    def per_unit_metric(self, model: Any, split: str = "test") -> list[tuple[str, float]]:
        """For each test unit: (opaque_unit_id, metric_value).

        THE LOAD-BEARING METHOD of the statistics gate (P10). `unit_id` MUST
        be stable across seeds and models so paired per-unit deltas are
        computable. The metric must be a true per-unit quantity — never a
        batch-aggregated value (see the aggregation-artifact postmortem in
        the protocol doc).
        """
        ...

    def per_site_stats(self, model: Any) -> dict[str, dict[str, float]]:
        """Per-site activation summary, keyed by site name."""
        ...

    def mechanism_activation_stats(self, model: Any) -> dict[str, float]:
        """Stats over the injected mechanism's output. Empty dict if no mechanism."""
        ...

    # ─── Phase-0 diagnostic methods (optional; may raise NotImplementedError) ──
    def per_class_metric(self, model: Any, split: str = "test") -> list[dict[str, Any]]:
        ...

    def per_scale_metric(self, model: Any, split: str = "test") -> dict[str, float]:
        ...

    def loss_decomposition(self, model: Any, split: str = "test") -> dict[str, float]:
        ...

    def top_failure_units(self, model: Any, k: int = 20, split: str = "test") -> list[dict[str, Any]]:
        ...

    def noise_floor(self, stage: StageName) -> float:
        """Calibrated primary-metric spread under same model + different seeds.
        In the from-scratch regime this MUST be measured with >= 3 seeds of the
        vanilla baseline (exp000) before Phase-1 starts."""
        ...
