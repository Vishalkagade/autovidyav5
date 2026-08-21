"""P9 — Matched-control attribution gate.

The from-scratch regime's replacement for identity-init as the attribution
instrument. A mechanism's effect is only attributable to its FUNCTIONAL FORM
if a parameter-matched generic control, trained identically, does not
reproduce it.

Rules enforced here:
1. Budget check (pre-GPU): mechanism param_count_added must not exceed
   adapter.param_budget_fraction of baseline params. Violations are rejected
   before any training.
2. Pairing check (pre-classification): a mechanism experiment cannot be
   classified Winner unless its `param_matched_generic` control has completed
   the same stages on the same seeds.
3. Attribution check: mechanism must beat BOTH the baseline AND its control
   by the classification margins. (mechanism - control) is the
   form-specific effect; (control - baseline) is the generic-capacity effect.
   Both are reported in the experiment JSON.

`frozen_random` is an optional deepening probe; further controls (e.g. a
capacity-matched widening of the site block) are designed per-mechanism when
promoting a Winner to a paper-level claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class P9Result:
    passed: bool
    reason: str
    param_count_added: int | None = None
    param_budget: int | None = None
    control_present: bool = False
    mech_minus_baseline: float | None = None
    mech_minus_control: float | None = None
    control_minus_baseline: float | None = None


def budget_check(adapter: Any, baseline: Any, mechanism_meta: dict) -> P9Result:
    """Rule 1. Call BEFORE spending any GPU time on the mechanism."""
    added = int(mechanism_meta.get("param_count_added", -1))
    if added < 0:
        return P9Result(False, "mechanism_meta missing param_count_added")
    base_params = sum(p.numel() for p in baseline.parameters())
    budget = int(base_params * adapter.param_budget_fraction)
    if added > budget:
        return P9Result(
            False,
            f"param budget exceeded: added {added} > budget {budget} "
            f"({adapter.param_budget_fraction:.0%} of {base_params})",
            param_count_added=added,
            param_budget=budget,
        )
    return P9Result(True, "within budget", param_count_added=added, param_budget=budget)


def attribution_check(
    *,
    mech_seedavg: float,
    control_seedavg: float | None,
    baseline_seedavg: float,
    noise_floor_s2: float,
    higher_is_better: bool = True,
) -> P9Result:
    """Rules 2+3. Call at classification time (after Stage-2 multi-seed).

    Winner requires:
      (mech - baseline) >= 2 * noise_floor  AND
      (mech - control)  >= 1 * noise_floor
    (signs flipped when lower is better).
    """
    sign = 1.0 if higher_is_better else -1.0
    if control_seedavg is None:
        return P9Result(
            False,
            "no completed param_matched_generic control — Winner classification blocked",
            control_present=False,
            mech_minus_baseline=sign * (mech_seedavg - baseline_seedavg),
        )
    mb = sign * (mech_seedavg - baseline_seedavg)
    mc = sign * (mech_seedavg - control_seedavg)
    cb = sign * (control_seedavg - baseline_seedavg)
    if mb < 2.0 * noise_floor_s2:
        reason = f"(mech-baseline)={mb:+.4f} < 2x noise floor {2*noise_floor_s2:.4f}"
        ok = False
    elif mc < 1.0 * noise_floor_s2:
        reason = (
            f"(mech-control)={mc:+.4f} < noise floor {noise_floor_s2:.4f} — "
            f"generic capacity explains the gain (control-baseline={cb:+.4f})"
        )
        ok = False
    else:
        reason = "attribution holds: mechanism beats baseline and matched control"
        ok = True
    return P9Result(
        ok, reason, control_present=True,
        mech_minus_baseline=mb, mech_minus_control=mc, control_minus_baseline=cb,
    )
