"""P11 — Sealed-seed replication gate.

A candidate that passes P9 (matched-control attribution) and P10 (per-unit
statistics) on the WORKING seed set is a Provisional Winner, nothing more.
P11 then runs the IDENTICAL config on the CONFIRMATION seed set — seeds chosen
before the trajectory started, never touched by any experiment, probe, or
debug run (exp000's baseline legs are the one exception, so the baseline side
of the comparison already exists when P11 fires).

Why this exists: a passed statistical test on the seeds a candidate was
developed against is still one observation. Repeating the effect on seeds the
candidate has never seen is the cheapest honest test of "is this real".

Verdicts (see the vocabulary table in program.md):
    pass -> Winner
    fail -> Fragile   (insight entry written; NO retry — one replication,
                       one verdict)

How to use (the agent calls this at P11 time):
    1. Train the mechanism on each confirmation seed (Stage-2).
    2. If `control_must_replicate(...)` says so, train the control on the
       confirmation seeds too.
    3. Run core.discipline.p10_stats_gate.paired_unit_test twice over ALL
       seeds combined (working + confirmation): mechanism-vs-baseline and
       mechanism-vs-control.
    4. Feed everything into `replication_check(...)` below and record the
       returned verdict in the experiment JSON's `p11` object.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .p10_stats_gate import P10Result

# ── The threshold program.md's P11 refers to ─────────────────────────────
# If the Provisional Winner's P9 margin (mechanism minus control, seed-avg,
# working set) is below this many S2 noise floors, the working-set control
# runs are too close to trust on their own: the control must ALSO be re-run
# on the confirmation seeds, and both P10 comparisons re-tested on the
# combined data. Value is surfaced in the adapter program.md.
CONTROL_REPLICATION_NOISE_MULTIPLE = 2.0


def control_must_replicate(mech_minus_control: float, noise_floor_s2: float) -> bool:
    """Decide BEFORE spending confirmation-seed GPU time on the control.

    mech_minus_control: seed-averaged P9 margin from the working set, already
    sign-normalized so that positive = favorable.
    """
    return mech_minus_control < CONTROL_REPLICATION_NOISE_MULTIPLE * noise_floor_s2


@dataclass
class P11Result:
    verdict: str                 # "Winner" or "Fragile" — nothing else
    reason: str
    per_seed_confirmation_deltas: dict = field(default_factory=dict)
    combined_vs_baseline_passed: bool = False
    combined_vs_control_passed: bool = False
    control_replicated: bool = False


def replication_check(
    *,
    per_seed_confirmation_deltas: dict[int, float],
    combined_vs_baseline: P10Result,
    combined_vs_control: P10Result,
    control_replicated: bool,
) -> P11Result:
    """The P11 verdict. Requires BOTH:

    1. every confirmation seed individually shows the effect in the favorable
       direction (deltas here are sign-normalized: positive = favorable;
       a seed at exactly 0 counts as going backwards), AND
    2. the per-unit paired test over ALL seeds combined still passes for both
       P10 comparisons (vs baseline and vs control).

    `per_seed_confirmation_deltas`: {seed: primary-metric delta vs baseline}
    for the confirmation seeds only.
    `control_replicated`: True if the control was re-run on the confirmation
    seeds (recorded in the JSON either way, so the record shows which data
    the combined vs-control test used).
    """
    if not per_seed_confirmation_deltas:
        return P11Result(
            verdict="Fragile",
            reason="no confirmation-seed runs supplied — P11 cannot pass on zero data",
        )

    backwards = {s: d for s, d in per_seed_confirmation_deltas.items() if d <= 0}
    if backwards:
        return P11Result(
            verdict="Fragile",
            reason=f"confirmation seed(s) went backwards or flat: {backwards}",
            per_seed_confirmation_deltas=dict(per_seed_confirmation_deltas),
            control_replicated=control_replicated,
        )

    if not combined_vs_baseline.passed:
        return P11Result(
            verdict="Fragile",
            reason=f"combined per-unit test vs BASELINE failed: {combined_vs_baseline.reason}",
            per_seed_confirmation_deltas=dict(per_seed_confirmation_deltas),
            combined_vs_baseline_passed=False,
            combined_vs_control_passed=combined_vs_control.passed,
            control_replicated=control_replicated,
        )

    if not combined_vs_control.passed:
        return P11Result(
            verdict="Fragile",
            reason=f"combined per-unit test vs CONTROL failed: {combined_vs_control.reason}",
            per_seed_confirmation_deltas=dict(per_seed_confirmation_deltas),
            combined_vs_baseline_passed=True,
            combined_vs_control_passed=False,
            control_replicated=control_replicated,
        )

    return P11Result(
        verdict="Winner",
        reason="replicated: every confirmation seed favorable, both combined per-unit tests pass",
        per_seed_confirmation_deltas=dict(per_seed_confirmation_deltas),
        combined_vs_baseline_passed=True,
        combined_vs_control_passed=True,
        control_replicated=control_replicated,
    )
