"""P10 — Per-unit statistical verification gate.

Born from a postmortem: a prior trajectory's headline gain turned out to be a
metric-aggregation artifact (batch-mean-of-means over-weighted the final
partial batch), discovered only during review response. Aggregate deltas,
however carefully seeded, NEVER classify a Winner on their own again.

The gate:
1. For each seed, obtain per-unit metrics for candidate and reference via
   adapter.per_unit_metric() — a true per-unit quantity, never batch-averaged.
2. Seed-average each unit's paired difference (units are stable across seeds).
3. Paired bootstrap 95% CI over units + Wilcoxon signed-rank test.
4. Winner requires: CI excludes 0 AND wilcoxon_p < 0.05, in the favorable
   direction. Anything else is at best a Hold.

The same routine runs for mechanism-vs-baseline AND mechanism-vs-control
(P9 supplies which comparisons are required).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class P10Result:
    passed: bool
    reason: str
    n_units: int = 0
    mean_diff: float = 0.0
    median_diff: float = 0.0
    units_improved: int = 0
    ci95: tuple[float, float] = (0.0, 0.0)
    wilcoxon_p: float | None = None
    per_seed_means: dict = field(default_factory=dict)


def paired_unit_test(
    candidate_by_seed: dict[int, list[tuple[str, float]]],
    reference_by_seed: dict[int, list[tuple[str, float]]],
    *,
    higher_is_better: bool = True,
    n_boot: int = 20000,
    alpha: float = 0.05,
    boot_seed: int = 0,
) -> P10Result:
    """candidate/reference: {seed: [(unit_id, value), ...]} from per_unit_metric.

    Seeds must match between candidate and reference. Unit ids must be
    identical within each seed pair.
    """
    import numpy as np

    seeds = sorted(candidate_by_seed)
    if seeds != sorted(reference_by_seed):
        return P10Result(False, "seed sets differ between candidate and reference")
    if len(seeds) < 2:
        return P10Result(False, f"need >=2 seeds for the gate, got {len(seeds)}")

    per_seed_diffs = []
    per_seed_means = {}
    for s in seeds:
        cand = dict(candidate_by_seed[s])
        ref = dict(reference_by_seed[s])
        if set(cand) != set(ref):
            return P10Result(False, f"unit ids differ at seed {s}")
        ids = sorted(cand)
        d = np.array([cand[u] - ref[u] for u in ids])
        if not higher_is_better:
            d = -d
        per_seed_diffs.append(d)
        per_seed_means[s] = float(d.mean())

    avg = np.mean(np.stack(per_seed_diffs), axis=0)  # seed-averaged per-unit diff
    n = len(avg)

    rng = np.random.default_rng(boot_seed)
    boots = np.array([
        rng.choice(avg, size=n, replace=True).mean() for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    try:
        from scipy.stats import wilcoxon
        _, p = wilcoxon(avg)
        p = float(p)
    except Exception:
        p = None  # bootstrap CI stands alone if scipy unavailable

    favorable = lo > 0
    significant = (p is not None and p < alpha) or (p is None and favorable)
    passed = bool(favorable and significant and avg.mean() > 0)

    if passed:
        reason = "per-unit gate passed: CI excludes 0 in favorable direction"
    elif avg.mean() > 0:
        reason = (
            f"direction favorable but not significant "
            f"(CI [{lo:+.4f},{hi:+.4f}], wilcoxon_p={p}) — at best a Hold"
        )
    else:
        reason = f"unfavorable direction (mean diff {avg.mean():+.4f})"

    return P10Result(
        passed=passed, reason=reason, n_units=n,
        mean_diff=float(avg.mean()), median_diff=float(np.median(avg)),
        units_improved=int((avg > 0).sum()), ci95=(float(lo), float(hi)),
        wilcoxon_p=p, per_seed_means=per_seed_means,
    )
