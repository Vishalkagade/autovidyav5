"""P0 — pre-Phase-1 baseline diagnostic mandate.

Trigger:  Phase-1 transition. Fires once, before the first Phase-1 proposal.
Action:   block Phase-1 until `trajectory/profiles/baseline.json` exists with
          a non-empty `analysis.candidate_failure_modes`.

Rationale: without a structured look at WHERE the vanilla baseline already
fails, every Phase-1 proposal is a guess against an imagined menu. P0 forces
the agent to characterize the baseline before proposing interventions, so
proposals are evidence-driven instead of vocabulary-driven.

Two halves to the gate:
  1. `data` sections — adapter-produced. Run the diagnostic methods on the
     baseline; gracefully skip any that raise NotImplementedError, logging
     the gap.
  2. `analysis.candidate_failure_modes` — agent-produced. The agent reads the
     data and writes a list of falsifiable failure-mode hypotheses, each with
     a stable `fm_*` id. Phase-1 experiments cite these IDs.

P0 is the only discipline that produces a *required artifact* before Phase-1
begins. (P7 is the analog for Phase-2.)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..adapter import Adapter

GateStatus = Literal["proceed", "blocked"]


@dataclass
class P0Result:
    status: GateStatus
    profile_path: Path
    message: str
    gaps: list[str]


class P0_BaselineDiagnostic:
    """Build the baseline profile, or check that a usable one already exists.

    Usage::

        p0 = P0_BaselineDiagnostic(adapter, trajectory_dir)
        result = p0.run(baseline_model)  # produces data sections
        # ... agent writes analysis.candidate_failure_modes into the file ...
        result = p0.gate()              # checks the gate; status == 'proceed' if ready
        if result.status == "blocked":
            # halt Phase-1
            ...
    """

    PROFILE_NAME = "baseline.json"

    def __init__(self, adapter: Adapter, trajectory_dir: Path | str):
        if not isinstance(adapter, Adapter):
            raise TypeError(
                f"{type(adapter).__name__} does not satisfy the Adapter Protocol."
            )
        self.adapter = adapter
        self.profiles_dir = Path(trajectory_dir) / "profiles"
        self.profile_path = self.profiles_dir / self.PROFILE_NAME

    # ─── data half (adapter-produced) ────────────────────────────────────
    def run(self, baseline_model: Any, baseline_id: str | None = None) -> P0Result:
        """Run the adapter's diagnostic suite and write `data` to baseline.json.

        Does NOT write the `analysis` section — that's the agent's job after
        reading the data. Re-running overwrites the data section but preserves
        analysis if present.
        """
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {}
        gaps: list[str] = []

        # primary_metric is collected by the caller (P0 never retrains) and
        # merged into baseline.json alongside the agent's analysis.
        self._safe(data, "per_class_metric", gaps,
                   self.adapter.per_class_metric, baseline_model)
        self._safe(data, "per_scale_metric", gaps,
                   self.adapter.per_scale_metric, baseline_model)
        self._safe(data, "loss_decomposition", gaps,
                   self.adapter.loss_decomposition, baseline_model)
        self._safe(data, "top_failure_units", gaps,
                   lambda m: self.adapter.top_failure_units(m, k=20), baseline_model)
        self._safe(data, "model_topology_summary", gaps,
                   lambda _: self.adapter.describe_model(), baseline_model)

        bid = baseline_id or f"{self.adapter.name}_unpatched"
        payload = {
            "adapter_name": self.adapter.name,
            "baseline_id": bid,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data": data,
            "analysis": self._load_existing_analysis(),
        }

        # Preserve any existing analysis the agent wrote earlier.
        with self.profile_path.open("w") as f:
            json.dump(payload, f, indent=2, default=str)

        msg = (
            f"baseline.json written ({len(data)} data sections, "
            f"{len(gaps)} gap{'s' if len(gaps) != 1 else ''}). "
            f"Agent must populate analysis.candidate_failure_modes."
        )
        return P0Result(status="blocked", profile_path=self.profile_path,
                        message=msg, gaps=gaps)

    # ─── gate half (agent-produced; called before Phase-1) ───────────────
    def gate(self) -> P0Result:
        """Check whether Phase-1 may start.

        Pass condition: baseline.json exists AND
        analysis.candidate_failure_modes is a non-empty list.
        """
        if not self.profile_path.exists():
            return P0Result(
                status="blocked",
                profile_path=self.profile_path,
                message=(
                    f"baseline.json does not exist at {self.profile_path}. "
                    f"Run P0_BaselineDiagnostic.run(baseline) first."
                ),
                gaps=[],
            )

        try:
            with self.profile_path.open() as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            return P0Result(
                status="blocked",
                profile_path=self.profile_path,
                message=f"baseline.json unreadable: {e}",
                gaps=[],
            )

        analysis = payload.get("analysis") or {}
        cfm = analysis.get("candidate_failure_modes") or []
        if not isinstance(cfm, list) or len(cfm) == 0:
            return P0Result(
                status="blocked",
                profile_path=self.profile_path,
                message=(
                    "baseline.json exists but analysis.candidate_failure_modes "
                    "is empty. Agent must read the `data` sections and write at "
                    "least one falsifiable failure-mode hypothesis with an `fm_*` id."
                ),
                gaps=analysis.get("gaps", []),
            )

        # Sanity: each entry must have id + label + evidence
        for i, fm in enumerate(cfm):
            for key in ("id", "label", "evidence"):
                if key not in fm:
                    return P0Result(
                        status="blocked",
                        profile_path=self.profile_path,
                        message=(
                            f"candidate_failure_modes[{i}] missing '{key}'. "
                            f"Each entry needs id (fm_*), label, evidence."
                        ),
                        gaps=analysis.get("gaps", []),
                    )

        return P0Result(
            status="proceed",
            profile_path=self.profile_path,
            message=f"P0 gate passed: {len(cfm)} candidate failure modes recorded.",
            gaps=analysis.get("gaps", []),
        )

    # ─── helpers ─────────────────────────────────────────────────────────
    def _safe(self, data: dict, key: str, gaps: list,
              fn, *args) -> Any:
        try:
            data[key] = fn(*args)
            return data[key]
        except NotImplementedError as e:
            gaps.append(f"{key}: {e or 'NotImplementedError'}")
            return None
        except Exception as e:  # noqa: BLE001
            gaps.append(f"{key}: {type(e).__name__}: {e}")
            return None

    def _load_existing_analysis(self) -> dict[str, Any]:
        if not self.profile_path.exists():
            return {"candidate_failure_modes": []}
        try:
            with self.profile_path.open() as f:
                old = json.load(f)
            return old.get("analysis", {"candidate_failure_modes": []})
        except (json.JSONDecodeError, OSError):
            return {"candidate_failure_modes": []}
