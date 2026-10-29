"""Regret Ledger: persists top-K feasible alternatives so every
redistribution decision is auditable and explainable.

When the solver returns multiple samples, the ledger decodes, validates,
deduplicates, and ranks them. Each entry records how it differs from the
chosen plan, making "show alternatives" queries trivial.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .qubo_builder import compute_objective_value, decode_bitstring
from .schemas import (
    InventoryData,
    InventorySnapshot,
    TransferVariable,
    ValidationResult,
)
from .validation import validate_plan

logger = logging.getLogger(__name__)

MAX_ALTERNATIVES = 5


@dataclass
class RegretEntry:
    plan_rank: int
    transfers: list[TransferVariable]
    objective_value: float
    delta_from_best: float
    is_feasible: bool
    blocking_constraints: list[str] = field(default_factory=list)
    diff_from_chosen: list[str] = field(default_factory=list)


def _transfer_set_key(transfers: list[TransferVariable]) -> frozenset[str]:
    """Canonical fingerprint for deduplication."""
    return frozenset(t.var_id for t in transfers)


class RegretLedger:
    """Stores ranked alternative plans for auditability."""

    def __init__(self) -> None:
        self.entries: list[RegretEntry] = []

    # ------------------------------------------------------------------
    # Core: record alternatives from solver top-K
    # ------------------------------------------------------------------

    def record_alternatives(
        self,
        top_k_bitstrings: list[dict[str, int]],
        Q: dict[tuple[str, str], float],
        variables: list[TransferVariable],
        inventory_snapshot: InventorySnapshot,
        data: InventoryData,
    ) -> list[RegretEntry]:
        """Decode, validate, deduplicate, and rank top-K solver samples.

        Keeps up to MAX_ALTERNATIVES feasible, unique plans.
        """
        candidates: list[tuple[float, list[TransferVariable], ValidationResult]] = []
        seen: set[frozenset[str]] = set()

        for bits in top_k_bitstrings:
            transfers = decode_bitstring(bits, variables)
            key = _transfer_set_key(transfers)
            if key in seen:
                continue
            seen.add(key)

            h = compute_objective_value(bits, Q)
            val = validate_plan(transfers, variables, inventory_snapshot, data)
            candidates.append((h, transfers, val))

        candidates.sort(key=lambda c: c[0])

        feasible = [(h, t, v) for h, t, v in candidates if v.is_valid]
        if not feasible:
            feasible = candidates[:MAX_ALTERNATIVES]

        best_h = feasible[0][0] if feasible else 0.0
        chosen_ids = (
            _transfer_set_key(feasible[0][1]) if feasible else frozenset()
        )

        entries: list[RegretEntry] = []
        for rank, (h, transfers, val) in enumerate(feasible[:MAX_ALTERNATIVES], start=1):
            current_ids = _transfer_set_key(transfers)
            diff = sorted(current_ids.symmetric_difference(chosen_ids))

            entry = RegretEntry(
                plan_rank=rank,
                transfers=transfers,
                objective_value=h,
                delta_from_best=h - best_h,
                is_feasible=val.is_valid,
                blocking_constraints=list(val.violations) if not val.is_valid else [],
                diff_from_chosen=diff if rank > 1 else [],
            )
            entries.append(entry)

        self.entries = entries
        logger.info(
            "Regret ledger: %d alternatives recorded (%d feasible)",
            len(entries), sum(1 for e in entries if e.is_feasible),
        )
        return entries

    # ------------------------------------------------------------------
    # Query: alternatives for a specific transfer
    # ------------------------------------------------------------------

    def get_alternatives_for_transfer(
        self,
        transfer: TransferVariable,
    ) -> list[dict]:
        """Find plans that include or exclude a given transfer.

        Returns a list of dicts describing what each alternative does
        differently for this transfer's lot/destination.
        """
        results: list[dict] = []

        for entry in self.entries:
            if entry.plan_rank == 1:
                continue

            alt_ids = {t.var_id for t in entry.transfers}
            in_chosen = transfer.var_id in (
                _transfer_set_key(self.entries[0].transfers)
                if self.entries else set()
            )
            in_alt = transfer.var_id in alt_ids

            if in_chosen == in_alt:
                same_lot_transfers = [
                    t for t in entry.transfers
                    if t.lot_id == transfer.lot_id and t.var_id != transfer.var_id
                ]
                if not same_lot_transfers:
                    continue
                for t in same_lot_transfers:
                    results.append({
                        "rank": entry.plan_rank,
                        "action": "redirected",
                        "detail": (
                            f"Rank {entry.plan_rank}: lot {t.lot_id} sent to "
                            f"{t.destination_id} instead (delta={entry.delta_from_best:+.3f})"
                        ),
                        "destination": t.destination_id,
                        "delta": entry.delta_from_best,
                    })
            elif in_chosen and not in_alt:
                results.append({
                    "rank": entry.plan_rank,
                    "action": "dropped",
                    "detail": (
                        f"Rank {entry.plan_rank}: this transfer not selected "
                        f"(delta={entry.delta_from_best:+.3f})"
                    ),
                    "destination": None,
                    "delta": entry.delta_from_best,
                })
            else:
                results.append({
                    "rank": entry.plan_rank,
                    "action": "added",
                    "detail": (
                        f"Rank {entry.plan_rank}: this transfer added "
                        f"(delta={entry.delta_from_best:+.3f})"
                    ),
                    "destination": transfer.destination_id,
                    "delta": entry.delta_from_best,
                })

        return results

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialize the full ledger to JSON."""
        data = []
        for e in self.entries:
            d = {
                "plan_rank": e.plan_rank,
                "transfer_var_ids": [t.var_id for t in e.transfers],
                "objective_value": e.objective_value,
                "delta_from_best": e.delta_from_best,
                "is_feasible": e.is_feasible,
                "blocking_constraints": e.blocking_constraints,
                "diff_from_chosen": e.diff_from_chosen,
            }
            data.append(d)
        return json.dumps(data, indent=2)

    def summary(self) -> str:
        """Human-readable summary of the ledger."""
        if not self.entries:
            return "Regret ledger: empty (no alternatives recorded)."

        lines: list[str] = []
        for e in self.entries:
            n = len(e.transfers)
            if e.plan_rank == 1:
                lines.append(
                    f"Chosen plan (rank 1): {n} transfers, H={e.objective_value:.3f}."
                )
            else:
                feasible_tag = "" if e.is_feasible else " [INFEASIBLE]"
                diff_count = len(e.diff_from_chosen)
                lines.append(
                    f"Alternative {e.plan_rank}: differs in {diff_count} transfers, "
                    f"H={e.objective_value:.3f} (delta={e.delta_from_best:+.3f}).{feasible_tag}"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def persist_ledger(
    ledger: RegretLedger,
    scenario_name: str,
    evidence_dir: str = "evidence/",
) -> Path:
    """Save the ledger JSON to evidence/opt_runs/{scenario}/regret_ledger.json."""
    out_dir = Path(evidence_dir) / "opt_runs" / scenario_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "regret_ledger.json"
    path.write_text(ledger.to_json())
    logger.info("Regret ledger persisted: %s", path)
    return path
