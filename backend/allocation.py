"""Greedy FEFO baseline allocator — the phone-coordination benchmark.

Simulates what happens when facilities informally coordinate: transfer
the most urgently expiring surplus to the facility with the greatest
need, one transfer at a time, greedily. The QpiAI quantum solution
will be compared against this baseline.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from .schemas import (
    InventoryData,
    InventorySnapshot,
    TransferVariable,
)
from .validation import validate_plan

logger = logging.getLogger(__name__)


def greedy_fefo_allocate(
    variables: list[TransferVariable],
    inventory_snapshot: InventorySnapshot,
    data: InventoryData,
) -> list[TransferVariable]:
    """Select transfers greedily by earliest-expiry, highest-criticality,
    highest value-at-risk.

    Returns the list of selected TransferVariables (x=1 decisions).
    The result is validated through validate_plan before returning.
    """
    drug_criticality: dict[str, int] = {
        code: drug.criticality for code, drug in data.drugs.items()
    }

    sorted_vars = sorted(
        variables,
        key=lambda v: (
            v.days_to_expiry,
            -drug_criticality.get(v.drug_code, 0),
            -v.value_at_risk,
        ),
    )

    assigned_lots: set[str] = set()
    remaining_need: dict[str, int] = dict(inventory_snapshot.needs)
    remaining_surplus: dict[str, int] = {}
    for key, stock in data.stock_by_facility_drug.items():
        reserve = data.reserves.get(key)
        min_stock = reserve.min_stock if reserve else 0
        remaining_surplus[key] = max(0, stock - min_stock)

    selected: list[TransferVariable] = []

    for v in sorted_vars:
        if v.lot_id in assigned_lots:
            continue

        need_key = f"{v.destination_id}|{v.drug_code}"
        if remaining_need.get(need_key, 0) <= 0:
            continue

        surplus_key = f"{v.source_id}|{v.drug_code}"
        if remaining_surplus.get(surplus_key, 0) < v.chunk_quantity:
            continue

        selected.append(v)
        assigned_lots.add(v.lot_id)
        remaining_need[need_key] = remaining_need.get(need_key, 0) - v.usable_quantity
        remaining_surplus[surplus_key] -= v.chunk_quantity

    result = validate_plan(selected, variables, inventory_snapshot, data)
    if not result.is_valid:
        logger.warning(
            "Greedy plan has %d violations: %s",
            len(result.violations), result.violations,
        )

    logger.info(
        "Greedy FEFO: selected %d of %d variables",
        len(selected), len(variables),
    )

    return selected


def evaluate_greedy(
    selected: list[TransferVariable],
    inventory_snapshot: InventorySnapshot,
) -> dict:
    """Compute summary metrics for a greedy allocation result."""
    total_units = sum(v.usable_quantity for v in selected)
    total_cost = sum(v.transport_cost for v in selected)
    total_var = sum(v.value_at_risk for v in selected)

    fulfilled: set[str] = set()
    for v in selected:
        fulfilled.add(f"{v.destination_id}|{v.drug_code}")

    return {
        "total_transfers": len(selected),
        "total_units_moved": total_units,
        "total_transport_cost": total_cost,
        "total_value_saved": total_var - total_cost,
        "shortages_addressed": len(fulfilled),
        "shortages_remaining": len(inventory_snapshot.needs) - len(fulfilled),
    }
