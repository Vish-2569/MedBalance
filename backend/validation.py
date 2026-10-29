"""Post-solve validation: independently re-checks every decoded transfer plan
against physical constraints, cold-chain rules, reserve floors, and
allocation limits. This is the second line of defense — no plan reaches an
officer unless it passes every check here."""

from __future__ import annotations

import logging
from collections import defaultdict
from math import floor

from .schemas import (
    InventoryData,
    InventorySnapshot,
    Route,
    TransferVariable,
    ValidationResult,
)

logger = logging.getLogger(__name__)

BUFFER_DAYS = 3


def validate_plan(
    selected_transfers: list[TransferVariable],
    all_variables: list[TransferVariable],
    inventory_snapshot: InventorySnapshot,
    data: InventoryData,
) -> ValidationResult:
    """Validate a decoded transfer plan against all hard constraints.

    Any single violation makes is_valid=False. FEFO deviations are warnings
    only — they flag sub-optimal ordering but do not block the plan.
    """
    violations: list[str] = []
    warnings: list[str] = []
    fefo_deviations: list[str] = []

    if not selected_transfers:
        return ValidationResult(
            is_valid=True, violations=[], warnings=["Empty plan — nothing to transfer"],
            fefo_deviations=[],
        )

    # (a) + (f): No duplicate lot assignment / no lot assigned to two destinations
    _check_duplicate_lots(selected_transfers, violations)

    # (b): No facility drawn below reserve
    _check_reserve_protection(selected_transfers, data, violations)

    # (c): No wrong drug match
    _check_drug_identity(selected_transfers, inventory_snapshot, violations)

    # (d): No cold-chain violation
    _check_cold_chain(selected_transfers, data, violations)

    # (e): No transfer arriving after expiry minus usable window
    _check_usable_before_expiry(selected_transfers, data, violations)

    # (g): No over-allocation beyond need + tolerance
    _check_over_allocation(selected_transfers, inventory_snapshot, violations)

    # (h): FEFO deviation warnings
    _check_fefo_deviations(
        selected_transfers, all_variables, fefo_deviations,
    )
    warnings.extend(fefo_deviations)

    is_valid = len(violations) == 0

    if is_valid:
        logger.info(
            "Plan validated: %d transfers, %d warnings",
            len(selected_transfers), len(warnings),
        )
    else:
        logger.warning(
            "Plan INVALID: %d violations, %d warnings",
            len(violations), len(warnings),
        )

    return ValidationResult(
        is_valid=is_valid,
        violations=violations,
        warnings=warnings,
        fefo_deviations=fefo_deviations,
    )


def _check_duplicate_lots(
    transfers: list[TransferVariable],
    violations: list[str],
) -> None:
    lot_destinations: dict[str, list[str]] = defaultdict(list)
    for t in transfers:
        lot_destinations[t.lot_id].append(t.destination_id)

    for lot_id, dests in lot_destinations.items():
        if len(dests) > 1:
            unique = sorted(set(dests))
            if len(unique) > 1:
                violations.append(
                    f"Lot {lot_id} assigned to both "
                    f"{unique[0]} and {unique[1]}"
                )
            else:
                violations.append(
                    f"Lot {lot_id} assigned to {unique[0]} "
                    f"{len(dests)} times (duplicate)"
                )


def _check_reserve_protection(
    transfers: list[TransferVariable],
    data: InventoryData,
    violations: list[str],
) -> None:
    outgoing: dict[str, int] = defaultdict(int)
    for t in transfers:
        outgoing[f"{t.source_id}|{t.drug_code}"] += t.chunk_quantity

    for key, sent in outgoing.items():
        fac_id, drug_code = key.split("|", 1)
        stock = data.stock_by_facility_drug.get(key, 0)
        reserve = data.reserves.get(key)
        min_stock = reserve.min_stock if reserve else 0
        remaining = stock - sent
        if remaining < min_stock:
            violations.append(
                f"Facility {fac_id} drawn below reserve for {drug_code}: "
                f"remaining {remaining} < reserve {min_stock}"
            )


def _check_drug_identity(
    transfers: list[TransferVariable],
    snapshot: InventorySnapshot,
    violations: list[str],
) -> None:
    for t in transfers:
        need_key = f"{t.destination_id}|{t.drug_code}"
        if need_key not in snapshot.needs:
            violations.append(
                f"Transfer {t.var_id}: destination {t.destination_id} "
                f"has no need for {t.drug_code}"
            )


def _check_cold_chain(
    transfers: list[TransferVariable],
    data: InventoryData,
    violations: list[str],
) -> None:
    for t in transfers:
        drug = data.drugs.get(t.drug_code)
        if drug is None:
            violations.append(f"Transfer {t.var_id}: unknown drug {t.drug_code}")
            continue
        if not drug.requires_cold_chain:
            continue
        route_key = f"{t.source_id}|{t.destination_id}"
        route = data.route_map.get(route_key)
        if route is None:
            violations.append(
                f"Transfer {t.var_id}: no route from "
                f"{t.source_id} to {t.destination_id}"
            )
            continue
        if not route.has_cold_transport:
            violations.append(
                f"Transfer {t.var_id}: cold-chain drug {t.drug_code} "
                f"on non-refrigerated route {t.source_id}->{t.destination_id}"
            )
        dest_facility = data.facilities.get(t.destination_id)
        if dest_facility and not dest_facility.has_cold_storage:
            violations.append(
                f"Transfer {t.var_id}: cold-chain drug {t.drug_code} "
                f"to facility {t.destination_id} which lacks cold storage"
            )


def _check_usable_before_expiry(
    transfers: list[TransferVariable],
    data: InventoryData,
    violations: list[str],
) -> None:
    for t in transfers:
        rate_obj = data.usage_rates.get(f"{t.destination_id}|{t.drug_code}")
        rate = rate_obj.units_per_day if rate_obj else 0.0
        route = data.route_map.get(f"{t.source_id}|{t.destination_id}")
        transit = route.transit_days if route else t.transit_days
        effective_days = t.days_to_expiry - transit - BUFFER_DAYS
        if effective_days <= 0:
            violations.append(
                f"Transfer {t.var_id}: expires before usable at destination "
                f"(expiry={t.days_to_expiry}d, transit={transit}d, "
                f"buffer={BUFFER_DAYS}d, effective={effective_days}d)"
            )
            continue
        usable = min(t.chunk_quantity, floor(rate * effective_days))
        if usable <= 0:
            violations.append(
                f"Transfer {t.var_id}: zero usable quantity at destination "
                f"(rate={rate}/day, effective_days={effective_days})"
            )


def _check_over_allocation(
    transfers: list[TransferVariable],
    snapshot: InventorySnapshot,
    violations: list[str],
) -> None:
    sent_to: dict[str, int] = defaultdict(int)
    for t in transfers:
        sent_to[f"{t.destination_id}|{t.drug_code}"] += t.usable_quantity

    for key, total_sent in sent_to.items():
        need = snapshot.needs.get(key, 0)
        tolerance = max(1, floor(0.05 * need)) if need > 0 else 1
        if total_sent > need + tolerance:
            dest_id, drug_code = key.split("|", 1)
            violations.append(
                f"Over-allocation to {dest_id} for {drug_code}: "
                f"sent {total_sent} but need is only {need}"
            )


def _check_fefo_deviations(
    selected: list[TransferVariable],
    all_variables: list[TransferVariable],
    deviations: list[str],
) -> None:
    selected_ids = {t.var_id for t in selected}

    by_dest_drug: dict[str, list[TransferVariable]] = defaultdict(list)
    for t in selected:
        by_dest_drug[f"{t.destination_id}|{t.drug_code}"].append(t)

    all_by_dest_drug: dict[str, list[TransferVariable]] = defaultdict(list)
    for v in all_variables:
        all_by_dest_drug[f"{v.destination_id}|{v.drug_code}"].append(v)

    for key, chosen in by_dest_drug.items():
        candidates = all_by_dest_drug.get(key, [])
        for sel in chosen:
            for cand in candidates:
                if cand.var_id in selected_ids:
                    continue
                if cand.days_to_expiry < sel.days_to_expiry:
                    deviations.append(
                        f"FEFO deviation at {sel.destination_id} for "
                        f"{sel.drug_code}: selected {sel.var_id} "
                        f"(expiry={sel.days_to_expiry}d) over unselected "
                        f"{cand.var_id} (expiry={cand.days_to_expiry}d)"
                    )


# ---------------------------------------------------------------------------
# Single-transfer validation
# ---------------------------------------------------------------------------

def validate_single_transfer(
    transfer: TransferVariable,
    data: InventoryData,
) -> ValidationResult:
    """Quick validation of one transfer against the hard rules."""
    violations: list[str] = []
    warnings: list[str] = []

    drug = data.drugs.get(transfer.drug_code)
    if drug is None:
        violations.append(f"Unknown drug code: {transfer.drug_code}")
        return ValidationResult(
            is_valid=False, violations=violations, warnings=warnings,
        )

    route_key = f"{transfer.source_id}|{transfer.destination_id}"
    route = data.route_map.get(route_key)
    if route is None:
        violations.append(
            f"No route from {transfer.source_id} to {transfer.destination_id}"
        )
    elif drug.requires_cold_chain and not route.has_cold_transport:
        violations.append(
            f"Cold-chain drug {transfer.drug_code} on non-refrigerated route"
        )

    if transfer.source_id == transfer.destination_id:
        violations.append("Source and destination are the same facility")

    if transfer.chunk_quantity <= 0:
        violations.append(f"Invalid chunk quantity: {transfer.chunk_quantity}")

    if route is not None:
        effective = transfer.days_to_expiry - route.transit_days - BUFFER_DAYS
        if effective <= 0:
            violations.append(
                f"Expires before usable (effective_days={effective})"
            )
        else:
            rate_obj = data.usage_rates.get(
                f"{transfer.destination_id}|{transfer.drug_code}"
            )
            rate = rate_obj.units_per_day if rate_obj else 0.0
            usable = min(transfer.chunk_quantity, floor(rate * effective))
            if usable <= 0:
                violations.append("Zero usable quantity at destination")

    stock_key = f"{transfer.source_id}|{transfer.drug_code}"
    stock = data.stock_by_facility_drug.get(stock_key, 0)
    reserve = data.reserves.get(stock_key)
    min_stock = reserve.min_stock if reserve else 0
    if stock - transfer.chunk_quantity < min_stock:
        violations.append(
            f"Would draw {transfer.source_id} below reserve for "
            f"{transfer.drug_code}"
        )

    return ValidationResult(
        is_valid=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Violation resolution
# ---------------------------------------------------------------------------

def resolve_violations(
    selected_transfers: list[TransferVariable],
    all_variables: list[TransferVariable],
    inventory_snapshot: InventorySnapshot,
    data: InventoryData,
    max_retries: int = 3,
) -> tuple[list[TransferVariable], ValidationResult]:
    """Attempt to fix a failing plan by removing violating transfers.

    Returns the cleaned plan and its final validation result.
    """
    plan = list(selected_transfers)

    for attempt in range(1, max_retries + 1):
        result = validate_plan(plan, all_variables, inventory_snapshot, data)
        if result.is_valid:
            if attempt > 1:
                removed = len(selected_transfers) - len(plan)
                logger.info(
                    "Plan cleaned on attempt %d: removed %d transfers",
                    attempt, removed,
                )
            return plan, result

        to_remove = _identify_violating_transfers(plan, result, data, inventory_snapshot)
        if not to_remove:
            logger.warning(
                "Cannot resolve violations: no removable transfers identified"
            )
            return plan, result

        remove_ids = {t.var_id for t in to_remove}
        logger.info(
            "Attempt %d: removing %d transfers: %s",
            attempt, len(to_remove), sorted(remove_ids),
        )
        plan = [t for t in plan if t.var_id not in remove_ids]

    result = validate_plan(plan, all_variables, inventory_snapshot, data)
    if not result.is_valid:
        logger.warning(
            "Plan still invalid after %d retries (%d violations remain)",
            max_retries, len(result.violations),
        )
    return plan, result


def _identify_violating_transfers(
    plan: list[TransferVariable],
    result: ValidationResult,
    data: InventoryData,
    snapshot: InventorySnapshot,
) -> list[TransferVariable]:
    """Identify the minimal set of transfers to remove to fix violations."""
    to_remove: list[TransferVariable] = []
    remove_ids: set[str] = set()

    # Duplicate lots: keep the one with higher usable_quantity
    lot_groups: dict[str, list[TransferVariable]] = defaultdict(list)
    for t in plan:
        lot_groups[t.lot_id].append(t)
    for lot_id, group in lot_groups.items():
        if len(group) > 1:
            group.sort(key=lambda t: t.usable_quantity, reverse=True)
            for t in group[1:]:
                if t.var_id not in remove_ids:
                    to_remove.append(t)
                    remove_ids.add(t.var_id)

    # Over-allocation: remove lowest-value transfers until under the cap
    sent_to: dict[str, list[TransferVariable]] = defaultdict(list)
    for t in plan:
        if t.var_id not in remove_ids:
            sent_to[f"{t.destination_id}|{t.drug_code}"].append(t)
    for key, transfers in sent_to.items():
        need = snapshot.needs.get(key, 0)
        tolerance = max(1, floor(0.05 * need)) if need > 0 else 1
        total = sum(t.usable_quantity for t in transfers)
        if total > need + tolerance:
            transfers.sort(key=lambda t: t.value_at_risk)
            for t in transfers:
                if total <= need + tolerance:
                    break
                if t.var_id not in remove_ids:
                    to_remove.append(t)
                    remove_ids.add(t.var_id)
                    total -= t.usable_quantity

    # Reserve violations: remove transfers draining each source most
    remaining: dict[str, int] = dict(data.stock_by_facility_drug)
    for t in plan:
        if t.var_id not in remove_ids:
            key = f"{t.source_id}|{t.drug_code}"
            remaining[key] = remaining.get(key, 0) - t.chunk_quantity
    for key, stock in remaining.items():
        fac_id, drug_code = key.split("|", 1)
        reserve = data.reserves.get(key)
        min_stock = reserve.min_stock if reserve else 0
        if stock < min_stock:
            fac_transfers = [
                t for t in plan
                if t.source_id == fac_id and t.drug_code == drug_code
                and t.var_id not in remove_ids
            ]
            fac_transfers.sort(key=lambda t: t.value_at_risk)
            deficit = min_stock - stock
            for t in fac_transfers:
                if deficit <= 0:
                    break
                to_remove.append(t)
                remove_ids.add(t.var_id)
                deficit -= t.chunk_quantity

    return to_remove
