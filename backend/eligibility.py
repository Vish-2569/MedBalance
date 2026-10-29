"""Eligibility filter: determines which lots can move to which facilities
based on expiry, cold-chain, and transit constraints. Creates binary
decision variables x(k,d) — one per valid (surplus_chunk, destination) pair."""

from __future__ import annotations

import logging
from math import floor

from .schemas import (
    Drug,
    InventoryData,
    Route,
    SurplusChunk,
    TransferVariable,
)

logger = logging.getLogger(__name__)

BUFFER_DAYS = 3


def check_drug_identity(chunk: SurplusChunk, destination_drug_code: str) -> bool:
    """Chunk's canonical drug code must exactly match destination's needed drug code."""
    return chunk.drug_code == destination_drug_code


def check_route_exists(
    source_id: str, destination_id: str, route_map: dict[str, Route]
) -> Route | None:
    """Return the route if it exists in the permitted routes table, None otherwise."""
    return route_map.get(f"{source_id}|{destination_id}")


def check_cold_chain(drug: Drug, route: Route) -> bool:
    """If the drug requires cold chain, the route must have cold transport."""
    if not drug.requires_cold_chain:
        return True
    return route.has_cold_transport


def check_usable_before_expiry(
    chunk: SurplusChunk,
    destination_id: str,
    usage_rates: dict[str, "UsageRate"],
    route: Route,
    buffer_days: int = BUFFER_DAYS,
) -> int:
    """Compute usable quantity at destination before expiry.

    usable = min(chunk.quantity, floor(usage_rate * effective_days))
    where effective_days = days_to_expiry - transit_days - buffer
    Returns 0 if the chunk can't be used in time.
    """
    key = f"{destination_id}|{chunk.drug_code}"
    rate = usage_rates.get(key)
    if rate is None or rate.units_per_day <= 0:
        return 0

    effective_days = chunk.days_to_expiry - route.transit_days - buffer_days
    if effective_days <= 0:
        return 0

    usable = floor(rate.units_per_day * effective_days)
    return min(chunk.quantity, usable)


def check_reserve_protection(chunk: SurplusChunk) -> bool:
    """Verify the chunk quantity is positive — surplus computation in inventory.py
    already guarantees this chunk doesn't dip below reserve."""
    return chunk.quantity > 0


def create_transfer_variables(
    surplus_chunks: list[SurplusChunk],
    data: InventoryData,
) -> list[TransferVariable]:
    """Build the list of valid TransferVariables by applying all five hard rules.

    For each surplus chunk × each facility that needs that drug:
      1. Drug identity must match
      2. Route must exist from source to destination
      3. Cold-chain must be satisfied
      4. Usable quantity before expiry must be > 0
      5. Reserve protection (chunk quantity > 0)

    Returns only variables where ALL five rules pass.
    """
    variables: list[TransferVariable] = []

    total_candidates = 0
    filtered_self = 0
    filtered_no_route = 0
    filtered_cold_chain = 0
    filtered_expiry = 0
    filtered_reserve = 0

    needs_by_drug: dict[str, list[str]] = {}
    for key in data.needs:
        fac_id, drug_code = key.split("|", 1)
        needs_by_drug.setdefault(drug_code, []).append(fac_id)

    for chunk in surplus_chunks:
        drug = data.drugs.get(chunk.drug_code)
        if drug is None:
            continue

        needing_facilities = needs_by_drug.get(chunk.drug_code, [])

        for dest_id in needing_facilities:
            if dest_id == chunk.source_facility_id:
                filtered_self += 1
                continue

            total_candidates += 1

            # Drug identity is guaranteed by the needs-dict keying: we look up
            # needs[chunk.drug_code] so dest always needs the same drug_code.
            # check_drug_identity exists for callers that match across different codes.

            route = check_route_exists(chunk.source_facility_id, dest_id, data.route_map)
            if route is None:
                filtered_no_route += 1
                continue

            if not check_cold_chain(drug, route):
                filtered_cold_chain += 1
                continue

            usable = check_usable_before_expiry(
                chunk, dest_id, data.usage_rates, route
            )
            if usable <= 0:
                filtered_expiry += 1
                continue

            if not check_reserve_protection(chunk):
                filtered_reserve += 1
                continue

            var_id = f"x_{chunk.chunk_index}_{dest_id}"
            variables.append(TransferVariable(
                var_id=var_id,
                chunk_index=chunk.chunk_index,
                lot_id=chunk.lot_id,
                source_id=chunk.source_facility_id,
                destination_id=dest_id,
                drug_code=chunk.drug_code,
                chunk_quantity=chunk.quantity,
                usable_quantity=usable,
                days_to_expiry=chunk.days_to_expiry,
                transit_days=route.transit_days,
                value_at_risk=chunk.value_at_risk,
                transport_cost=route.trip_cost_inr,
            ))

    created = len(variables)
    logger.info(
        "Eligibility: %d candidate pairs -> "
        "%d no-route filtered -> "
        "%d cold-chain filtered -> "
        "%d usable-before-expiry filtered -> "
        "%d reserve filtered -> "
        "%d variables created "
        "(%d self-transfers skipped)",
        total_candidates + filtered_self,
        filtered_no_route,
        filtered_cold_chain,
        filtered_expiry,
        filtered_reserve,
        created,
        filtered_self,
    )

    return variables
