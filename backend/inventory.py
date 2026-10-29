"""Inventory engine: loads CSV data, computes facility stock, shortage needs,
surplus chunks (FEFO-ordered), usable quantities, and value-at-risk."""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from math import floor
from pathlib import Path

from .schemas import (
    Drug,
    DrugTier,
    Facility,
    FacilityType,
    InventoryData,
    InventorySnapshot,
    Lot,
    Reserve,
    Route,
    StorageCondition,
    SurplusChunk,
    UsageRate,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

COVER_DAYS_DEFAULT = 14
BUFFER_DAYS_DEFAULT = 3


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def _bool(val: str) -> bool:
    return val.strip().lower() in ("true", "1", "yes")


def load_facilities(path: Path | None = None) -> dict[str, Facility]:
    path = path or DATA_DIR / "facilities.csv"
    out: dict[str, Facility] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            fac = Facility(
                id=row["id"],
                name=row["name"],
                type=FacilityType(row["type"]),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                has_cold_storage=_bool(row["has_cold_storage"]),
            )
            out[fac.id] = fac
    return out


def load_drugs(path: Path | None = None) -> dict[str, Drug]:
    path = path or DATA_DIR / "drugs.csv"
    out: dict[str, Drug] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            drug = Drug(
                code=row["code"],
                name=row["name"],
                molecule=row["molecule"],
                strength=row["strength"],
                form=row["form"],
                tier=DrugTier(row["tier"]),
                criticality=int(row["criticality"]),
                requires_cold_chain=_bool(row["requires_cold_chain"]),
                unit_price_inr=float(row["unit_price_inr"]),
            )
            out[drug.code] = drug
    return out


def load_lots(path: Path | None = None) -> list[Lot]:
    path = path or DATA_DIR / "lots.csv"
    out: list[Lot] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append(Lot(
                lot_id=row["lot_id"],
                facility_id=row["facility_id"],
                drug_code=row["drug_code"],
                batch_number=row["batch_number"],
                quantity=int(row["quantity"]),
                days_to_expiry=int(row["days_to_expiry"]),
                storage_condition=StorageCondition(row["storage_condition"]),
            ))
    return out


def load_routes(path: Path | None = None) -> list[Route]:
    path = path or DATA_DIR / "routes.csv"
    out: list[Route] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append(Route(
                source_id=row["source_id"],
                destination_id=row["destination_id"],
                transit_days=int(row["transit_days"]),
                trip_cost_inr=float(row["trip_cost_inr"]),
                has_cold_transport=_bool(row["has_cold_transport"]),
            ))
    return out


def load_usage_rates(path: Path | None = None) -> dict[str, UsageRate]:
    path = path or DATA_DIR / "usage_rates.csv"
    out: dict[str, UsageRate] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ur = UsageRate(
                facility_id=row["facility_id"],
                drug_code=row["drug_code"],
                units_per_day=float(row["units_per_day"]),
            )
            out[f"{ur.facility_id}|{ur.drug_code}"] = ur
    return out


def load_reserves(path: Path | None = None) -> dict[str, Reserve]:
    path = path or DATA_DIR / "reserves.csv"
    out: dict[str, Reserve] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            r = Reserve(
                facility_id=row["facility_id"],
                drug_code=row["drug_code"],
                min_stock=int(row["min_stock"]),
            )
            out[f"{r.facility_id}|{r.drug_code}"] = r
    return out


# ---------------------------------------------------------------------------
# load_data — single entry point for all CSV data
# ---------------------------------------------------------------------------

def load_data(data_dir: Path | None = None) -> dict:
    """Load and parse all CSV files into Pydantic models.

    Returns a dict with keys: facilities, drugs, lots, routes,
    usage_rates, reserves.
    """
    d = data_dir or DATA_DIR
    return {
        "facilities": load_facilities(d / "facilities.csv"),
        "drugs": load_drugs(d / "drugs.csv"),
        "lots": load_lots(d / "lots.csv"),
        "routes": load_routes(d / "routes.csv"),
        "usage_rates": load_usage_rates(d / "usage_rates.csv"),
        "reserves": load_reserves(d / "reserves.csv"),
    }


# ---------------------------------------------------------------------------
# Stock helpers
# ---------------------------------------------------------------------------

def build_route_map(routes: list[Route]) -> dict[str, Route]:
    return {f"{r.source_id}|{r.destination_id}": r for r in routes}


def compute_facility_stock(
    lots: list[Lot],
) -> dict[str, int]:
    """Sum lot quantities per (facility_id, drug_code).

    Key format: "facility_id|drug_code".
    """
    stock: dict[str, int] = defaultdict(int)
    for lot in lots:
        stock[f"{lot.facility_id}|{lot.drug_code}"] += lot.quantity
    return dict(stock)


def _get_stock(
    facility_id: str,
    drug_code: str,
    lots: list[Lot],
) -> int:
    """Total stock of a drug at a facility."""
    return sum(
        lot.quantity for lot in lots
        if lot.facility_id == facility_id and lot.drug_code == drug_code
    )


def _get_usage_rate(
    facility_id: str,
    drug_code: str,
    usage_rates: dict[str, UsageRate],
) -> float:
    key = f"{facility_id}|{drug_code}"
    rate = usage_rates.get(key)
    return rate.units_per_day if rate else 0.0


# ---------------------------------------------------------------------------
# Need computation
# ---------------------------------------------------------------------------

def compute_need(
    facility_id: str,
    drug_code: str,
    usage_rates: dict[str, UsageRate],
    lots: list[Lot],
    cover_days: int = COVER_DAYS_DEFAULT,
) -> int:
    """Shortage at a facility: units needed to reach cover_days of stock.

    n(d,g) = max(0, cover_days × u(d,g) - stock(d,g))
    """
    rate = _get_usage_rate(facility_id, drug_code, usage_rates)
    stock = _get_stock(facility_id, drug_code, lots)
    target = floor(cover_days * rate)
    return max(0, target - stock)


# ---------------------------------------------------------------------------
# Surplus computation
# ---------------------------------------------------------------------------

def _compute_lot_waste(
    facility_lots: list[Lot],
    usage_rate: float,
) -> dict[str, int]:
    """Simulate FEFO consumption to find per-lot waste (units not consumed
    before expiry).

    Returns {lot_id: waste_qty}.
    """
    fefo = sorted(facility_lots, key=lambda l: (l.days_to_expiry, l.lot_id))
    consumed_so_far = 0
    waste: dict[str, int] = {}
    for lot in fefo:
        if lot.days_to_expiry <= 0:
            waste[lot.lot_id] = lot.quantity
            continue
        total_consumed_by_expiry = floor(usage_rate * lot.days_to_expiry)
        can_consume_from_lot = min(
            lot.quantity,
            max(0, total_consumed_by_expiry - consumed_so_far),
        )
        waste[lot.lot_id] = lot.quantity - can_consume_from_lot
        consumed_so_far += can_consume_from_lot
    return waste


def compute_surplus_lots(
    facility_id: str,
    drug_code: str,
    lots: list[Lot],
    reserves: dict[str, Reserve],
    usage_rates: dict[str, UsageRate],
    drugs: dict[str, Drug],
) -> list[SurplusChunk]:
    """Build FEFO-ordered surplus chunks for one (facility, drug) pair.

    Surplus = total stock above reserve. Chunks are carved from lots in
    FEFO order (earliest expiry first). value_at_risk reflects units the
    source would NOT consume before expiry.
    """
    facility_lots = [
        lot for lot in lots
        if lot.facility_id == facility_id and lot.drug_code == drug_code
    ]
    if not facility_lots:
        return []

    total_stock = sum(lot.quantity for lot in facility_lots)
    reserve_key = f"{facility_id}|{drug_code}"
    reserve = reserves.get(reserve_key)
    min_stock = reserve.min_stock if reserve else 0
    surplus_total = max(0, total_stock - min_stock)
    if surplus_total == 0:
        return []

    drug = drugs.get(drug_code)
    unit_price = drug.unit_price_inr if drug else 0.0
    usage_rate = _get_usage_rate(facility_id, drug_code, usage_rates)

    lot_waste = _compute_lot_waste(facility_lots, usage_rate)

    fefo = sorted(facility_lots, key=lambda l: (l.days_to_expiry, l.lot_id))

    chunks: list[SurplusChunk] = []
    surplus_remaining = surplus_total
    for lot in fefo:
        if lot.days_to_expiry <= 0:
            continue
        if surplus_remaining <= 0:
            break
        qty = min(lot.quantity, surplus_remaining)
        surplus_remaining -= qty

        at_risk_qty = min(qty, lot_waste.get(lot.lot_id, 0))

        chunks.append(SurplusChunk(
            chunk_id="",
            chunk_index=0,
            lot_id=lot.lot_id,
            source_facility_id=facility_id,
            drug_code=drug_code,
            quantity=qty,
            days_to_expiry=lot.days_to_expiry,
            storage_condition=lot.storage_condition.value,
            batch_number=lot.batch_number,
            value_at_risk=at_risk_qty * unit_price,
        ))

    return chunks


# ---------------------------------------------------------------------------
# Usable quantity
# ---------------------------------------------------------------------------

def compute_usable_quantity(
    chunk: SurplusChunk,
    destination_facility_id: str,
    usage_rates: dict[str, UsageRate],
    routes: list[Route] | dict[str, Route],
    buffer_days: int = BUFFER_DAYS_DEFAULT,
) -> int:
    """Usable quantity of a chunk at a destination.

    usable(k,d) = min(chunk.qty, floor(u(d,g) × (expiry - transit - buffer)))
    Returns 0 if the transfer would waste medicine.
    """
    if isinstance(routes, list):
        route_map = build_route_map(routes)
    else:
        route_map = routes

    route_key = f"{chunk.source_facility_id}|{destination_facility_id}"
    route = route_map.get(route_key)
    if route is None:
        return 0

    rate = _get_usage_rate(
        destination_facility_id, chunk.drug_code, usage_rates
    )
    if rate <= 0:
        return 0

    effective_days = chunk.days_to_expiry - route.transit_days - buffer_days
    if effective_days <= 0:
        return 0

    return min(chunk.quantity, floor(rate * effective_days))


# ---------------------------------------------------------------------------
# Inventory snapshot
# ---------------------------------------------------------------------------

def build_inventory_snapshot(
    data: dict,
    cover_days: int = COVER_DAYS_DEFAULT,
) -> InventorySnapshot:
    """Build a complete inventory snapshot: needs, surplus chunks, and stock.

    Args:
        data: dict from load_data() with keys facilities, drugs, lots,
              routes, usage_rates, reserves.
        cover_days: target stock coverage in days for need calculation.
    """
    facilities: dict[str, Facility] = data["facilities"]
    drugs: dict[str, Drug] = data["drugs"]
    lots: list[Lot] = data["lots"]
    usage_rates: dict[str, UsageRate] = data["usage_rates"]
    reserves: dict[str, Reserve] = data["reserves"]

    facility_stock = compute_facility_stock(lots)

    needs: dict[str, int] = {}
    for fac_id in facilities:
        for drug_code in drugs:
            n = compute_need(fac_id, drug_code, usage_rates, lots, cover_days)
            if n > 0:
                needs[f"{fac_id}|{drug_code}"] = n

    all_chunks: list[SurplusChunk] = []
    for fac_id in facilities:
        for drug_code in drugs:
            chunks = compute_surplus_lots(
                fac_id, drug_code, lots, reserves, usage_rates, drugs
            )
            all_chunks.extend(chunks)

    all_chunks.sort(key=lambda c: (c.days_to_expiry, c.lot_id))
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_id = f"CK-{i + 1:03d}"
        chunk.chunk_index = i

    logger.info(
        "Inventory snapshot: %d surplus chunks, %d facility-drug needs, "
        "%d stock entries",
        len(all_chunks),
        len(needs),
        len(facility_stock),
    )

    return InventorySnapshot(
        needs=needs,
        surplus_chunks=all_chunks,
        facility_stock=facility_stock,
    )


# ---------------------------------------------------------------------------
# load_all — backward-compatible entry point for eligibility.py
# ---------------------------------------------------------------------------

def load_all(
    data_dir: Path | None = None,
    cover_days: int = COVER_DAYS_DEFAULT,
) -> tuple[list[SurplusChunk], InventoryData]:
    """Load all data and build surplus chunks + InventoryData for eligibility.

    This is the bridge between the v2 inventory engine and the eligibility
    filter which expects InventoryData with needs keyed by "fac|drug".
    """
    raw = load_data(data_dir)
    snapshot = build_inventory_snapshot(raw, cover_days)

    route_map = build_route_map(raw["routes"])

    inv = InventoryData(
        facilities=raw["facilities"],
        drugs=raw["drugs"],
        lots=raw["lots"],
        routes=raw["routes"],
        usage_rates=raw["usage_rates"],
        reserves=raw["reserves"],
        route_map=route_map,
        stock_by_facility_drug=snapshot.facility_stock,
        needs=snapshot.needs,
    )
    return snapshot.surplus_chunks, inv
