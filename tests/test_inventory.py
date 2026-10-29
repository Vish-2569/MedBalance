"""Tests for the v2 inventory engine with hand-calculated expected values.

Tiny dataset: 3 facilities, 2 drugs, 5 lots — small enough to verify
every number by hand.
"""

import pytest
from math import floor

from backend.schemas import (
    Drug, DrugTier, Facility, FacilityType, Lot, Reserve, Route,
    StorageCondition, SurplusChunk, UsageRate,
)
from backend.inventory import (
    compute_need, compute_surplus_lots, compute_usable_quantity,
    build_route_map,
)


# ---------------------------------------------------------------------------
# Helpers — tiny reusable builders
# ---------------------------------------------------------------------------

def _drug(code="d1", price=10.0, cold=False):
    return Drug(code=code, name="T", molecule="M", strength="1mg",
                form="tablet", tier=DrugTier.essential, criticality=2,
                requires_cold_chain=cold, unit_price_inr=price)


def _lot(lot_id="L1", fac="F1", drug="d1", qty=100, expiry=30):
    return Lot(lot_id=lot_id, facility_id=fac, drug_code=drug,
               batch_number="B1", quantity=qty, days_to_expiry=expiry,
               storage_condition=StorageCondition.room_temp)


def _usage(fac="F1", drug="d1", rate=10.0):
    return UsageRate(facility_id=fac, drug_code=drug, units_per_day=rate)


def _reserve(fac="F1", drug="d1", min_stock=50):
    return Reserve(facility_id=fac, drug_code=drug, min_stock=min_stock)


def _route(src="F1", dst="F2", transit=1, cost=500.0, cold=False):
    return Route(source_id=src, destination_id=dst, transit_days=transit,
                 trip_cost_inr=cost, has_cold_transport=cold)


def _chunk(qty=100, expiry=30, src="F1", drug="d1"):
    return SurplusChunk(chunk_id="CK-001", chunk_index=0, lot_id="L1",
                        source_facility_id=src, drug_code=drug,
                        quantity=qty, days_to_expiry=expiry,
                        storage_condition="room_temp",
                        batch_number="B1", value_at_risk=0.0)


# ---------------------------------------------------------------------------
# compute_need
# ---------------------------------------------------------------------------

class TestComputeNeed:
    def test_compute_need_with_shortage(self):
        """F2 has 10 units of d1, uses 2/day, cover=14.
        need = max(0, floor(14×2) - 10) = max(0, 28 - 10) = 18
        """
        rates = {"F2|d1": _usage(fac="F2", drug="d1", rate=2.0)}
        lots = [_lot(lot_id="L1", fac="F2", drug="d1", qty=10)]
        assert compute_need("F2", "d1", rates, lots, cover_days=14) == 18

    def test_compute_need_zero_demand(self):
        """F1 has 50 units of d1, uses 1/day, cover=14.
        need = max(0, floor(14×1) - 50) = max(0, 14 - 50) = 0
        """
        rates = {"F1|d1": _usage(fac="F1", drug="d1", rate=1.0)}
        lots = [_lot(lot_id="L1", fac="F1", drug="d1", qty=50)]
        assert compute_need("F1", "d1", rates, lots, cover_days=14) == 0

    def test_compute_need_zero_stock(self):
        """F3 has 0 units of d1, uses 3/day, cover=14.
        need = max(0, floor(14×3) - 0) = 42
        """
        rates = {"F3|d1": _usage(fac="F3", drug="d1", rate=3.0)}
        assert compute_need("F3", "d1", rates, [], cover_days=14) == 42

    def test_compute_need_fractional_rate(self):
        """F1 uses 1.5/day, stock=0, cover=14.
        need = floor(14 × 1.5) = floor(21.0) = 21
        """
        rates = {"F1|d1": _usage(fac="F1", drug="d1", rate=1.5)}
        assert compute_need("F1", "d1", rates, []) == 21

    def test_compute_need_missing_usage_rate(self):
        """No usage rate entry → rate=0 → need=0."""
        assert compute_need("F1", "d1", {}, []) == 0

    def test_compute_need_multiple_lots_summed(self):
        """Two lots at F1: 30 + 40 = 70, rate=10/day, cover=14.
        need = max(0, 140 - 70) = 70
        """
        rates = {"F1|d1": _usage(fac="F1", drug="d1", rate=10.0)}
        lots = [_lot(lot_id="A", fac="F1", qty=30),
                _lot(lot_id="B", fac="F1", qty=40)]
        assert compute_need("F1", "d1", rates, lots, cover_days=14) == 70


# ---------------------------------------------------------------------------
# compute_surplus_lots
# ---------------------------------------------------------------------------

class TestSurplusLots:
    def test_surplus_fefo_ordering(self):
        """F1 has 3 lots of d1 expiring in 10, 30, 60 days.
        Quantities: 20, 30, 30. Total=80. Reserve=20. Surplus=60.
        FEFO: L1(10d)→20, L2(30d)→30, L3(60d)→min(30,10)=10.
        Chunks ordered earliest-first.
        """
        lots = [
            _lot(lot_id="L3", fac="F1", qty=30, expiry=60),
            _lot(lot_id="L1", fac="F1", qty=20, expiry=10),
            _lot(lot_id="L2", fac="F1", qty=30, expiry=30),
        ]
        reserves = {"F1|d1": _reserve(min_stock=20)}
        rates = {"F1|d1": _usage(fac="F1", rate=0.0)}
        drugs = {"d1": _drug(price=10.0)}

        chunks = compute_surplus_lots("F1", "d1", lots, reserves, rates, drugs)

        assert len(chunks) == 3
        assert chunks[0].lot_id == "L1"
        assert chunks[0].days_to_expiry == 10
        assert chunks[0].quantity == 20
        assert chunks[1].lot_id == "L2"
        assert chunks[1].quantity == 30
        assert chunks[2].lot_id == "L3"
        assert chunks[2].quantity == 10
        assert sum(c.quantity for c in chunks) == 60

    def test_surplus_respects_reserve(self):
        """F1 has 50 units, reserve=40 → surplus=10 only."""
        lots = [_lot(fac="F1", qty=50, expiry=30)]
        reserves = {"F1|d1": _reserve(min_stock=40)}
        rates = {"F1|d1": _usage(fac="F1", rate=10.0)}
        drugs = {"d1": _drug()}

        chunks = compute_surplus_lots("F1", "d1", lots, reserves, rates, drugs)

        assert len(chunks) == 1
        assert chunks[0].quantity == 10

    def test_surplus_zero_when_at_reserve(self):
        """Stock == reserve → no surplus."""
        lots = [_lot(fac="F1", qty=40, expiry=30)]
        reserves = {"F1|d1": _reserve(min_stock=40)}
        rates = {"F1|d1": _usage(fac="F1", rate=10.0)}
        drugs = {"d1": _drug()}

        chunks = compute_surplus_lots("F1", "d1", lots, reserves, rates, drugs)
        assert len(chunks) == 0

    def test_surplus_excludes_expired_lots(self):
        """Lots with days_to_expiry <= 0 are never surplus."""
        lots = [_lot(lot_id="EXP", fac="F1", qty=200, expiry=0),
                _lot(lot_id="NEG", fac="F1", qty=100, expiry=-5)]
        reserves = {"F1|d1": _reserve(min_stock=0)}
        rates = {"F1|d1": _usage(fac="F1", rate=1.0)}
        drugs = {"d1": _drug()}

        chunks = compute_surplus_lots("F1", "d1", lots, reserves, rates, drugs)
        assert len(chunks) == 0

    def test_value_at_risk_calculation(self):
        """Lot: 50 units, expires in 15 days. Source usage=2/day.
        FEFO sim: consumed_by_expiry = floor(2×15) = 30.
        can_consume = min(50, 30) = 30.  waste = 50 - 30 = 20.
        Reserve=0 → surplus=50, chunk qty=50.
        at_risk = min(50, 20) = 20.  value_at_risk = 20 × 5.0 = 100.0
        """
        lots = [_lot(fac="F1", qty=50, expiry=15)]
        reserves = {"F1|d1": _reserve(min_stock=0)}
        rates = {"F1|d1": _usage(fac="F1", rate=2.0)}
        drugs = {"d1": _drug(price=5.0)}

        chunks = compute_surplus_lots("F1", "d1", lots, reserves, rates, drugs)

        assert len(chunks) == 1
        assert chunks[0].quantity == 50
        assert chunks[0].value_at_risk == 100.0  # 20 × 5.0

    def test_value_at_risk_high_usage_no_waste(self):
        """Usage=100/day, expiry=30 → consumed=3000 >> 200 qty → waste=0."""
        lots = [_lot(fac="F1", qty=200, expiry=30)]
        reserves = {"F1|d1": _reserve(min_stock=50)}
        rates = {"F1|d1": _usage(fac="F1", rate=100.0)}
        drugs = {"d1": _drug(price=5.0)}

        chunks = compute_surplus_lots("F1", "d1", lots, reserves, rates, drugs)
        assert chunks[0].value_at_risk == 0.0

    def test_deterministic_tiebreak_on_lot_id(self):
        """Same expiry → sorted by lot_id alphabetically."""
        lots = [_lot(lot_id="Z", fac="F1", qty=30, expiry=10),
                _lot(lot_id="A", fac="F1", qty=30, expiry=10)]
        reserves = {"F1|d1": _reserve(min_stock=0)}
        rates = {"F1|d1": _usage(fac="F1", rate=0.0)}
        drugs = {"d1": _drug()}

        chunks = compute_surplus_lots("F1", "d1", lots, reserves, rates, drugs)
        assert chunks[0].lot_id == "A"
        assert chunks[1].lot_id == "Z"


# ---------------------------------------------------------------------------
# compute_usable_quantity
# ---------------------------------------------------------------------------

class TestUsableQuantity:
    def test_usable_quantity(self):
        """Chunk: 50 units, expiry=20d, transit=2d, buffer=3.
        effective = 20 - 2 - 3 = 15.
        Dest usage = 3/day → floor(3 × 15) = 45.
        usable = min(50, 45) = 45.
        """
        chunk = _chunk(qty=50, expiry=20, src="F1")
        rates = {"F2|d1": _usage(fac="F2", drug="d1", rate=3.0)}
        routes = [_route(src="F1", dst="F2", transit=2)]

        assert compute_usable_quantity(chunk, "F2", rates, routes) == 45

    def test_usable_quantity_zero(self):
        """Chunk: expiry=5d, transit=3d, buffer=3.
        effective = 5 - 3 - 3 = -1 → 0.
        """
        chunk = _chunk(qty=100, expiry=5, src="F1")
        rates = {"F2|d1": _usage(fac="F2", drug="d1", rate=10.0)}
        routes = [_route(src="F1", dst="F2", transit=3)]

        assert compute_usable_quantity(chunk, "F2", rates, routes) == 0

    def test_usable_capped_by_chunk_quantity(self):
        """Chunk: 10 units, expiry=30d, transit=1d.
        effective = 26, floor(50 × 26) = 1300.
        usable = min(10, 1300) = 10.
        """
        chunk = _chunk(qty=10, expiry=30, src="F1")
        rates = {"F2|d1": _usage(fac="F2", drug="d1", rate=50.0)}
        routes = [_route(src="F1", dst="F2", transit=1)]

        assert compute_usable_quantity(chunk, "F2", rates, routes) == 10

    def test_usable_no_route(self):
        """No route → 0."""
        chunk = _chunk(qty=100, expiry=30)
        rates = {"F2|d1": _usage(fac="F2", rate=10.0)}
        assert compute_usable_quantity(chunk, "F2", rates, []) == 0

    def test_usable_zero_usage_rate(self):
        """Rate=0 → 0."""
        chunk = _chunk(qty=100, expiry=30)
        rates = {"F2|d1": _usage(fac="F2", rate=0.0)}
        routes = [_route(src="F1", dst="F2", transit=1)]
        assert compute_usable_quantity(chunk, "F2", rates, routes) == 0

    def test_usable_accepts_route_map_dict(self):
        """compute_usable_quantity also accepts a dict route_map."""
        chunk = _chunk(qty=50, expiry=20, src="F1")
        rates = {"F2|d1": _usage(fac="F2", rate=3.0)}
        rmap = {"F1|F2": _route(src="F1", dst="F2", transit=2)}
        assert compute_usable_quantity(chunk, "F2", rates, rmap) == 45

    def test_usable_effective_days_exactly_zero(self):
        """expiry=4, transit=1, buffer=3 → effective=0 → 0."""
        chunk = _chunk(qty=100, expiry=4)
        rates = {"F2|d1": _usage(fac="F2", rate=10.0)}
        routes = [_route(src="F1", dst="F2", transit=1)]
        assert compute_usable_quantity(chunk, "F2", rates, routes) == 0
