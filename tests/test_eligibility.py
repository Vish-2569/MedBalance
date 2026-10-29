"""Tests for eligibility filtering and transfer variable creation.

Hand-calculated scenarios with 3 facilities, 2 drugs, known routes.
"""

import pytest
from math import floor

from backend.schemas import (
    Drug, DrugTier, Facility, FacilityType, InventoryData, Reserve,
    Route, StorageCondition, SurplusChunk, TransferVariable, UsageRate,
)
from backend.eligibility import (
    check_cold_chain, check_drug_identity, check_reserve_protection,
    check_route_exists, check_usable_before_expiry, create_transfer_variables,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drug(code="d1", price=10.0, cold=False):
    return Drug(code=code, name="T", molecule="M", strength="1mg",
                form="tablet", tier=DrugTier.essential, criticality=2,
                requires_cold_chain=cold, unit_price_inr=price)


def _chunk(idx=0, lot_id="L1", fac="F1", drug="d1", qty=100,
           expiry=30, batch="B1"):
    return SurplusChunk(
        chunk_id=f"CK-{idx + 1:03d}", chunk_index=idx, lot_id=lot_id,
        source_facility_id=fac, drug_code=drug, quantity=qty,
        days_to_expiry=expiry, storage_condition="room_temp",
        batch_number=batch, value_at_risk=qty * 10.0,
    )


def _route(src="F1", dst="F2", transit=1, cost=500.0, cold=False):
    return Route(source_id=src, destination_id=dst, transit_days=transit,
                 trip_cost_inr=cost, has_cold_transport=cold)


def _usage(fac="F2", drug="d1", rate=10.0):
    return UsageRate(facility_id=fac, drug_code=drug, units_per_day=rate)


def _facility(fid="F1"):
    return Facility(id=fid, name="T", type=FacilityType.phc,
                    latitude=0.0, longitude=0.0, has_cold_storage=False)


def _data(drugs=None, route_map=None, usage_rates=None, needs=None,
          facilities=None, reserves=None, stock=None):
    return InventoryData(
        facilities=facilities or {},
        drugs=drugs or {},
        lots=[],
        routes=[],
        usage_rates=usage_rates or {},
        reserves=reserves or {},
        route_map=route_map or {},
        stock_by_facility_drug=stock or {},
        needs=needs or {},
    )


# ---------------------------------------------------------------------------
# check_drug_identity
# ---------------------------------------------------------------------------

class TestDrugIdentity:
    def test_drug_identity_match(self):
        c = _chunk(drug="amoxicillin_500mg")
        assert check_drug_identity(c, "amoxicillin_500mg") is True

    def test_drug_identity_mismatch(self):
        c = _chunk(drug="amoxicillin_500mg")
        assert check_drug_identity(c, "metformin_500mg") is False

    def test_case_sensitive(self):
        c = _chunk(drug="Amoxicillin")
        assert check_drug_identity(c, "amoxicillin") is False


# ---------------------------------------------------------------------------
# check_route_exists
# ---------------------------------------------------------------------------

class TestRouteExists:
    def test_route_exists(self):
        r = _route(src="F1", dst="F2")
        rmap = {"F1|F2": r}
        assert check_route_exists("F1", "F2", rmap) is r

    def test_route_not_exists(self):
        rmap = {"F1|F2": _route(src="F1", dst="F2")}
        assert check_route_exists("F2", "F1", rmap) is None

    def test_empty_map(self):
        assert check_route_exists("F1", "F2", {}) is None


# ---------------------------------------------------------------------------
# check_cold_chain
# ---------------------------------------------------------------------------

class TestColdChain:
    def test_cold_chain_pass(self):
        assert check_cold_chain(_drug(cold=True), _route(cold=True)) is True

    def test_cold_chain_fail(self):
        assert check_cold_chain(_drug(cold=True), _route(cold=False)) is False

    def test_cold_chain_room_temp(self):
        assert check_cold_chain(_drug(cold=False), _route(cold=False)) is True


# ---------------------------------------------------------------------------
# check_usable_before_expiry
# ---------------------------------------------------------------------------

class TestUsableBeforeExpiry:
    def test_basic(self):
        """chunk: 100 units, expiry=30, transit=1, buffer=3.
        effective = 30-1-3 = 26.  rate=5.  usable = min(100, floor(5×26)) = 100.
        """
        chunk = _chunk(qty=100, expiry=30)
        rates = {"F2|d1": _usage(rate=5.0)}
        assert check_usable_before_expiry(chunk, "F2", rates, _route(transit=1)) == 100

    def test_capped_by_consumption(self):
        """chunk: 1000 units, expiry=10, transit=1, buffer=3.
        effective = 6.  rate=5.  usable = min(1000, 30) = 30.
        """
        chunk = _chunk(qty=1000, expiry=10)
        rates = {"F2|d1": _usage(rate=5.0)}
        assert check_usable_before_expiry(chunk, "F2", rates, _route(transit=1)) == 30

    def test_zero_when_expires_too_soon(self):
        """expiry=4, transit=1, buffer=3 → effective=0 → 0."""
        chunk = _chunk(qty=100, expiry=4)
        rates = {"F2|d1": _usage(rate=10.0)}
        assert check_usable_before_expiry(chunk, "F2", rates, _route(transit=1)) == 0

    def test_zero_usage_rate(self):
        """rate=0 → 0."""
        chunk = _chunk(qty=100, expiry=30)
        rates = {"F2|d1": _usage(rate=0.0)}
        assert check_usable_before_expiry(chunk, "F2", rates, _route(transit=1)) == 0

    def test_missing_usage_rate(self):
        """No rate entry → 0."""
        chunk = _chunk(qty=100, expiry=30)
        assert check_usable_before_expiry(chunk, "F2", {}, _route(transit=1)) == 0


# ---------------------------------------------------------------------------
# check_reserve_protection
# ---------------------------------------------------------------------------

class TestReserveProtection:
    def test_positive(self):
        assert check_reserve_protection(_chunk(qty=1)) is True

    def test_zero(self):
        assert check_reserve_protection(_chunk(qty=0)) is False


# ---------------------------------------------------------------------------
# create_transfer_variables — hand-calculated scenario
# ---------------------------------------------------------------------------

class TestVariableCreation:
    def test_variable_creation_filters_correctly(self):
        """Setup:
        F1: surplus chunk of d1, 100 units, expiry=30, chunk_index=0
        F2: needs d1 (need=50), route F1→F2 transit=1, usage=5/day
        F3: needs d1 (need=30), route F1→F3 transit=2, usage=3/day

        Expected variables:
        x_0_F2: usable = min(100, floor(5×(30-1-3))) = min(100, 130) = 100
        x_0_F3: usable = min(100, floor(3×(30-2-3))) = min(100, 75) = 75
        Total: 2 variables.
        """
        chunk = _chunk(idx=0, fac="F1", drug="d1", qty=100, expiry=30)
        data = _data(
            drugs={"d1": _drug()},
            route_map={
                "F1|F2": _route(src="F1", dst="F2", transit=1, cost=300.0),
                "F1|F3": _route(src="F1", dst="F3", transit=2, cost=600.0),
            },
            usage_rates={
                "F2|d1": _usage(fac="F2", rate=5.0),
                "F3|d1": _usage(fac="F3", rate=3.0),
            },
            needs={"F2|d1": 50, "F3|d1": 30},
        )
        variables = create_transfer_variables([chunk], data)

        assert len(variables) == 2
        by_dest = {v.destination_id: v for v in variables}
        assert "F2" in by_dest
        assert "F3" in by_dest

        assert by_dest["F2"].var_id == "x_0_F2"
        assert by_dest["F2"].usable_quantity == 100
        assert by_dest["F2"].transport_cost == 300.0

        assert by_dest["F3"].var_id == "x_0_F3"
        assert by_dest["F3"].usable_quantity == 75
        assert by_dest["F3"].transport_cost == 600.0

    def test_no_self_transfer(self):
        """F1 has surplus of d1 and F1 also needs d1 → no variable."""
        chunk = _chunk(idx=0, fac="F1", drug="d1", qty=100, expiry=30)
        data = _data(
            drugs={"d1": _drug()},
            route_map={"F1|F1": _route(src="F1", dst="F1")},
            usage_rates={"F1|d1": _usage(fac="F1", rate=10.0)},
            needs={"F1|d1": 50},
        )
        assert create_transfer_variables([chunk], data) == []

    def test_zero_need_no_variable(self):
        """F2 has zero need for d1 → not in needs dict → no variable."""
        chunk = _chunk(idx=0, fac="F1", drug="d1", qty=100, expiry=30)
        data = _data(
            drugs={"d1": _drug()},
            route_map={"F1|F2": _route(src="F1", dst="F2")},
            usage_rates={"F2|d1": _usage(fac="F2", rate=5.0)},
            needs={},  # no need for d1 at F2
        )
        assert create_transfer_variables([chunk], data) == []

    def test_cold_chain_blocks_variable(self):
        """Cold drug d2 on non-cold route → no variable."""
        chunk = _chunk(idx=0, fac="F1", drug="d2", qty=100, expiry=30)
        data = _data(
            drugs={"d2": _drug(code="d2", cold=True)},
            route_map={"F1|F2": _route(src="F1", dst="F2", cold=False)},
            usage_rates={"F2|d2": _usage(fac="F2", drug="d2", rate=10.0)},
            needs={"F2|d2": 50},
        )
        assert create_transfer_variables([chunk], data) == []

    def test_expiry_too_short_blocks_variable(self):
        """Chunk expiry=3d, transit=1d, buffer=3 → effective=-1 → filtered."""
        chunk = _chunk(idx=0, fac="F1", drug="d1", qty=100, expiry=3)
        data = _data(
            drugs={"d1": _drug()},
            route_map={"F1|F2": _route(src="F1", dst="F2", transit=1)},
            usage_rates={"F2|d1": _usage(fac="F2", rate=10.0)},
            needs={"F2|d1": 50},
        )
        assert create_transfer_variables([chunk], data) == []

    def test_no_route_blocks_variable(self):
        """No route from F1 to F2 → filtered."""
        chunk = _chunk(idx=0, fac="F1", drug="d1", qty=100, expiry=30)
        data = _data(
            drugs={"d1": _drug()},
            route_map={},
            needs={"F2|d1": 50},
        )
        assert create_transfer_variables([chunk], data) == []

    def test_multiple_chunks_multiple_destinations(self):
        """Two chunks from different sources to same destination.
        CK0 at F1, CK1 at F3. F2 needs d1. Routes: F1→F2, F3→F2.
        """
        c0 = _chunk(idx=0, fac="F1", drug="d1", qty=80, expiry=20)
        c1 = _chunk(idx=1, lot_id="L2", fac="F3", drug="d1", qty=60, expiry=40)
        data = _data(
            drugs={"d1": _drug()},
            route_map={
                "F1|F2": _route(src="F1", dst="F2", transit=1),
                "F3|F2": _route(src="F3", dst="F2", transit=2),
            },
            usage_rates={"F2|d1": _usage(fac="F2", rate=4.0)},
            needs={"F2|d1": 100},
        )
        variables = create_transfer_variables([c0, c1], data)
        assert len(variables) == 2
        assert {v.source_id for v in variables} == {"F1", "F3"}
        # c0: usable = min(80, floor(4×(20-1-3))) = min(80, 64) = 64
        assert variables[0].usable_quantity == 64
        # c1: usable = min(60, floor(4×(40-2-3))) = min(60, 140) = 60
        assert variables[1].usable_quantity == 60
