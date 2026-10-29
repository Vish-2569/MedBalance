"""Tests for post-solve validation: plan checks, single-transfer checks,
and violation resolution.

Hand-calculated scenarios — every number traceable by hand.
"""

import pytest
from math import floor

from backend.schemas import (
    Drug, DrugTier, Facility, FacilityType, InventoryData, InventorySnapshot,
    Reserve, Route, StorageCondition, TransferVariable,
    UsageRate, ValidationResult,
)
from backend.validation import (
    validate_plan, validate_single_transfer, resolve_violations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drug(code="d1", price=10.0, cold=False):
    return Drug(code=code, name="T", molecule="M", strength="1mg",
                form="tablet", tier=DrugTier.essential, criticality=2,
                requires_cold_chain=cold, unit_price_inr=price)


def _route(src="F1", dst="F2", transit=1, cost=500.0, cold=False):
    return Route(source_id=src, destination_id=dst, transit_days=transit,
                 trip_cost_inr=cost, has_cold_transport=cold)


def _usage(fac="F2", drug="d1", rate=10.0):
    return UsageRate(facility_id=fac, drug_code=drug, units_per_day=rate)


def _tv(var_id="x_0_F2", idx=0, lot="L1", src="F1", dst="F2", drug="d1",
        chunk_qty=100, usable=50, expiry=30, transit=1,
        val_risk=500.0, cost=300.0):
    return TransferVariable(
        var_id=var_id, chunk_index=idx, lot_id=lot, source_id=src,
        destination_id=dst, drug_code=drug, chunk_quantity=chunk_qty,
        usable_quantity=usable, days_to_expiry=expiry, transit_days=transit,
        value_at_risk=val_risk, transport_cost=cost,
    )


def _snapshot(needs=None, stock=None):
    return InventorySnapshot(
        needs=needs or {},
        surplus_chunks=[],
        facility_stock=stock or {},
    )


def _data(drugs=None, route_map=None, usage_rates=None, reserves=None,
          stock=None):
    return InventoryData(
        facilities={}, drugs=drugs or {}, lots=[], routes=[],
        usage_rates=usage_rates or {}, reserves=reserves or {},
        route_map=route_map or {}, stock_by_facility_drug=stock or {},
        needs={},
    )


def _valid_scenario():
    """Minimal valid scenario: F1 sends 100 units to F2, which needs 60.
    Stock at F1=500, reserve=50 → remaining=400 >= 50. OK.
    Drug d1, not cold, route exists, expiry=30, transit=1, buffer=3.
    effective=26, rate=10 → usable=min(100, 260)=100. But we set usable=50.
    Check (e) recomputes: effective=26>0, floor(10×26)=260>0. Pass.
    Over-alloc: sent=50, need=60, tol=max(1,3)=3. 50 <= 63. Pass.
    """
    tv = _tv(var_id="x_0_F2", lot="L1", src="F1", dst="F2", drug="d1",
             chunk_qty=100, usable=50, expiry=30, transit=1)
    snap = _snapshot(needs={"F2|d1": 60}, stock={"F1|d1": 500})
    data = _data(
        drugs={"d1": _drug()},
        route_map={"F1|F2": _route(src="F1", dst="F2", transit=1)},
        usage_rates={"F2|d1": _usage(rate=10.0)},
        reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                   min_stock=50)},
        stock={"F1|d1": 500},
    )
    return [tv], [tv], snap, data


# ---------------------------------------------------------------------------
# validate_plan
# ---------------------------------------------------------------------------

class TestValidatePlan:
    def test_valid_plan_passes(self):
        """A plan with no violations → is_valid=True, violations=[]."""
        selected, all_vars, snap, data = _valid_scenario()
        result = validate_plan(selected, all_vars, snap, data)
        assert result.is_valid is True
        assert result.violations == []

    def test_empty_plan(self):
        _, all_vars, snap, data = _valid_scenario()
        result = validate_plan([], all_vars, snap, data)
        assert result.is_valid is True
        assert any("Empty" in w for w in result.warnings)


class TestLotDoubleAssignment:
    def test_lot_double_assignment(self):
        """Same lot L1 sent to F2 AND F3 → violation."""
        t1 = _tv(var_id="x_0_F2", lot="L1", dst="F2", usable=30, chunk_qty=50)
        t2 = _tv(var_id="x_0_F3", lot="L1", dst="F3", usable=30, chunk_qty=50)
        snap = _snapshot(needs={"F2|d1": 60, "F3|d1": 60}, stock={"F1|d1": 500})
        data = _data(
            drugs={"d1": _drug()},
            route_map={"F1|F2": _route(dst="F2"),
                       "F1|F3": _route(dst="F3")},
            usage_rates={"F2|d1": _usage(fac="F2"),
                         "F3|d1": _usage(fac="F3")},
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock={"F1|d1": 500},
        )
        result = validate_plan([t1, t2], [t1, t2], snap, data)
        assert result.is_valid is False
        assert any("L1" in v and "assigned to both" in v
                    for v in result.violations)


class TestDuplicateLot:
    def test_duplicate_lot(self):
        """Same lot L1 appears twice for the same destination → violation."""
        t1 = _tv(var_id="x_0_F2", lot="L1", dst="F2", usable=30, chunk_qty=50)
        t2 = _tv(var_id="x_1_F2", lot="L1", dst="F2", usable=30, chunk_qty=50)
        snap = _snapshot(needs={"F2|d1": 100}, stock={"F1|d1": 500})
        data = _data(
            drugs={"d1": _drug()},
            route_map={"F1|F2": _route(dst="F2")},
            usage_rates={"F2|d1": _usage(fac="F2")},
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock={"F1|d1": 500},
        )
        result = validate_plan([t1, t2], [t1, t2], snap, data)
        assert result.is_valid is False
        assert any("duplicate" in v for v in result.violations)


class TestReserveViolation:
    def test_reserve_violation(self):
        """F1 stock=500, reserve=50, sending chunk_qty=460.
        Remaining = 500-460 = 40 < 50 → violation.
        """
        tv = _tv(chunk_qty=460, usable=50)
        snap = _snapshot(needs={"F2|d1": 60}, stock={"F1|d1": 500})
        data = _data(
            drugs={"d1": _drug()},
            route_map={"F1|F2": _route()},
            usage_rates={"F2|d1": _usage()},
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock={"F1|d1": 500},
        )
        result = validate_plan([tv], [tv], snap, data)
        assert result.is_valid is False
        assert any("below reserve" in v for v in result.violations)

    def test_exactly_at_reserve_ok(self):
        """500-450=50 == reserve → OK."""
        tv = _tv(chunk_qty=450, usable=50)
        snap = _snapshot(needs={"F2|d1": 60}, stock={"F1|d1": 500})
        data = _data(
            drugs={"d1": _drug()},
            route_map={"F1|F2": _route()},
            usage_rates={"F2|d1": _usage()},
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock={"F1|d1": 500},
        )
        result = validate_plan([tv], [tv], snap, data)
        assert result.is_valid is True


class TestWrongDrugMatch:
    def test_wrong_drug_match(self):
        """Sending d1 to F2 but F2 has no need for d1 → violation."""
        tv = _tv(drug="d1", dst="F2")
        snap = _snapshot(needs={}, stock={"F1|d1": 500})
        data = _data(
            drugs={"d1": _drug()},
            route_map={"F1|F2": _route()},
            usage_rates={"F2|d1": _usage()},
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock={"F1|d1": 500},
        )
        result = validate_plan([tv], [tv], snap, data)
        assert result.is_valid is False
        assert any("no need" in v for v in result.violations)


class TestColdChainViolation:
    def test_cold_chain_violation(self):
        """Cold drug d1 on non-cold route → violation."""
        tv = _tv(drug="d1", expiry=30, transit=1)
        snap = _snapshot(needs={"F2|d1": 60}, stock={"F1|d1": 500})
        data = _data(
            drugs={"d1": _drug(cold=True)},
            route_map={"F1|F2": _route(cold=False)},
            usage_rates={"F2|d1": _usage()},
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock={"F1|d1": 500},
        )
        result = validate_plan([tv], [tv], snap, data)
        assert result.is_valid is False
        assert any("cold-chain" in v for v in result.violations)

    def test_cold_drug_cold_route_ok(self):
        tv = _tv(drug="d1", expiry=30, transit=1)
        snap = _snapshot(needs={"F2|d1": 60}, stock={"F1|d1": 500})
        data = _data(
            drugs={"d1": _drug(cold=True)},
            route_map={"F1|F2": _route(cold=True)},
            usage_rates={"F2|d1": _usage()},
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock={"F1|d1": 500},
        )
        result = validate_plan([tv], [tv], snap, data)
        assert result.is_valid is True

    def test_cold_drug_destination_lacks_cold_storage(self):
        """Cold drug on cold route but destination has no cold storage → violation."""
        tv = _tv(drug="d1", expiry=30, transit=1)
        snap = _snapshot(needs={"F2|d1": 60}, stock={"F1|d1": 500})
        f1 = Facility(id="F1", name="Source", type=FacilityType.hospital,
                      latitude=0, longitude=0, has_cold_storage=True)
        f2 = Facility(id="F2", name="Dest", type=FacilityType.phc,
                      latitude=0, longitude=0, has_cold_storage=False)
        data = InventoryData(
            facilities={"F1": f1, "F2": f2},
            drugs={"d1": _drug(cold=True)},
            lots=[], routes=[],
            route_map={"F1|F2": _route(cold=True)},
            usage_rates={"F2|d1": _usage()},
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock_by_facility_drug={"F1|d1": 500},
            needs={"F2|d1": 60},
        )
        result = validate_plan([tv], [tv], snap, data)
        assert result.is_valid is False
        assert any("lacks cold storage" in v for v in result.violations)

    def test_cold_drug_destination_has_cold_storage_ok(self):
        """Cold drug, cold route, destination has cold storage → valid."""
        tv = _tv(drug="d1", expiry=30, transit=1)
        snap = _snapshot(needs={"F2|d1": 60}, stock={"F1|d1": 500})
        f1 = Facility(id="F1", name="Source", type=FacilityType.hospital,
                      latitude=0, longitude=0, has_cold_storage=True)
        f2 = Facility(id="F2", name="Dest", type=FacilityType.hospital,
                      latitude=0, longitude=0, has_cold_storage=True)
        data = InventoryData(
            facilities={"F1": f1, "F2": f2},
            drugs={"d1": _drug(cold=True)},
            lots=[], routes=[],
            route_map={"F1|F2": _route(cold=True)},
            usage_rates={"F2|d1": _usage()},
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock_by_facility_drug={"F1|d1": 500},
            needs={"F2|d1": 60},
        )
        result = validate_plan([tv], [tv], snap, data)
        assert result.is_valid is True


class TestOverAllocation:
    def test_over_allocation(self):
        """F2 needs 20 units of d1, receives usable=30.
        tolerance = max(1, floor(0.05 × 20)) = max(1, 1) = 1.
        sent=30 > 20+1=21 → violation.
        """
        tv = _tv(usable=30, chunk_qty=100)
        snap = _snapshot(needs={"F2|d1": 20}, stock={"F1|d1": 500})
        data = _data(
            drugs={"d1": _drug()},
            route_map={"F1|F2": _route()},
            usage_rates={"F2|d1": _usage()},
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock={"F1|d1": 500},
        )
        result = validate_plan([tv], [tv], snap, data)
        assert result.is_valid is False
        assert any("Over-allocation" in v for v in result.violations)

    def test_within_tolerance(self):
        """need=50, sent=52. tolerance=max(1, floor(0.05×50))=2. 52 <= 52. OK."""
        tv = _tv(usable=52, chunk_qty=100)
        snap = _snapshot(needs={"F2|d1": 50}, stock={"F1|d1": 500})
        data = _data(
            drugs={"d1": _drug()},
            route_map={"F1|F2": _route()},
            usage_rates={"F2|d1": _usage()},
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock={"F1|d1": 500},
        )
        result = validate_plan([tv], [tv], snap, data)
        assert result.is_valid is True


class TestFEFODeviationWarning:
    def test_fefo_deviation_warning(self):
        """Selected lot L2 (expiry=30d) while unselected L1 (expiry=10d)
        is available for same (dest, drug). → FEFO deviation warning.
        Plan remains valid (warnings don't block).
        """
        sel = _tv(var_id="x_1_F2", lot="L2", idx=1, expiry=30, usable=50)
        unsel = _tv(var_id="x_0_F2", lot="L1", idx=0, expiry=10, usable=30)
        snap = _snapshot(needs={"F2|d1": 60}, stock={"F1|d1": 500})
        data = _data(
            drugs={"d1": _drug()},
            route_map={"F1|F2": _route()},
            usage_rates={"F2|d1": _usage()},
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock={"F1|d1": 500},
        )
        result = validate_plan([sel], [sel, unsel], snap, data)
        assert result.is_valid is True
        assert len(result.fefo_deviations) > 0
        assert any("FEFO deviation" in d for d in result.fefo_deviations)

    def test_no_fefo_deviation_when_respected(self):
        """Select earlier-expiring lot → no deviation."""
        sel = _tv(var_id="x_0_F2", lot="L1", idx=0, expiry=10, usable=30)
        unsel = _tv(var_id="x_1_F2", lot="L2", idx=1, expiry=30, usable=50)
        snap = _snapshot(needs={"F2|d1": 60}, stock={"F1|d1": 500})
        data = _data(
            drugs={"d1": _drug()},
            route_map={"F1|F2": _route()},
            usage_rates={"F2|d1": _usage()},
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock={"F1|d1": 500},
        )
        result = validate_plan([sel], [sel, unsel], snap, data)
        assert result.fefo_deviations == []


class TestUsableBeforeExpiry:
    def test_expires_before_usable(self):
        """expiry=3, transit=1, buffer=3 → effective=-1 → violation."""
        tv = _tv(expiry=3, transit=1)
        snap = _snapshot(needs={"F2|d1": 60}, stock={"F1|d1": 500})
        data = _data(
            drugs={"d1": _drug()},
            route_map={"F1|F2": _route(transit=1)},
            usage_rates={"F2|d1": _usage()},
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock={"F1|d1": 500},
        )
        result = validate_plan([tv], [tv], snap, data)
        assert result.is_valid is False
        assert any("expires before usable" in v for v in result.violations)


# ---------------------------------------------------------------------------
# validate_single_transfer
# ---------------------------------------------------------------------------

class TestValidateSingleTransfer:
    def test_valid_single(self):
        tv = _tv(expiry=30, transit=1, chunk_qty=100)
        data = _data(
            drugs={"d1": _drug()},
            route_map={"F1|F2": _route(transit=1)},
            usage_rates={"F2|d1": _usage(rate=10.0)},
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock={"F1|d1": 500},
        )
        result = validate_single_transfer(tv, data)
        assert result.is_valid is True

    def test_self_transfer_blocked(self):
        tv = _tv(src="F1", dst="F1")
        data = _data(
            drugs={"d1": _drug()},
            route_map={"F1|F1": _route(src="F1", dst="F1")},
            usage_rates={"F1|d1": _usage(fac="F1")},
            stock={"F1|d1": 500},
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
        )
        result = validate_single_transfer(tv, data)
        assert result.is_valid is False
        assert any("same facility" in v for v in result.violations)

    def test_unknown_drug(self):
        tv = _tv(drug="nonexistent")
        data = _data(drugs={})
        result = validate_single_transfer(tv, data)
        assert result.is_valid is False
        assert any("Unknown drug" in v for v in result.violations)

    def test_no_route(self):
        tv = _tv()
        data = _data(drugs={"d1": _drug()}, route_map={})
        result = validate_single_transfer(tv, data)
        assert result.is_valid is False
        assert any("No route" in v for v in result.violations)


# ---------------------------------------------------------------------------
# resolve_violations
# ---------------------------------------------------------------------------

class TestResolveViolations:
    def test_resolves_duplicate_lot(self):
        """L1 sent to both F2 (usable=30) and F3 (usable=50).
        Resolver keeps higher usable → keeps x_0_F3.
        """
        t1 = _tv(var_id="x_0_F2", lot="L1", dst="F2", usable=30, chunk_qty=50)
        t2 = _tv(var_id="x_0_F3", lot="L1", dst="F3", usable=50, chunk_qty=50)
        snap = _snapshot(
            needs={"F2|d1": 60, "F3|d1": 60},
            stock={"F1|d1": 500},
        )
        data = _data(
            drugs={"d1": _drug()},
            route_map={"F1|F2": _route(dst="F2"),
                       "F1|F3": _route(dst="F3")},
            usage_rates={"F2|d1": _usage(fac="F2"),
                         "F3|d1": _usage(fac="F3")},
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock={"F1|d1": 500},
        )
        plan, result = resolve_violations([t1, t2], [t1, t2], snap, data)
        assert result.is_valid is True
        assert len(plan) == 1
        assert plan[0].var_id == "x_0_F3"

    def test_already_valid_unchanged(self):
        selected, all_vars, snap, data = _valid_scenario()
        plan, result = resolve_violations(selected, all_vars, snap, data)
        assert result.is_valid is True
        assert len(plan) == len(selected)

    def test_resolves_over_allocation(self):
        """F2 needs 10. Two transfers send usable=40 each (total=80).
        tolerance=max(1, floor(0.05×10))=1.  80 > 11.
        Resolver removes lowest value_at_risk first.
        """
        t1 = _tv(var_id="x_0_F2", lot="L1", usable=40, chunk_qty=40,
                 val_risk=100.0)
        t2 = _tv(var_id="x_1_F2", lot="L2", usable=40, chunk_qty=40,
                 val_risk=500.0)
        snap = _snapshot(needs={"F2|d1": 10}, stock={"F1|d1": 500})
        data = _data(
            drugs={"d1": _drug()},
            route_map={"F1|F2": _route()},
            usage_rates={"F2|d1": _usage()},
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock={"F1|d1": 500},
        )
        plan, result = resolve_violations([t1, t2], [t1, t2], snap, data)
        assert len(plan) < 2
