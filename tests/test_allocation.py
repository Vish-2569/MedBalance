"""Tests for the greedy FEFO baseline allocator.

Hand-calculated scenarios verifying sort order, lot exclusivity,
over-allocation prevention, and validation pass-through.
"""

import pytest
from math import floor

from backend.schemas import (
    Drug, DrugTier, InventoryData, InventorySnapshot,
    Reserve, Route, StorageCondition, SurplusChunk,
    TransferVariable, UsageRate, ValidationResult,
)
from backend.allocation import greedy_fefo_allocate, evaluate_greedy
from backend.validation import validate_plan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drug(code="d1", price=10.0, cold=False, tier=DrugTier.essential,
          criticality=2):
    return Drug(code=code, name="T", molecule="M", strength="1mg",
                form="tablet", tier=tier, criticality=criticality,
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGreedyPicksEarliestExpiryFirst:
    def test_greedy_picks_earliest_expiry_first(self):
        """Two variables for same (dest, drug): L1 expires in 10d, L2 in 30d.
        Greedy should pick L1 first (earliest expiry).
        Need=40, each usable=40 → only one fits.
        """
        v1 = _tv(var_id="x_0_F2", lot="L1", expiry=10, usable=40,
                 chunk_qty=40, val_risk=200.0, cost=100.0)
        v2 = _tv(var_id="x_1_F2", lot="L2", expiry=30, usable=40,
                 chunk_qty=40, val_risk=200.0, cost=100.0)
        snap = _snapshot(needs={"F2|d1": 40}, stock={"F1|d1": 500})
        data = _data(
            drugs={"d1": _drug()},
            route_map={"F1|F2": _route()},
            usage_rates={"F2|d1": _usage()},
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock={"F1|d1": 500},
        )
        selected = greedy_fefo_allocate([v2, v1], snap, data)  # unsorted input

        assert len(selected) == 1
        assert selected[0].lot_id == "L1"
        assert selected[0].days_to_expiry == 10


class TestGreedyRespectsCriticality:
    def test_greedy_respects_criticality(self):
        """Two variables with same expiry=10d:
        - insulin (life_saving, criticality=3) at F1 → F2
        - paracetamol (routine, criticality=1) at F3 → F4
        Greedy sorts by criticality DESC as tiebreaker → picks insulin first.
        """
        v_insulin = _tv(
            var_id="x_0_F2", lot="L1", src="F1", dst="F2", drug="insulin",
            expiry=10, usable=30, chunk_qty=30, val_risk=900.0, cost=100.0,
        )
        v_paracetamol = _tv(
            var_id="x_1_F4", lot="L2", src="F3", dst="F4", drug="paracetamol",
            expiry=10, usable=30, chunk_qty=30, val_risk=100.0, cost=100.0,
        )
        snap = _snapshot(
            needs={"F2|insulin": 50, "F4|paracetamol": 50},
            stock={"F1|insulin": 500, "F3|paracetamol": 500},
        )
        data = _data(
            drugs={
                "insulin": _drug(code="insulin", tier=DrugTier.life_saving,
                                 criticality=3),
                "paracetamol": _drug(code="paracetamol", tier=DrugTier.routine,
                                     criticality=1),
            },
            route_map={
                "F1|F2": _route(src="F1", dst="F2"),
                "F3|F4": _route(src="F3", dst="F4"),
            },
            usage_rates={
                "F2|insulin": _usage(fac="F2", drug="insulin"),
                "F4|paracetamol": _usage(fac="F4", drug="paracetamol"),
            },
            reserves={
                "F1|insulin": Reserve(facility_id="F1", drug_code="insulin",
                                      min_stock=50),
                "F3|paracetamol": Reserve(facility_id="F3",
                                          drug_code="paracetamol",
                                          min_stock=50),
            },
            stock={"F1|insulin": 500, "F3|paracetamol": 500},
        )
        # Both should be selected, but insulin should come first
        selected = greedy_fefo_allocate(
            [v_paracetamol, v_insulin], snap, data,
        )
        assert len(selected) == 2
        assert selected[0].drug_code == "insulin"
        assert selected[1].drug_code == "paracetamol"


class TestGreedyDoesNotDoubleAssignLot:
    def test_greedy_does_not_double_assign_lot(self):
        """Same lot L1 eligible for F2 and F3. Greedy picks the first
        encountered (lower expiry sort → same here, so first in iteration).
        Second variable for L1 must be skipped.
        """
        v1 = _tv(var_id="x_0_F2", lot="L1", dst="F2", expiry=10,
                 usable=40, chunk_qty=40, val_risk=200.0, cost=100.0)
        v2 = _tv(var_id="x_0_F3", lot="L1", dst="F3", expiry=10,
                 usable=40, chunk_qty=40, val_risk=200.0, cost=100.0)
        snap = _snapshot(
            needs={"F2|d1": 60, "F3|d1": 60},
            stock={"F1|d1": 500},
        )
        data = _data(
            drugs={"d1": _drug()},
            route_map={
                "F1|F2": _route(dst="F2"),
                "F1|F3": _route(dst="F3"),
            },
            usage_rates={
                "F2|d1": _usage(fac="F2"),
                "F3|d1": _usage(fac="F3"),
            },
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock={"F1|d1": 500},
        )
        selected = greedy_fefo_allocate([v1, v2], snap, data)

        lot_ids = [s.lot_id for s in selected]
        assert lot_ids.count("L1") == 1


class TestGreedyDoesNotOverAllocate:
    def test_greedy_does_not_over_allocate(self):
        """F2 needs 30 units of d1. Two variables offer usable=25 each.
        After selecting L1 (usable=25), remaining_need=5.
        L2 also offers usable=25 but remaining need is 5 → greedy still
        selects L2 (need > 0), but the total stays at 50 which is within
        the plan.  However, since need tracking goes to -20 after L2,
        a third variable L3 must be skipped.
        """
        v1 = _tv(var_id="x_0_F2", lot="L1", expiry=10, usable=25,
                 chunk_qty=25, val_risk=100.0, cost=50.0)
        v2 = _tv(var_id="x_1_F2", lot="L2", expiry=20, usable=25,
                 chunk_qty=25, val_risk=100.0, cost=50.0)
        v3 = _tv(var_id="x_2_F2", lot="L3", expiry=30, usable=25,
                 chunk_qty=25, val_risk=100.0, cost=50.0)
        snap = _snapshot(needs={"F2|d1": 30}, stock={"F1|d1": 500})
        data = _data(
            drugs={"d1": _drug()},
            route_map={"F1|F2": _route()},
            usage_rates={"F2|d1": _usage()},
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock={"F1|d1": 500},
        )
        selected = greedy_fefo_allocate([v1, v2, v3], snap, data)

        # L1 fills 25 of 30 need → remaining=5. L2 fills rest → remaining=-20.
        # L3 skipped because remaining_need <= 0.
        assert len(selected) == 2
        assert selected[0].lot_id == "L1"
        assert selected[1].lot_id == "L2"


class TestGreedyResultPassesValidation:
    def test_greedy_result_passes_validation(self):
        """Build a realistic scenario and verify the greedy result
        passes validate_plan with no violations.
        """
        v1 = _tv(var_id="x_0_F2", lot="L1", src="F1", dst="F2", drug="d1",
                 chunk_qty=80, usable=60, expiry=15, transit=1,
                 val_risk=400.0, cost=200.0)
        v2 = _tv(var_id="x_1_F3", lot="L2", src="F1", dst="F3", drug="d1",
                 chunk_qty=50, usable=40, expiry=20, transit=2,
                 val_risk=300.0, cost=150.0)
        snap = _snapshot(
            needs={"F2|d1": 70, "F3|d1": 50},
            stock={"F1|d1": 500},
        )
        data = _data(
            drugs={"d1": _drug()},
            route_map={
                "F1|F2": _route(src="F1", dst="F2", transit=1),
                "F1|F3": _route(src="F1", dst="F3", transit=2),
            },
            usage_rates={
                "F2|d1": _usage(fac="F2", rate=10.0),
                "F3|d1": _usage(fac="F3", rate=10.0),
            },
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock={"F1|d1": 500},
        )
        selected = greedy_fefo_allocate([v1, v2], snap, data)

        result = validate_plan(selected, [v1, v2], snap, data)
        assert result.is_valid is True
        assert result.violations == []


class TestEvaluateGreedy:
    def test_evaluate_metrics(self):
        """Two selected transfers:
        v1: usable=60, cost=200, val_risk=400
        v2: usable=40, cost=150, val_risk=300

        total_transfers = 2
        total_units_moved = 100
        total_transport_cost = 350
        total_value_saved = 700 - 350 = 350
        shortages_addressed = 2  (F2|d1 and F3|d1)
        shortages_remaining = 1  (F4|d2 unaddressed)
        """
        v1 = _tv(var_id="x_0_F2", dst="F2", drug="d1",
                 usable=60, cost=200.0, val_risk=400.0)
        v2 = _tv(var_id="x_1_F3", dst="F3", drug="d1",
                 usable=40, cost=150.0, val_risk=300.0)
        snap = _snapshot(
            needs={"F2|d1": 70, "F3|d1": 50, "F4|d2": 20},
        )

        metrics = evaluate_greedy([v1, v2], snap)

        assert metrics["total_transfers"] == 2
        assert metrics["total_units_moved"] == 100
        assert metrics["total_transport_cost"] == 350.0
        assert metrics["total_value_saved"] == 350.0
        assert metrics["shortages_addressed"] == 2
        assert metrics["shortages_remaining"] == 1

    def test_evaluate_empty(self):
        snap = _snapshot(needs={"F2|d1": 50})
        metrics = evaluate_greedy([], snap)
        assert metrics["total_transfers"] == 0
        assert metrics["total_units_moved"] == 0
        assert metrics["shortages_addressed"] == 0
        assert metrics["shortages_remaining"] == 1


class TestGreedySurplusExhaustion:
    def test_greedy_skips_when_surplus_exhausted(self):
        """F1 stock=100, reserve=50 → surplus=50. chunk_qty=50.
        After selecting v1 (chunk_qty=50), surplus=0.
        v2 (also chunk_qty=50 from F1) must be skipped.
        """
        v1 = _tv(var_id="x_0_F2", lot="L1", src="F1", dst="F2",
                 chunk_qty=50, usable=40, expiry=10,
                 val_risk=200.0, cost=100.0)
        v2 = _tv(var_id="x_1_F3", lot="L2", src="F1", dst="F3",
                 chunk_qty=50, usable=40, expiry=20,
                 val_risk=200.0, cost=100.0)
        snap = _snapshot(
            needs={"F2|d1": 60, "F3|d1": 60},
            stock={"F1|d1": 100},
        )
        data = _data(
            drugs={"d1": _drug()},
            route_map={
                "F1|F2": _route(dst="F2"),
                "F1|F3": _route(dst="F3"),
            },
            usage_rates={
                "F2|d1": _usage(fac="F2"),
                "F3|d1": _usage(fac="F3"),
            },
            reserves={"F1|d1": Reserve(facility_id="F1", drug_code="d1",
                                       min_stock=50)},
            stock={"F1|d1": 100},
        )
        selected = greedy_fefo_allocate([v1, v2], snap, data)

        assert len(selected) == 1
        assert selected[0].lot_id == "L1"
