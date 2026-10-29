"""Brute-force verification of QUBO formulation.

Enumerates ALL 2^n bitstrings on tiny instances to find the exact global
optimum. The QUBO is not accepted until it matches exact enumeration on
every test case.
"""

import pytest
from itertools import product
from math import floor

from backend.schemas import (
    Drug, DrugTier, Facility, FacilityType, InventoryData,
    InventorySnapshot, Lot, Reserve, Route, StorageCondition,
    SurplusChunk, TransferVariable, UsageRate,
)
from backend.eligibility import create_transfer_variables
from backend.qubo_builder import (
    build_q_matrix, compute_objective_value, decode_bitstring,
    FEFO_BONUS_WEIGHT,
)
from backend.validation import validate_plan
from backend.allocation import greedy_fefo_allocate


# ---------------------------------------------------------------------------
# Brute-force solver
# ---------------------------------------------------------------------------

def brute_force_solve(
    Q: dict[tuple[str, str], float],
    variables: list[TransferVariable],
) -> dict:
    """Enumerate all 2^n bitstrings and find the exact global optimum.

    Returns dict with optimal_bitstring, optimal_value, and all_solutions
    sorted by objective value ascending (best first).
    """
    var_ids = [v.var_id for v in variables]
    n = len(var_ids)
    assert n <= 15, f"Too many variables for brute-force: {n}"

    best_bits: dict[str, int] = {v: 0 for v in var_ids}
    best_h = 0.0  # all-zeros baseline
    all_solutions: list[tuple[float, dict[str, int]]] = []

    for mask in range(2 ** n):
        bits = {}
        for i, vid in enumerate(var_ids):
            bits[vid] = (mask >> i) & 1
        h = compute_objective_value(bits, Q)
        all_solutions.append((h, dict(bits)))
        if h < best_h:
            best_h = h
            best_bits = dict(bits)

    all_solutions.sort(key=lambda x: x[0])

    return {
        "optimal_bitstring": best_bits,
        "optimal_value": best_h,
        "all_solutions": all_solutions,
    }


# ---------------------------------------------------------------------------
# Helpers — tiny dataset builders
# ---------------------------------------------------------------------------

def _drug(code="d1", price=10.0, cold=False, criticality=2,
          tier=DrugTier.essential):
    return Drug(code=code, name="T", molecule="M", strength="1mg",
                form="tablet", tier=tier, criticality=criticality,
                requires_cold_chain=cold, unit_price_inr=price)


def _chunk(idx=0, lot_id="L1", src="F1", drug="d1", qty=100,
           expiry=20, val_risk=1000.0):
    return SurplusChunk(
        chunk_id=f"CK-{idx + 1:03d}", chunk_index=idx, lot_id=lot_id,
        source_facility_id=src, drug_code=drug, quantity=qty,
        days_to_expiry=expiry, storage_condition="room_temp",
        batch_number="B1", value_at_risk=val_risk,
    )


def _route(src="F1", dst="F2", transit=1, cost=300.0, cold=False):
    return Route(source_id=src, destination_id=dst, transit_days=transit,
                 trip_cost_inr=cost, has_cold_transport=cold)


def _usage(fac="F2", drug="d1", rate=10.0):
    return UsageRate(facility_id=fac, drug_code=drug, units_per_day=rate)


def _reserve(fac="F1", drug="d1", min_stock=50):
    return Reserve(facility_id=fac, drug_code=drug, min_stock=min_stock)


def _make_scenario(chunks, drugs, routes, usage_rates, needs,
                   reserves=None, stock=None):
    """Build (chunks, snapshot, data) from minimal inputs for the pipeline."""
    route_map = {f"{r.source_id}|{r.destination_id}": r for r in routes}
    rates_dict = {f"{u.facility_id}|{u.drug_code}": u for u in usage_rates}
    drugs_dict = {d.code: d for d in drugs}
    reserves_list = reserves or []
    reserves_dict = {f"{r.facility_id}|{r.drug_code}": r for r in reserves_list}
    stock_dict = stock or {}

    snapshot = InventorySnapshot(
        needs=needs,
        surplus_chunks=chunks,
        facility_stock=stock_dict,
    )
    data = InventoryData(
        facilities={},
        drugs=drugs_dict,
        lots=[],
        routes=routes,
        usage_rates=rates_dict,
        reserves=reserves_dict,
        route_map=route_map,
        stock_by_facility_drug=stock_dict,
        needs=needs,
    )
    return chunks, snapshot, data


def _run_pipeline(chunks, snapshot, data, **qubo_kwargs):
    """Run eligibility → QUBO → brute-force → decode → validate."""
    variables = create_transfer_variables(chunks, data)
    if not variables:
        return {
            "variables": [],
            "Q": {},
            "bf_result": {"optimal_bitstring": {}, "optimal_value": 0.0,
                          "all_solutions": []},
            "selected": [],
            "validation": None,
        }

    Q = build_q_matrix(variables, snapshot, data, **qubo_kwargs)
    bf = brute_force_solve(Q, variables)
    selected = decode_bitstring(bf["optimal_bitstring"], variables)
    val_result = validate_plan(selected, variables, snapshot, data)

    return {
        "variables": variables,
        "Q": Q,
        "bf_result": bf,
        "selected": selected,
        "validation": val_result,
    }


# ---------------------------------------------------------------------------
# Test 1 — Zero demand at destination
# ---------------------------------------------------------------------------

class TestZeroDemand:
    def test_no_variables_created(self):
        """F1 has surplus of d1. F2 has zero need.
        No variables should be created → empty Q → no division by zero.
        """
        chunks = [_chunk(idx=0, src="F1", drug="d1", qty=100, expiry=20)]
        drugs = [_drug(code="d1")]
        routes = [_route(src="F1", dst="F2", transit=1)]
        rates = [_usage(fac="F2", drug="d1", rate=10.0)]
        needs = {}  # no need at F2

        _, snap, data = _make_scenario(chunks, drugs, routes, rates, needs)
        result = _run_pipeline(chunks, snap, data)

        assert result["variables"] == []
        assert result["Q"] == {}
        assert result["selected"] == []


# ---------------------------------------------------------------------------
# Test 2 — One chunk, two competing destinations
# ---------------------------------------------------------------------------

class TestOneChunkTwoDestinations:
    def test_optimizer_picks_higher_need_destination(self):
        """One chunk at F1, two destinations: F2 (need=30), F3 (need=80).
        Same lot → P1 prevents both. F3 gives better fill → selected.

        x_0_F2: usable=min(100, floor(5×16))=80, cap=30/80=0.375
        x_0_F3: usable=min(100, floor(10×15))=100, cap=80/100=0.8
        F3 has higher reward AND better P2 ratio → optimizer picks F3.
        """
        chunks = [_chunk(idx=0, lot_id="L1", src="F1", drug="d1",
                         qty=100, expiry=20, val_risk=1000.0)]
        drugs = [_drug(code="d1", criticality=2)]
        routes = [_route(src="F1", dst="F2", transit=1, cost=300.0),
                  _route(src="F1", dst="F3", transit=2, cost=500.0)]
        rates = [_usage(fac="F2", drug="d1", rate=5.0),
                 _usage(fac="F3", drug="d1", rate=10.0)]
        needs = {"F2|d1": 30, "F3|d1": 80}
        stock = {"F1|d1": 1000}
        reserves = [_reserve(fac="F1", drug="d1", min_stock=50)]

        _, snap, data = _make_scenario(
            chunks, drugs, routes, rates, needs,
            reserves=reserves, stock=stock,
        )
        result = _run_pipeline(chunks, snap, data)

        assert len(result["variables"]) == 2
        assert len(result["selected"]) == 1
        assert result["selected"][0].destination_id == "F3"
        assert result["bf_result"]["optimal_value"] < 0


# ---------------------------------------------------------------------------
# Test 3 — Chunk larger than shortage (over-allocation guard)
# ---------------------------------------------------------------------------

class TestChunkLargerThanShortage:
    def test_overallocation_guard_discourages_massive_overshoot(self):
        """Chunk qty=100, F2 needs only 20. usable=100.
        ratio=100/20=5.0 → P2 penalty dominates.
        cap_ratio=20/100=0.2 → reward is capped.
        Optimizer correctly declines: H(select) > 0 > H(skip).
        """
        chunks = [_chunk(idx=0, src="F1", drug="d1", qty=100,
                         expiry=30, val_risk=1000.0)]
        drugs = [_drug(code="d1", criticality=2)]
        routes = [_route(src="F1", dst="F2", transit=1, cost=200.0)]
        rates = [_usage(fac="F2", drug="d1", rate=10.0)]
        needs = {"F2|d1": 20}

        _, snap, data = _make_scenario(chunks, drugs, routes, rates, needs)
        result = _run_pipeline(chunks, snap, data)

        assert len(result["variables"]) == 1
        # P2 penalty for ratio=5.0 should dominate → optimizer skips
        assert len(result["selected"]) == 0
        assert result["bf_result"]["optimal_value"] == pytest.approx(0.0)

    def test_moderate_overshoot_still_selected(self):
        """Chunk usable=26, need=20. ratio=1.3 — moderate overshoot.
        Reward and P2 benefit should outweigh the small overshoot penalty.
        """
        chunks = [_chunk(idx=0, src="F1", drug="d1", qty=30,
                         expiry=30, val_risk=600.0)]
        drugs = [_drug(code="d1", criticality=2)]
        routes = [_route(src="F1", dst="F2", transit=1, cost=100.0)]
        rates = [_usage(fac="F2", drug="d1", rate=1.0)]
        needs = {"F2|d1": 20}
        stock = {"F1|d1": 1000}
        reserves = [_reserve(fac="F1", drug="d1", min_stock=50)]

        _, snap, data = _make_scenario(
            chunks, drugs, routes, rates, needs,
            reserves=reserves, stock=stock,
        )
        result = _run_pipeline(chunks, snap, data)

        # usable = min(30, floor(1×26)) = 26, ratio=26/20=1.3
        assert len(result["variables"]) == 1
        v = result["variables"][0]
        assert v.usable_quantity == 26
        # Moderate overshoot: P2 term (1.3-1)² is small → should select
        assert len(result["selected"]) == 1
        assert result["bf_result"]["optimal_value"] < 0


# ---------------------------------------------------------------------------
# Test 4 — Multiple chunks, FEFO ordering
# ---------------------------------------------------------------------------

class TestMultipleChunksFEFO:
    def test_fefo_selects_earliest_expiry(self):
        """3 chunks of d1: L1(10d), L2(30d), L3(60d). Need=40.
        Each usable=40 (rate high enough). Identical rewards.
        Need only allows one (P2 perfect fill at ratio=1.0 for one).
        FEFO bonus: L1 > L2 > L3 → optimizer picks L1.
        """
        chunks = [
            _chunk(idx=0, lot_id="L1", src="F1", qty=40, expiry=10, val_risk=400.0),
            _chunk(idx=1, lot_id="L2", src="F1", qty=40, expiry=30, val_risk=400.0),
            _chunk(idx=2, lot_id="L3", src="F1", qty=40, expiry=60, val_risk=400.0),
        ]
        drugs = [_drug(code="d1", criticality=2)]
        routes = [_route(src="F1", dst="F2", transit=1, cost=100.0)]
        rates = [_usage(fac="F2", drug="d1", rate=10.0)]
        needs = {"F2|d1": 40}
        stock = {"F1|d1": 1000}
        reserves = [_reserve(fac="F1", drug="d1", min_stock=50)]

        _, snap, data = _make_scenario(
            chunks, drugs, routes, rates, needs,
            reserves=reserves, stock=stock,
        )
        result = _run_pipeline(chunks, snap, data)

        # Verify usable for each:
        # L1: min(40, floor(10×(10-1-3))) = min(40, 60) = 40
        # L2: min(40, floor(10×(30-1-3))) = min(40, 260) = 40
        # L3: min(40, floor(10×(60-1-3))) = min(40, 560) = 40
        assert len(result["variables"]) == 3

        # P2 ratio=1.0 each. One gives (1-1)²=0 (perfect).
        # Two gives (2-1)²=1 (penalty). So select exactly one.
        assert len(result["selected"]) == 1
        assert result["selected"][0].lot_id == "L1"  # earliest expiry
        assert result["selected"][0].days_to_expiry == 10


# ---------------------------------------------------------------------------
# Test 5 — Cold-chain drug on non-refrigerated route
# ---------------------------------------------------------------------------

class TestColdChainFilter:
    def test_cold_drug_non_cold_route_blocked(self):
        """Cold drug d2, route has_cold_transport=False.
        Eligibility blocks it → 0 variables → empty QUBO.
        """
        chunks = [_chunk(idx=0, src="F1", drug="d2", qty=100,
                         expiry=20, val_risk=1000.0)]
        drugs = [_drug(code="d2", cold=True, criticality=3)]
        routes = [_route(src="F1", dst="F2", transit=1, cold=False)]
        rates = [_usage(fac="F2", drug="d2", rate=10.0)]
        needs = {"F2|d2": 50}

        _, snap, data = _make_scenario(chunks, drugs, routes, rates, needs)
        result = _run_pipeline(chunks, snap, data)

        assert result["variables"] == []
        assert result["Q"] == {}


# ---------------------------------------------------------------------------
# Test 6 — Medicine expires before consumption at destination
# ---------------------------------------------------------------------------

class TestExpiresBeforeConsumption:
    def test_too_short_expiry_blocked(self):
        """Chunk expiry=5, transit=3, buffer=3 → effective=-1.
        Eligibility blocks → 0 variables.
        """
        chunks = [_chunk(idx=0, src="F1", drug="d1", qty=100,
                         expiry=5, val_risk=500.0)]
        drugs = [_drug(code="d1")]
        routes = [_route(src="F1", dst="F2", transit=3)]
        rates = [_usage(fac="F2", drug="d1", rate=10.0)]
        needs = {"F2|d1": 50}

        _, snap, data = _make_scenario(chunks, drugs, routes, rates, needs)
        result = _run_pipeline(chunks, snap, data)

        assert result["variables"] == []
        assert result["Q"] == {}


# ---------------------------------------------------------------------------
# Test 7 — Two chunks, destination needs only one
# ---------------------------------------------------------------------------

class TestTwoChunksNeedOne:
    def test_optimizer_selects_exactly_one(self):
        """Two chunks L1 and L2, both usable=40, F2 needs 40.
        ratio=1.0 each. Selecting one: (1-1)²=0 → perfect fill.
        Selecting both: (2-1)²=1 → P2 penalty.
        P2×crit = 3.0 > reward ≈ 1.0 → one is better than both.
        """
        chunks = [
            _chunk(idx=0, lot_id="L1", src="F1", qty=40, expiry=20,
                   val_risk=400.0),
            _chunk(idx=1, lot_id="L2", src="F1", qty=40, expiry=25,
                   val_risk=400.0),
        ]
        drugs = [_drug(code="d1", criticality=2)]
        routes = [_route(src="F1", dst="F2", transit=1, cost=100.0)]
        rates = [_usage(fac="F2", drug="d1", rate=10.0)]
        needs = {"F2|d1": 40}
        stock = {"F1|d1": 1000}
        reserves = [_reserve(fac="F1", drug="d1", min_stock=50)]

        _, snap, data = _make_scenario(
            chunks, drugs, routes, rates, needs,
            reserves=reserves, stock=stock,
        )
        result = _run_pipeline(chunks, snap, data)

        assert len(result["variables"]) == 2
        assert len(result["selected"]) == 1
        # L1 has higher FEFO bonus (expiry=20 < 25) → preferred
        assert result["selected"][0].lot_id == "L1"

        # Verify: selecting one is strictly better than selecting both
        bf = result["bf_result"]
        h_one = bf["optimal_value"]
        # Find H for selecting both
        both_bits = {v.var_id: 1 for v in result["variables"]}
        h_both = compute_objective_value(both_bits, result["Q"])
        assert h_one < h_both


# ---------------------------------------------------------------------------
# Test 8 — FEFO preference between equal-reward lots
# ---------------------------------------------------------------------------

class TestFEFOPreference:
    def test_fefo_prefers_earlier_expiry(self):
        """Two lots, identical val_risk/cost, expiry=10 vs 60.
        Need allows one. FEFO bonus makes 10-day lot preferred.
        """
        chunks = [
            _chunk(idx=0, lot_id="L_NEAR", src="F1", qty=50,
                   expiry=10, val_risk=500.0),
            _chunk(idx=1, lot_id="L_FAR", src="F1", qty=50,
                   expiry=60, val_risk=500.0),
        ]
        drugs = [_drug(code="d1", criticality=2)]
        routes = [_route(src="F1", dst="F2", transit=1, cost=100.0)]
        rates = [_usage(fac="F2", drug="d1", rate=10.0)]
        needs = {"F2|d1": 50}
        stock = {"F1|d1": 1000}
        reserves = [_reserve(fac="F1", drug="d1", min_stock=50)]

        _, snap, data = _make_scenario(
            chunks, drugs, routes, rates, needs,
            reserves=reserves, stock=stock,
        )
        result = _run_pipeline(chunks, snap, data)

        # Both produce usable ≤ 50. P2 allows only one.
        assert len(result["selected"]) == 1
        assert result["selected"][0].lot_id == "L_NEAR"
        assert result["selected"][0].days_to_expiry == 10

        # Verify FEFO bonus difference in Q matrix
        Q = result["Q"]
        d_near = Q[(result["variables"][0].var_id, result["variables"][0].var_id)]
        d_far = Q[(result["variables"][1].var_id, result["variables"][1].var_id)]
        assert d_near < d_far  # more negative → more attractive


# ---------------------------------------------------------------------------
# QUBO vs Greedy comparison
# ---------------------------------------------------------------------------

class TestQUBOvsGreedy:
    def test_qubo_at_least_as_good_as_greedy(self):
        """On a small instance, QUBO optimal should be equal or better
        (lower H) than the greedy solution.
        """
        chunks = [
            _chunk(idx=0, lot_id="L1", src="F1", qty=40, expiry=10,
                   val_risk=400.0),
            _chunk(idx=1, lot_id="L2", src="F1", qty=40, expiry=30,
                   val_risk=400.0),
            _chunk(idx=2, lot_id="L3", src="F2", qty=60, expiry=15,
                   val_risk=600.0),
        ]
        drugs = [_drug(code="d1", criticality=2)]
        routes = [
            _route(src="F1", dst="F3", transit=1, cost=200.0),
            _route(src="F2", dst="F3", transit=2, cost=400.0),
        ]
        rates = [_usage(fac="F3", drug="d1", rate=8.0)]
        needs = {"F3|d1": 60}
        stock = {"F1|d1": 1000, "F2|d1": 1000}
        reserves = [
            _reserve(fac="F1", drug="d1", min_stock=50),
            _reserve(fac="F2", drug="d1", min_stock=50),
        ]

        _, snap, data = _make_scenario(
            chunks, drugs, routes, rates, needs,
            reserves=reserves, stock=stock,
        )
        variables = create_transfer_variables(chunks, data)
        Q = build_q_matrix(variables, snap, data)

        # Brute force optimal
        bf = brute_force_solve(Q, variables)
        h_qubo = bf["optimal_value"]

        # Greedy
        greedy_sel = greedy_fefo_allocate(variables, snap, data)
        greedy_bits = {v.var_id: 0 for v in variables}
        for v in greedy_sel:
            greedy_bits[v.var_id] = 1
        h_greedy = compute_objective_value(greedy_bits, Q)

        assert h_qubo <= h_greedy + 1e-9


# ---------------------------------------------------------------------------
# Validated plan from brute force
# ---------------------------------------------------------------------------

class TestBruteForceValidation:
    def test_optimal_plan_passes_validation(self):
        """The brute-force optimal plan on a well-formed instance must
        pass validation (no violations).
        """
        chunks = [
            _chunk(idx=0, lot_id="L1", src="F1", qty=50, expiry=15,
                   val_risk=500.0),
            _chunk(idx=1, lot_id="L2", src="F1", qty=30, expiry=25,
                   val_risk=300.0),
        ]
        drugs = [_drug(code="d1", criticality=2)]
        routes = [
            _route(src="F1", dst="F2", transit=1, cost=150.0),
            _route(src="F1", dst="F3", transit=2, cost=250.0),
        ]
        rates = [
            _usage(fac="F2", drug="d1", rate=5.0),
            _usage(fac="F3", drug="d1", rate=4.0),
        ]
        needs = {"F2|d1": 50, "F3|d1": 30}
        stock = {"F1|d1": 1000}
        reserves = [_reserve(fac="F1", drug="d1", min_stock=50)]

        _, snap, data = _make_scenario(
            chunks, drugs, routes, rates, needs,
            reserves=reserves, stock=stock,
        )
        result = _run_pipeline(chunks, snap, data)

        if result["selected"]:
            assert result["validation"].is_valid is True, (
                f"Violations: {result['validation'].violations}"
            )
