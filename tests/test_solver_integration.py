"""Integration tests for the MedBalanceOptSolver.

Mocks the QpiAI Opt SDK to verify:
  - String→int key conversion
  - SDK result decoding back to var_id bitstrings
  - Objective recomputation (don't trust SDK energy)
  - Evidence logging to disk
  - Graceful fallback (returns None) on SDK failure
  - Brute-force comparison on tiny instances
"""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from itertools import product
from unittest.mock import MagicMock, patch

import pytest

from backend.schemas import (
    Drug, DrugTier, InventoryData, InventorySnapshot,
    Route, SurplusChunk, TransferVariable, UsageRate,
)
from backend.eligibility import create_transfer_variables
from backend.qubo_builder import (
    build_q_matrix, compute_objective_value, decode_bitstring,
)
from backend.solver import MedBalanceOptSolver, OptResult


# ---------------------------------------------------------------------------
# Helpers
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


def _tv(var_id="x_0_F2", idx=0, lot="L1", src="F1", dst="F2", drug="d1",
        chunk_qty=100, usable=50, expiry=30, transit=1,
        val_risk=500.0, cost=300.0):
    return TransferVariable(
        var_id=var_id, chunk_index=idx, lot_id=lot, source_id=src,
        destination_id=dst, drug_code=drug, chunk_quantity=chunk_qty,
        usable_quantity=usable, days_to_expiry=expiry, transit_days=transit,
        value_at_risk=val_risk, transport_cost=cost,
    )


def _make_scenario(chunks, drugs, routes, usage_rates, needs,
                   stock=None):
    route_map = {f"{r.source_id}|{r.destination_id}": r for r in routes}
    rates_dict = {f"{u.facility_id}|{u.drug_code}": u for u in usage_rates}
    drugs_dict = {d.code: d for d in drugs}
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
        reserves={},
        route_map=route_map,
        stock_by_facility_drug=stock_dict,
        needs=needs,
    )
    return snapshot, data


@dataclass
class FakeSolveResult:
    """Mimics the SDK's SolveResult for testing."""
    objective: float = 0.0
    solution: list[int] = field(default_factory=list)
    variable_assignment: dict[str, int] = field(default_factory=dict)
    solver_time_s: float = 0.01
    compute_time_s: float = 0.01
    job_id: str = "test-job"
    problem_id: str = "test-prob"


def brute_force_solve(Q, variables):
    var_ids = [v.var_id for v in variables]
    n = len(var_ids)
    best_bits = {v: 0 for v in var_ids}
    best_h = 0.0

    for mask in range(2 ** n):
        bits = {}
        for i, vid in enumerate(var_ids):
            bits[vid] = (mask >> i) & 1
        h = compute_objective_value(bits, Q)
        if h < best_h:
            best_h = h
            best_bits = dict(bits)

    return best_bits, best_h


# ---------------------------------------------------------------------------
# Test: Q-matrix key conversion (string → int)
# ---------------------------------------------------------------------------

class TestConvertQ:
    def test_converts_string_keys_to_int(self):
        Q = {("x_0_F2", "x_0_F2"): -0.5, ("x_0_F2", "x_1_F3"): 0.3}
        names = ["x_0_F2", "x_1_F3"]
        name_to_idx = {n: i for i, n in enumerate(names)}
        result = MedBalanceOptSolver._convert_q(Q, name_to_idx)
        assert result[(0, 0)] == pytest.approx(-0.5)
        assert result[(0, 1)] == pytest.approx(0.3)

    def test_canonical_ordering(self):
        Q = {("b", "a"): 1.0}
        name_to_idx = {"a": 0, "b": 1}
        result = MedBalanceOptSolver._convert_q(Q, name_to_idx)
        assert (0, 1) in result
        assert (1, 0) not in result

    def test_drops_near_zero(self):
        Q = {("a", "a"): 1e-16}
        result = MedBalanceOptSolver._convert_q(Q, {"a": 0})
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Test: SDK result decoding
# ---------------------------------------------------------------------------

class TestDecodeSDKResult:
    def test_decode_via_variable_assignment(self):
        sdk = FakeSolveResult(
            variable_assignment={"x_0_F2": 1, "x_1_F3": 0},
        )
        bits = MedBalanceOptSolver._decode_sdk_result(sdk, ["x_0_F2", "x_1_F3"])
        assert bits == {"x_0_F2": 1, "x_1_F3": 0}

    def test_decode_via_solution_list(self):
        sdk = FakeSolveResult(
            variable_assignment={},
            solution=[1, 0, 1],
        )
        names = ["a", "b", "c"]
        bits = MedBalanceOptSolver._decode_sdk_result(sdk, names)
        assert bits == {"a": 1, "b": 0, "c": 1}


# ---------------------------------------------------------------------------
# Test: Objective recomputation
# ---------------------------------------------------------------------------

class TestObjectiveRecomputation:
    def test_solver_recomputes_with_our_function(self):
        """The solver must use our compute_objective_value, not SDK energy."""
        v1 = _tv(var_id="x_0_F2", usable=50, val_risk=500, cost=100)
        v2 = _tv(var_id="x_1_F3", idx=1, lot="L2", dst="F3",
                 usable=40, val_risk=400, cost=80)

        Q = {("x_0_F2", "x_0_F2"): -0.8, ("x_1_F3", "x_1_F3"): -0.6}
        names = ["x_0_F2", "x_1_F3"]

        fake_result = FakeSolveResult(
            objective=999.0,  # deliberately wrong SDK energy
            variable_assignment={"x_0_F2": 1, "x_1_F3": 1},
        )

        solver = MedBalanceOptSolver(
            api_key="test-key",
            evidence_dir=tempfile.mkdtemp(),
        )

        with patch.object(solver, "_solve_local", return_value=fake_result):
            with patch.object(solver, "_ensure_local"):
                result = solver.solve(Q, names, num_reads=1)

        expected_h = compute_objective_value(
            {"x_0_F2": 1, "x_1_F3": 1}, Q
        )
        assert result is not None
        assert result.objective_value == pytest.approx(expected_h)
        assert result.objective_value != 999.0


# ---------------------------------------------------------------------------
# Test: Evidence logging
# ---------------------------------------------------------------------------

class TestEvidenceLogging:
    def test_creates_evidence_files(self):
        tmpdir = tempfile.mkdtemp()
        try:
            Q = {("a", "a"): -1.0}
            names = ["a"]
            fake = FakeSolveResult(
                variable_assignment={"a": 1},
            )

            solver = MedBalanceOptSolver(api_key="test-key", evidence_dir=tmpdir)
            with patch.object(solver, "_solve_local", return_value=fake):
                with patch.object(solver, "_ensure_local"):
                    result = solver.solve(Q, names, num_reads=1)

            assert result is not None
            files = os.listdir(tmpdir)
            q_files = [f for f in files if f.endswith(".q.json")]
            res_files = [f for f in files if f.endswith(".result.json")]
            assert len(q_files) == 1
            assert len(res_files) == 1

            q_data = json.loads(open(os.path.join(tmpdir, q_files[0])).read())
            assert "variable_names" in q_data
            assert "Q" in q_data

            res_data = json.loads(open(os.path.join(tmpdir, res_files[0])).read())
            assert "best" in res_data
            assert "objective_value" in res_data
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test: Graceful failure → None
# ---------------------------------------------------------------------------

class TestGracefulFailure:
    def test_returns_none_on_sdk_error(self):
        solver = MedBalanceOptSolver(api_key="test-key",
                                     evidence_dir=tempfile.mkdtemp())
        with patch.object(solver, "_solve_local", side_effect=RuntimeError("boom")):
            with patch.object(solver, "_ensure_local"):
                result = solver.solve(
                    {("a", "a"): -1.0}, ["a"], num_reads=1
                )
        assert result is None

    def test_returns_none_when_no_api_key(self):
        solver = MedBalanceOptSolver(api_key=None,
                                     evidence_dir=tempfile.mkdtemp())
        solver._api_key = None
        result = solver.solve({("a", "a"): -1.0}, ["a"], num_reads=1)
        assert result is None


# ---------------------------------------------------------------------------
# Test: Solver vs brute-force on tiny scenario
# ---------------------------------------------------------------------------

class TestSolverVsBruteForce:
    def test_one_chunk_two_destinations(self):
        """F1 has L1 (100 units, 20 days). F2 needs 30, F3 needs 80.
        Brute-force picks F3. Mock solver to return brute-force solution.
        Verify objectives match.
        """
        chunks = [_chunk(idx=0, src="F1", drug="d1", qty=100,
                         expiry=20, val_risk=1000)]
        drugs = [_drug(code="d1")]
        routes = [_route(src="F1", dst="F2", cost=300),
                  _route(src="F1", dst="F3", cost=300)]
        rates = [_usage(fac="F2", drug="d1", rate=10),
                 _usage(fac="F3", drug="d1", rate=10)]
        needs = {"F2|d1": 30, "F3|d1": 80}

        snap, data = _make_scenario(chunks, drugs, routes, rates, needs)
        variables = create_transfer_variables(chunks, data)
        Q = build_q_matrix(variables, snap, data)

        bf_bits, bf_h = brute_force_solve(Q, variables)
        selected = decode_bitstring(bf_bits, variables)
        assert len(selected) == 1
        assert selected[0].destination_id == "F3"

        var_names = [v.var_id for v in variables]
        fake = FakeSolveResult(
            variable_assignment=bf_bits,
        )

        solver = MedBalanceOptSolver(
            api_key="test-key",
            evidence_dir=tempfile.mkdtemp(),
        )
        with patch.object(solver, "_solve_local", return_value=fake):
            with patch.object(solver, "_ensure_local"):
                result = solver.solve(Q, var_names, num_reads=1)

        assert result is not None
        assert result.objective_value == pytest.approx(bf_h)
        assert result.best == bf_bits

    def test_fefo_preference(self):
        """Two lots: L1 (10 days), L2 (60 days). Both → F2.
        Brute-force should prefer L1. Solver should agree.
        """
        chunks = [
            _chunk(idx=0, lot_id="L1", src="F1", drug="d1", qty=50,
                   expiry=10, val_risk=500),
            _chunk(idx=1, lot_id="L2", src="F1", drug="d1", qty=50,
                   expiry=60, val_risk=500),
        ]
        drugs = [_drug(code="d1")]
        routes = [_route(src="F1", dst="F2", cost=200)]
        rates = [_usage(fac="F2", drug="d1", rate=5)]
        needs = {"F2|d1": 50}

        snap, data = _make_scenario(chunks, drugs, routes, rates, needs)
        variables = create_transfer_variables(chunks, data)
        Q = build_q_matrix(variables, snap, data)

        bf_bits, bf_h = brute_force_solve(Q, variables)
        selected = decode_bitstring(bf_bits, variables)
        lot_ids = {s.lot_id for s in selected}
        assert "L1" in lot_ids

        var_names = [v.var_id for v in variables]
        fake = FakeSolveResult(variable_assignment=bf_bits)

        solver = MedBalanceOptSolver(
            api_key="test-key",
            evidence_dir=tempfile.mkdtemp(),
        )
        with patch.object(solver, "_solve_local", return_value=fake):
            with patch.object(solver, "_ensure_local"):
                result = solver.solve(Q, var_names, num_reads=1)

        assert result is not None
        assert result.objective_value == pytest.approx(bf_h)


# ---------------------------------------------------------------------------
# Test: OptResult structure
# ---------------------------------------------------------------------------

class TestOptResultStructure:
    def test_fields_present(self):
        r = OptResult(
            best={"a": 1}, top_k=[{"a": 1}],
            objective_value=-1.0, wall_clock_s=0.5,
            raw_response={"foo": "bar"},
        )
        assert r.best == {"a": 1}
        assert len(r.top_k) == 1
        assert r.objective_value == -1.0
        assert r.wall_clock_s == 0.5
        assert r.raw_response == {"foo": "bar"}

    def test_default_raw_response(self):
        r = OptResult(best={}, top_k=[], objective_value=0.0, wall_clock_s=0.0)
        assert r.raw_response == {}
