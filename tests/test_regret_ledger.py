"""Tests for the Regret Ledger.

Verifies deduplication, ranking, feasibility tracking, diff computation,
query-by-transfer, serialisation, and persistence — using small QUBO
scenarios with known brute-force solutions.
"""

import json
import shutil
import tempfile
from itertools import product

import pytest

from backend.schemas import (
    Drug, DrugTier, InventoryData, InventorySnapshot,
    Route, SurplusChunk, TransferVariable, UsageRate,
)
from backend.eligibility import create_transfer_variables
from backend.qubo_builder import build_q_matrix, compute_objective_value, decode_bitstring
from backend.regret_ledger import (
    MAX_ALTERNATIVES,
    RegretEntry,
    RegretLedger,
    persist_ledger,
    _transfer_set_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drug(code="d1", price=10.0, cold=False, criticality=2):
    return Drug(code=code, name="T", molecule="M", strength="1mg",
                form="tablet", tier=DrugTier.essential, criticality=criticality,
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
        usable=50, expiry=30, transit=1, val_risk=500.0, cost=300.0):
    return TransferVariable(
        var_id=var_id, chunk_index=idx, lot_id=lot, source_id=src,
        destination_id=dst, drug_code=drug, chunk_quantity=100,
        usable_quantity=usable, days_to_expiry=expiry, transit_days=transit,
        value_at_risk=val_risk, transport_cost=cost,
    )


def _make_scenario(chunks, drugs, routes, usage_rates, needs, stock=None):
    route_map = {f"{r.source_id}|{r.destination_id}": r for r in routes}
    rates_dict = {f"{u.facility_id}|{u.drug_code}": u for u in usage_rates}
    drugs_dict = {d.code: d for d in drugs}
    stock_dict = stock or {}
    snapshot = InventorySnapshot(
        needs=needs, surplus_chunks=chunks, facility_stock=stock_dict,
    )
    data = InventoryData(
        facilities={}, drugs=drugs_dict, lots=[], routes=routes,
        usage_rates=rates_dict, reserves={}, route_map=route_map,
        stock_by_facility_drug=stock_dict, needs=needs,
    )
    return snapshot, data


def _brute_force_top_k(Q, variables, k=8):
    """Enumerate all bitstrings, return top-k by objective."""
    var_ids = [v.var_id for v in variables]
    n = len(var_ids)
    solutions = []
    for mask in range(2 ** n):
        bits = {}
        for i, vid in enumerate(var_ids):
            bits[vid] = (mask >> i) & 1
        h = compute_objective_value(bits, Q)
        solutions.append((h, bits))
    solutions.sort(key=lambda x: x[0])
    return [bits for _, bits in solutions[:k]]


# ---------------------------------------------------------------------------
# Test: basic recording
# ---------------------------------------------------------------------------

class TestRecordAlternatives:
    def _setup_scenario(self):
        """L1 (100 units, 20d expiry) → F2 (need=30) or F3 (need=80)."""
        chunks = [_chunk(idx=0, src="F1", drug="d1", qty=100, expiry=20, val_risk=1000)]
        drugs = [_drug(code="d1")]
        routes = [_route("F1", "F2", cost=300), _route("F1", "F3", cost=300)]
        rates = [_usage("F2", "d1", 10), _usage("F3", "d1", 10)]
        needs = {"F2|d1": 30, "F3|d1": 80}
        snap, data = _make_scenario(chunks, drugs, routes, rates, needs)
        variables = create_transfer_variables(chunks, data)
        Q = build_q_matrix(variables, snap, data)
        return Q, variables, snap, data

    def test_records_ranked_entries(self):
        Q, variables, snap, data = self._setup_scenario()
        top_k = _brute_force_top_k(Q, variables, k=4)

        ledger = RegretLedger()
        entries = ledger.record_alternatives(top_k, Q, variables, snap, data)

        assert len(entries) >= 1
        assert entries[0].plan_rank == 1
        assert entries[0].delta_from_best == 0.0
        for i in range(1, len(entries)):
            assert entries[i].plan_rank == i + 1
            assert entries[i].delta_from_best >= 0.0

    def test_chosen_plan_is_best_objective(self):
        Q, variables, snap, data = self._setup_scenario()
        top_k = _brute_force_top_k(Q, variables, k=4)

        ledger = RegretLedger()
        ledger.record_alternatives(top_k, Q, variables, snap, data)

        best_h = ledger.entries[0].objective_value
        for entry in ledger.entries[1:]:
            assert entry.objective_value >= best_h

    def test_deduplicates_identical_plans(self):
        Q, variables, snap, data = self._setup_scenario()
        bits = _brute_force_top_k(Q, variables, k=1)[0]
        duplicated = [bits, dict(bits), dict(bits)]

        ledger = RegretLedger()
        entries = ledger.record_alternatives(duplicated, Q, variables, snap, data)

        assert len(entries) == 1

    def test_caps_at_max_alternatives(self):
        """Two lots, two destinations → 4 vars → 16 bitstrings. Should cap at 5."""
        chunks = [
            _chunk(idx=0, lot_id="L1", src="F1", drug="d1", qty=50, expiry=20, val_risk=500),
            _chunk(idx=1, lot_id="L2", src="F1", drug="d1", qty=50, expiry=30, val_risk=500),
        ]
        drugs = [_drug(code="d1")]
        routes = [_route("F1", "F2", cost=200), _route("F1", "F3", cost=200)]
        rates = [_usage("F2", "d1", 5), _usage("F3", "d1", 5)]
        needs = {"F2|d1": 50, "F3|d1": 50}
        snap, data = _make_scenario(chunks, drugs, routes, rates, needs)
        variables = create_transfer_variables(chunks, data)
        Q = build_q_matrix(variables, snap, data)

        top_k = _brute_force_top_k(Q, variables, k=16)
        ledger = RegretLedger()
        entries = ledger.record_alternatives(top_k, Q, variables, snap, data)

        assert len(entries) <= MAX_ALTERNATIVES


# ---------------------------------------------------------------------------
# Test: diff from chosen
# ---------------------------------------------------------------------------

class TestDiffFromChosen:
    def test_chosen_has_no_diff(self):
        tv1 = _tv(var_id="x_0_F2", lot="L1", dst="F2")
        tv2 = _tv(var_id="x_0_F3", lot="L1", dst="F3")
        bits_chosen = {"x_0_F2": 1, "x_0_F3": 0}
        bits_alt = {"x_0_F2": 0, "x_0_F3": 1}

        chunks = [_chunk(idx=0, src="F1", drug="d1", qty=100, expiry=20, val_risk=1000)]
        drugs = [_drug(code="d1")]
        routes = [_route("F1", "F2", cost=300), _route("F1", "F3", cost=300)]
        rates = [_usage("F2", "d1", 10), _usage("F3", "d1", 10)]
        needs = {"F2|d1": 50, "F3|d1": 50}
        snap, data = _make_scenario(chunks, drugs, routes, rates, needs)
        variables = create_transfer_variables(chunks, data)
        Q = build_q_matrix(variables, snap, data)

        ledger = RegretLedger()
        ledger.record_alternatives([bits_chosen, bits_alt], Q, variables, snap, data)

        assert ledger.entries[0].diff_from_chosen == []
        if len(ledger.entries) > 1:
            assert len(ledger.entries[1].diff_from_chosen) > 0


# ---------------------------------------------------------------------------
# Test: get_alternatives_for_transfer
# ---------------------------------------------------------------------------

class TestGetAlternatives:
    def test_finds_alternatives(self):
        chunks = [_chunk(idx=0, src="F1", drug="d1", qty=100, expiry=20, val_risk=1000)]
        drugs = [_drug(code="d1")]
        routes = [_route("F1", "F2", cost=300), _route("F1", "F3", cost=300)]
        rates = [_usage("F2", "d1", 10), _usage("F3", "d1", 10)]
        needs = {"F2|d1": 50, "F3|d1": 80}
        snap, data = _make_scenario(chunks, drugs, routes, rates, needs)
        variables = create_transfer_variables(chunks, data)
        Q = build_q_matrix(variables, snap, data)

        top_k = _brute_force_top_k(Q, variables, k=4)
        ledger = RegretLedger()
        ledger.record_alternatives(top_k, Q, variables, snap, data)

        chosen_transfers = ledger.entries[0].transfers
        if chosen_transfers:
            alts = ledger.get_alternatives_for_transfer(chosen_transfers[0])
            assert isinstance(alts, list)

    def test_empty_ledger_returns_empty(self):
        ledger = RegretLedger()
        tv = _tv()
        assert ledger.get_alternatives_for_transfer(tv) == []


# ---------------------------------------------------------------------------
# Test: serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_to_json_valid(self):
        ledger = RegretLedger()
        ledger.entries = [
            RegretEntry(
                plan_rank=1,
                transfers=[_tv(var_id="x_0_F2")],
                objective_value=-1.5,
                delta_from_best=0.0,
                is_feasible=True,
            ),
            RegretEntry(
                plan_rank=2,
                transfers=[_tv(var_id="x_0_F3", dst="F3")],
                objective_value=-1.0,
                delta_from_best=0.5,
                is_feasible=True,
                diff_from_chosen=["x_0_F2", "x_0_F3"],
            ),
        ]
        raw = ledger.to_json()
        data = json.loads(raw)
        assert len(data) == 2
        assert data[0]["plan_rank"] == 1
        assert data[1]["delta_from_best"] == 0.5

    def test_empty_ledger_json(self):
        ledger = RegretLedger()
        assert json.loads(ledger.to_json()) == []


# ---------------------------------------------------------------------------
# Test: summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_format(self):
        ledger = RegretLedger()
        ledger.entries = [
            RegretEntry(1, [_tv()], -2.0, 0.0, True),
            RegretEntry(2, [_tv(var_id="x_1_F3", dst="F3")], -1.5, 0.5, True,
                        diff_from_chosen=["x_0_F2", "x_1_F3"]),
        ]
        s = ledger.summary()
        assert "Chosen plan (rank 1)" in s
        assert "Alternative 2" in s
        assert "delta=+0.500" in s

    def test_empty_summary(self):
        ledger = RegretLedger()
        assert "empty" in ledger.summary().lower()

    def test_infeasible_tag(self):
        ledger = RegretLedger()
        ledger.entries = [
            RegretEntry(1, [], -1.0, 0.0, True),
            RegretEntry(2, [], -0.5, 0.5, False,
                        blocking_constraints=["lot_double_assignment"]),
        ]
        s = ledger.summary()
        assert "INFEASIBLE" in s


# ---------------------------------------------------------------------------
# Test: persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_persist_creates_file(self):
        tmpdir = tempfile.mkdtemp()
        try:
            ledger = RegretLedger()
            ledger.entries = [
                RegretEntry(1, [], -1.0, 0.0, True),
            ]
            path = persist_ledger(ledger, "test_run", evidence_dir=tmpdir)
            assert path.exists()
            data = json.loads(path.read_text())
            assert len(data) == 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_persist_empty_ledger(self):
        tmpdir = tempfile.mkdtemp()
        try:
            ledger = RegretLedger()
            path = persist_ledger(ledger, "empty_run", evidence_dir=tmpdir)
            assert path.exists()
            assert json.loads(path.read_text()) == []
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test: transfer_set_key
# ---------------------------------------------------------------------------

class TestTransferSetKey:
    def test_order_independent(self):
        t1 = _tv(var_id="a")
        t2 = _tv(var_id="b")
        assert _transfer_set_key([t1, t2]) == _transfer_set_key([t2, t1])

    def test_different_sets_differ(self):
        t1 = _tv(var_id="a")
        t2 = _tv(var_id="b")
        assert _transfer_set_key([t1]) != _transfer_set_key([t2])
