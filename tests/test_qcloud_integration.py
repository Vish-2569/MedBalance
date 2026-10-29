"""Integration tests for QCloud QAOA module.

Tests Max-Cut scoring, k-means clustering, cluster comparison,
QUBO-to-Ising conversion, and balance checks — all without needing
QCloud auth (pure logic tests + mocked QAOA).
"""

import json
import shutil
import tempfile
from itertools import product
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest

from backend.schemas import (
    Drug, DrugTier, Facility, FacilityType, Route,
)
from backend.qcloud import (
    MaxCutResult,
    QAOAResult,
    build_facility_graph,
    check_cluster_balance,
    compare_clusterings,
    kmeans_clustering,
    log_qcloud_evidence,
    _bitstring_to_clusters,
    _intra_cluster_cost,
    _qubo_to_ising,
    _score_bitstring_maxcut,
)
from backend.qubo_builder import compute_objective_value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fac(fid, lat=0.0, lon=0.0, cold=False):
    return Facility(id=fid, name=fid, type=FacilityType.phc,
                    latitude=lat, longitude=lon, has_cold_storage=cold)


def _route(src, dst, transit=1, cost=500.0, cold=False):
    return Route(source_id=src, destination_id=dst, transit_days=transit,
                 trip_cost_inr=cost, has_cold_transport=cold)


# ---------------------------------------------------------------------------
# Test: build_facility_graph
# ---------------------------------------------------------------------------

class TestBuildFacilityGraph:
    def test_basic_graph(self):
        facs = [_fac("F1"), _fac("F2"), _fac("F3")]
        routes = [_route("F1", "F2", transit=2), _route("F2", "F3", transit=3)]
        G = build_facility_graph(facs, routes)
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2
        assert G["F1"]["F2"]["weight"] == 2
        assert G["F2"]["F3"]["weight"] == 3

    def test_takes_minimum_weight(self):
        facs = [_fac("A"), _fac("B")]
        routes = [_route("A", "B", transit=5), _route("A", "B", transit=2)]
        G = build_facility_graph(facs, routes)
        assert G["A"]["B"]["weight"] == 2

    def test_ignores_self_loops(self):
        facs = [_fac("A")]
        routes = [_route("A", "A", transit=0)]
        G = build_facility_graph(facs, routes)
        assert G.number_of_edges() == 0

    def test_empty(self):
        G = build_facility_graph([], [])
        assert G.number_of_nodes() == 0


# ---------------------------------------------------------------------------
# Test: Max-Cut scoring
# ---------------------------------------------------------------------------

class TestMaxCutScoring:
    def _square_graph(self):
        """4-node cycle: 0-1-2-3-0, all weight 1."""
        G = nx.Graph()
        G.add_nodes_from(["A", "B", "C", "D"])
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("B", "C", weight=1.0)
        G.add_edge("C", "D", weight=1.0)
        G.add_edge("D", "A", weight=1.0)
        return G

    def test_optimal_cut_on_square(self):
        """Optimal Max-Cut of a 4-cycle is 4 (alternating partition)."""
        G = self._square_graph()
        nodes = list(G.nodes())
        best_cut = 0.0
        best_bs = ""
        for mask in range(2 ** len(nodes)):
            bs = format(mask, f"0{len(nodes)}b")
            cut = _score_bitstring_maxcut(bs, G)
            if cut > best_cut:
                best_cut = cut
                best_bs = bs
        assert best_cut == 4.0

    def test_all_same_partition_zero_cut(self):
        G = self._square_graph()
        assert _score_bitstring_maxcut("0000", G) == 0.0
        assert _score_bitstring_maxcut("1111", G) == 0.0

    def test_bitstring_to_clusters(self):
        G = nx.Graph()
        G.add_nodes_from(["A", "B", "C"])
        clusters = _bitstring_to_clusters("101", G)
        assert sorted(clusters[0]) == ["B"]
        assert sorted(clusters[1]) == ["A", "C"]


# ---------------------------------------------------------------------------
# Test: k-means clustering
# ---------------------------------------------------------------------------

class TestKMeansClustering:
    def test_two_obvious_clusters(self):
        """Two well-separated groups should be correctly split."""
        facs = [
            _fac("N1", lat=20.0, lon=80.0),
            _fac("N2", lat=20.5, lon=80.5),
            _fac("S1", lat=10.0, lon=70.0),
            _fac("S2", lat=10.5, lon=70.5),
        ]
        clusters = kmeans_clustering(facs, n_clusters=2)
        assert len(clusters) == 2
        ids_0 = set(clusters[0])
        ids_1 = set(clusters[1])
        north = {"N1", "N2"}
        south = {"S1", "S2"}
        assert (ids_0 == north and ids_1 == south) or \
               (ids_0 == south and ids_1 == north)

    def test_single_facility(self):
        facs = [_fac("X", lat=0.0, lon=0.0)]
        clusters = kmeans_clustering(facs, n_clusters=2)
        all_ids = [fid for c in clusters for fid in c]
        assert all_ids == ["X"]

    def test_empty_facilities(self):
        clusters = kmeans_clustering([], n_clusters=2)
        assert clusters == [[], []]

    def test_deterministic(self):
        facs = [_fac(f"F{i}", lat=float(i), lon=float(i)) for i in range(6)]
        c1 = kmeans_clustering(facs, n_clusters=2)
        c2 = kmeans_clustering(facs, n_clusters=2)
        assert c1 == c2


# ---------------------------------------------------------------------------
# Test: compare_clusterings
# ---------------------------------------------------------------------------

class TestCompareClusterings:
    def test_kmeans_wins_when_qcloud_unavailable(self):
        route_map = {}
        result = compare_clusterings(None, [["A"], ["B"]], route_map)
        assert result["winner"] == "kmeans"
        assert result["reason"] == "qcloud_unavailable"

    def test_lower_cost_wins(self):
        route_map = {
            "A|B": _route("A", "B", cost=100),
            "B|A": _route("B", "A", cost=100),
        }
        qc_clusters = [["A", "B"], []]
        km_clusters = [["A"], ["B"]]
        result = compare_clusterings(qc_clusters, km_clusters, route_map)
        assert result["winner"] == "kmeans"
        assert result["kmeans_cost"] == 0.0
        assert result["qcloud_cost"] == 200.0


# ---------------------------------------------------------------------------
# Test: cluster balance
# ---------------------------------------------------------------------------

class TestClusterBalance:
    def test_balanced(self):
        assert check_cluster_balance([["A", "B"], ["C", "D"]]) is True

    def test_imbalanced(self):
        assert check_cluster_balance([["A", "B", "C", "D"], ["E"]]) is False

    def test_empty(self):
        assert check_cluster_balance([[], []]) is True

    def test_threshold(self):
        assert check_cluster_balance([["A", "B", "C"], ["D"]], threshold=0.80) is True
        assert check_cluster_balance([["A", "B", "C"], ["D"]], threshold=0.70) is False


# ---------------------------------------------------------------------------
# Test: QUBO → Ising conversion
# ---------------------------------------------------------------------------

class TestQUBOToIsing:
    def test_single_variable(self):
        """QUBO: H = -1 * x0. Ising: x0 = (1-Z0)/2 → H = -1*(1-Z0)/2 = -0.5 + 0.5*Z0."""
        Q = {("a", "a"): -1.0}
        terms = _qubo_to_ising(Q, ["a"])
        coeff_by_ops = {}
        for ops, c in terms:
            key = tuple(ops)
            coeff_by_ops[key] = coeff_by_ops.get(key, 0.0) + c

        assert coeff_by_ops[()] == pytest.approx(-0.5)
        assert coeff_by_ops[((0, "Z"),)] == pytest.approx(0.5)

    def test_two_variable_interaction(self):
        """QUBO: H = 2*x0*x1. Ising substitution should give ZZ, Z0, Z1, const terms."""
        Q = {("a", "b"): 2.0}
        terms = _qubo_to_ising(Q, ["a", "b"])
        coeff_by_ops = {}
        for ops, c in terms:
            key = tuple(sorted(ops))
            coeff_by_ops[key] = coeff_by_ops.get(key, 0.0) + c

        assert coeff_by_ops[()] == pytest.approx(0.5)
        assert coeff_by_ops[((0, "Z"),)] == pytest.approx(-0.5)
        assert coeff_by_ops[((1, "Z"),)] == pytest.approx(-0.5)
        assert coeff_by_ops[((0, "Z"), (1, "Z"))] == pytest.approx(0.5)

    def test_ising_recovers_qubo_energy(self):
        """Verify Ising energy matches QUBO energy for all bitstrings."""
        Q = {("a", "a"): -0.8, ("b", "b"): -0.6, ("a", "b"): 1.0}
        names = ["a", "b"]
        terms = _qubo_to_ising(Q, names)

        for xa, xb in product([0, 1], repeat=2):
            bits = {"a": xa, "b": xb}
            qubo_e = compute_objective_value(bits, Q)

            za = 1 - 2 * xa
            zb = 1 - 2 * xb
            ising_e = 0.0
            for ops, c in terms:
                val = c
                for qubit, pauli in ops:
                    z = za if qubit == 0 else zb
                    val *= z
                ising_e += val

            assert ising_e == pytest.approx(qubo_e, abs=1e-10), \
                f"xa={xa}, xb={xb}: qubo={qubo_e}, ising={ising_e}"


# ---------------------------------------------------------------------------
# Test: Evidence logging
# ---------------------------------------------------------------------------

class TestEvidenceLogging:
    def test_saves_maxcut_result(self):
        tmpdir = tempfile.mkdtemp()
        try:
            result = MaxCutResult(
                clusters=[["A"], ["B"]],
                best_bitstring="01",
                cut_value=1.0,
                backend="test",
                shots=100,
            )
            path = log_qcloud_evidence("test_mc", result, evidence_dir=tmpdir)
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["cut_value"] == 1.0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test: Max-Cut brute-force on 5-node graph
# ---------------------------------------------------------------------------

class TestMaxCutBruteForce5Node:
    def test_known_optimal_cut(self):
        """5-node path graph: A-B-C-D-E (weight 1 each).
        Optimal max-cut = 4 (e.g. partition {A,C,E} vs {B,D}).
        """
        G = nx.Graph()
        for n in ["A", "B", "C", "D", "E"]:
            G.add_node(n)
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("B", "C", weight=1.0)
        G.add_edge("C", "D", weight=1.0)
        G.add_edge("D", "E", weight=1.0)

        nodes = list(G.nodes())
        best_cut = 0.0
        best_bs = ""
        for mask in range(2 ** len(nodes)):
            bs = format(mask, f"0{len(nodes)}b")
            cut = _score_bitstring_maxcut(bs, G)
            if cut > best_cut:
                best_cut = cut
                best_bs = bs

        assert best_cut == 4.0
        clusters = _bitstring_to_clusters(best_bs, G)
        assert len(clusters[0]) + len(clusters[1]) == 5

    def test_complete_graph_k5(self):
        """K5: 10 edges. Optimal max-cut of K5 = ceil(5*4/4) = 6
        (partition sizes 2 and 3: 2*3 = 6 cross edges).
        """
        G = nx.complete_graph(5)
        nx.set_edge_attributes(G, 1.0, "weight")
        nodes = list(G.nodes())

        best_cut = 0.0
        for mask in range(2 ** 5):
            bs = format(mask, f"05b")
            cut = _score_bitstring_maxcut(bs, G)
            if cut > best_cut:
                best_cut = cut

        assert best_cut == 6.0


# ---------------------------------------------------------------------------
# Test: Small QUBO vs brute-force
# ---------------------------------------------------------------------------

class TestSmallQUBOBruteForce:
    def test_ising_conversion_preserves_optimum(self):
        """Build a tiny QUBO, find brute-force optimum, verify Ising
        Hamiltonian gives same energy ordering for all bitstrings.
        """
        Q = {
            ("x0", "x0"): -0.9,
            ("x1", "x1"): -0.7,
            ("x0", "x1"): 2.0,
        }
        names = ["x0", "x1"]

        bf_best = None
        bf_h = float("inf")
        for x0, x1 in product([0, 1], repeat=2):
            bits = {"x0": x0, "x1": x1}
            h = compute_objective_value(bits, Q)
            if h < bf_h:
                bf_h = h
                bf_best = bits

        assert bf_best["x0"] == 1 and bf_best["x1"] == 0

        terms = _qubo_to_ising(Q, names)
        for x0, x1 in product([0, 1], repeat=2):
            bits = {"x0": x0, "x1": x1}
            qubo_e = compute_objective_value(bits, Q)

            z0 = 1 - 2 * x0
            z1 = 1 - 2 * x1
            ising_e = 0.0
            for ops, c in terms:
                val = c
                for qubit, pauli in ops:
                    z = z0 if qubit == 0 else z1
                    val *= z
                ising_e += val

            assert ising_e == pytest.approx(qubo_e, abs=1e-10)
