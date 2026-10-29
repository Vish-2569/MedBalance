"""QpiAI QCloud integration: QAOA for Max-Cut clustering and small QUBO
verification.

SDK interface (qpiai-quantum 0.7.5):
  - QpiAIQuantumAuth: .login(api_key), .verify_api_key() → bool
  - MaxCutProblem(graph: nx.Graph): .decode_solution(bitstring) → dict,
    .compute_solution_quality(solution) → dict, .get_hamiltonian_terms()
  - QAOASolver(problem, layers, optimizer, max_iterations, verbose):
    .run(shots, device_name) → QAOAResult
  - QAOAResult: .optimal_energy, .bitstring, .counts, .solution,
    .quality_metrics, .is_valid, .metadata
  - Circuit: manual gate-level construction (h, rx, rz, rzz, cx, measure_all)
  - Backend: LOCAL_SIMULATOR, STATEVECTOR_SIMULATOR_CPU, INDUS_QPU
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
from dotenv import load_dotenv

from .schemas import Facility, Route

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MaxCutResult:
    clusters: list[list[str]]
    best_bitstring: str
    cut_value: float
    backend: str
    shots: int
    raw_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class QAOAResult:
    best_bitstring: dict[str, int]
    objective_value: float
    optimal_energy: float
    counts: dict[str, int] = field(default_factory=dict)
    backend: str = ""
    shots: int = 0
    comparison: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _load_qcloud_key() -> str | None:
    load_dotenv()
    key = os.environ.get("QPIAI_QCLOUD_API_KEY")
    if not key or key.startswith("PASTE"):
        return None
    return key


def authenticate(api_key: str | None = None) -> bool:
    key = api_key or _load_qcloud_key()
    if not key:
        logger.warning("No QPIAI_QCLOUD_API_KEY found")
        return False
    try:
        from qpiai_quantum import QpiAIQuantumAuth as Auth
        Auth.login(key)
        ok = Auth.verify_api_key()
        logger.info("QCloud auth: %s", "OK" if ok else "FAILED")
        return bool(ok)
    except Exception as e:
        logger.error("QCloud auth failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Job 1: Max-Cut facility clustering
# ---------------------------------------------------------------------------

def build_facility_graph(
    facilities: list[Facility],
    routes: list[Route],
) -> nx.Graph:
    """Build a networkx graph from facilities and routes.

    Nodes are facility IDs. Edge weights are transit_days.
    """
    G = nx.Graph()
    for f in facilities:
        G.add_node(f.id, lat=f.latitude, lon=f.longitude)
    for r in routes:
        if G.has_node(r.source_id) and G.has_node(r.destination_id):
            if r.source_id != r.destination_id:
                w = G.get_edge_data(r.source_id, r.destination_id, {}).get("weight", float("inf"))
                if r.transit_days < w:
                    G.add_edge(r.source_id, r.destination_id, weight=r.transit_days)
    return G


def _score_bitstring_maxcut(bitstring: str, G: nx.Graph) -> float:
    """Compute cut value: Σ weight(i,j) for edges crossing the partition."""
    nodes = list(G.nodes())
    cut = 0.0
    for i, j, d in G.edges(data=True):
        idx_i = nodes.index(i)
        idx_j = nodes.index(j)
        bi = int(bitstring[-(idx_i + 1)]) if idx_i < len(bitstring) else 0
        bj = int(bitstring[-(idx_j + 1)]) if idx_j < len(bitstring) else 0
        if bi != bj:
            cut += d.get("weight", 1.0)
    return cut


def _bitstring_to_clusters(bitstring: str, G: nx.Graph) -> list[list[str]]:
    """Convert a bitstring to two cluster lists of facility IDs."""
    nodes = list(G.nodes())
    c0, c1 = [], []
    for idx, node in enumerate(nodes):
        bit = int(bitstring[-(idx + 1)]) if idx < len(bitstring) else 0
        if bit == 0:
            c0.append(node)
        else:
            c1.append(node)
    return [c0, c1]


def _solve_maxcut_local(G: nx.Graph) -> MaxCutResult:
    """Local NetworkX-based greedy Max-Cut as fallback."""
    nodes = list(G.nodes())
    n = len(nodes)
    best_bs = ""
    best_cut = 0.0
    for mask in range(1 << n):
        bs = format(mask, f"0{n}b")
        c = _score_bitstring_maxcut(bs, G)
        if c > best_cut:
            best_cut = c
            best_bs = bs
        if n > 15:
            break
    if n > 15:
        partition = nx.community.kernighan_lin_bisection(G)
        c0 = sorted(partition[0])
        c1 = sorted(partition[1])
        return MaxCutResult(
            clusters=[c0, c1], best_bitstring="", cut_value=0.0,
            backend="local_networkx", shots=0,
        )
    clusters = _bitstring_to_clusters(best_bs, G)
    return MaxCutResult(
        clusters=clusters, best_bitstring=best_bs, cut_value=best_cut,
        backend="local_bruteforce", shots=0,
    )


def _to_int_graph(G: nx.Graph) -> tuple[nx.Graph, list[str]]:
    """Convert a string-node graph to integer-node graph for the QAOA SDK.

    Returns (int_graph, node_list) where node_list[i] is the original node name.
    """
    nodes = list(G.nodes())
    mapping = {n: i for i, n in enumerate(nodes)}
    G_int = nx.relabel_nodes(G, mapping)
    return G_int, nodes


def _run_qaoa_on_device(
    G: nx.Graph,
    G_int: nx.Graph,
    device: str,
    p_layers: int,
    shots: int,
    max_iterations: int,
) -> MaxCutResult:
    """Run QAOA on a specific device. Raises on failure."""
    from qpiai_quantum.algorithms.opt.problems import MaxCutProblem
    from qpiai_quantum.algorithms.opt.solvers import QAOASolver

    problem = MaxCutProblem(G_int)
    solver = QAOASolver(
        problem=problem,
        layers=p_layers,
        optimizer="COBYLA",
        max_iterations=max_iterations,
        verbose=False,
    )

    result = solver.run(
        shots=shots,
        device_name=device,
        experiment_name="Default Experiment",
    )

    counts = result.counts or {}
    best_bs = result.bitstring or ""
    best_cut = 0.0

    for bs, freq in counts.items():
        c = _score_bitstring_maxcut(bs, G)
        if c > best_cut:
            best_cut = c
            best_bs = bs

    clusters = _bitstring_to_clusters(best_bs, G)
    return MaxCutResult(
        clusters=clusters,
        best_bitstring=best_bs,
        cut_value=best_cut,
        backend=device,
        shots=shots,
        raw_counts=counts,
    )


def solve_maxcut_qaoa(
    G: nx.Graph,
    p_layers: int = 3,
    shots: int = 4096,
    backend: str = "simulator",
    max_iterations: int = 100,
) -> MaxCutResult | None:
    """Solve Max-Cut on the facility graph using QCloud QAOA.

    Execution order:
      1. Cloud simulator (QpiAI-QSV-Simulator) — jobs visible on qcloud.qpiai.tech
      2. Local simulator (QpiAI-QSV-Local) — runs locally via SDK
      3. Brute-force / NetworkX fallback — no SDK needed

    Returns MaxCutResult or None only if all paths fail.
    """
    if G.number_of_nodes() == 0:
        return MaxCutResult(clusters=[[], []], best_bitstring="",
                            cut_value=0.0, backend="none", shots=0)

    G_int, node_list = _to_int_graph(G)

    # Pick the target cloud device
    if backend == "qpu":
        cloud_device = "QpiAI-Indus-1"
    else:
        cloud_device = "QpiAI-QSV-Simulator"

    # --- Tier 1: Cloud simulator (creates jobs on the dashboard) ---
    try:
        result = _run_qaoa_on_device(
            G, G_int, cloud_device, p_layers, shots, max_iterations,
        )
        logger.info("QAOA Max-Cut solved via qcloud_platform (device=%s)", cloud_device)
        return result
    except Exception as e:
        logger.warning(
            "QCloud cloud submission failed (device=%s): %s. "
            "Falling back to local simulator.",
            cloud_device, e,
        )

    # --- Tier 2: Local SDK simulator ---
    try:
        result = _run_qaoa_on_device(
            G, G_int, "QpiAI-QSV-Local", p_layers, shots, max_iterations,
        )
        logger.info("QAOA Max-Cut solved via local_simulator (QpiAI-QSV-Local)")
        return result
    except Exception as e:
        logger.warning(
            "QCloud local simulator failed: %s. Falling back to brute-force.", e,
        )

    # --- Tier 3: Pure local fallback (no SDK) ---
    try:
        local_result = _solve_maxcut_local(G)
        logger.info("QAOA Max-Cut solved via local_bruteforce")
        return local_result
    except Exception as e:
        logger.error("All Max-Cut solvers failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# K-means fallback
# ---------------------------------------------------------------------------

def kmeans_clustering(
    facilities: list[Facility],
    n_clusters: int = 2,
) -> list[list[str]]:
    """K-means on (latitude, longitude). Always available as fallback."""
    if not facilities:
        return [[] for _ in range(n_clusters)]

    coords = np.array([[f.latitude, f.longitude] for f in facilities])
    n = len(facilities)

    if n <= n_clusters:
        return [[f.id] for f in facilities] + [[] for _ in range(n_clusters - n)]

    rng = np.random.RandomState(42)
    indices = rng.choice(n, size=n_clusters, replace=False)
    centroids = coords[indices].copy()

    for _ in range(50):
        dists = np.array([
            np.sum((coords - c) ** 2, axis=1) for c in centroids
        ])
        labels = np.argmin(dists, axis=0)

        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            members = coords[labels == k]
            if len(members) > 0:
                new_centroids[k] = members.mean(axis=0)
            else:
                new_centroids[k] = centroids[k]

        if np.allclose(centroids, new_centroids, atol=1e-8):
            break
        centroids = new_centroids

    clusters: list[list[str]] = [[] for _ in range(n_clusters)]
    for i, f in enumerate(facilities):
        clusters[labels[i]].append(f.id)
    return clusters


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _intra_cluster_cost(
    clusters: list[list[str]],
    route_map: dict[str, Route],
) -> float:
    """Total trip cost for routes within each cluster."""
    total = 0.0
    for cluster in clusters:
        fac_set = set(cluster)
        for i in cluster:
            for j in cluster:
                if i != j:
                    key = f"{i}|{j}"
                    if key in route_map:
                        total += route_map[key].trip_cost_inr
    return total


def compare_clusterings(
    qcloud_clusters: list[list[str]] | None,
    kmeans_clusters: list[list[str]],
    route_map: dict[str, Route],
) -> dict:
    """Compare QCloud vs k-means clustering by intra-cluster transport cost."""
    km_cost = _intra_cluster_cost(kmeans_clusters, route_map)

    if qcloud_clusters is None:
        return {
            "winner": "kmeans",
            "reason": "qcloud_unavailable",
            "kmeans_cost": km_cost,
            "qcloud_cost": None,
            "recommended_clusters": kmeans_clusters,
        }

    qc_cost = _intra_cluster_cost(qcloud_clusters, route_map)

    if km_cost <= qc_cost:
        winner = "kmeans"
    else:
        winner = "qcloud"

    return {
        "winner": winner,
        "kmeans_cost": km_cost,
        "qcloud_cost": qc_cost,
        "savings": abs(km_cost - qc_cost),
        "recommended_clusters": kmeans_clusters if winner == "kmeans" else qcloud_clusters,
    }


# ---------------------------------------------------------------------------
# Cluster balance check
# ---------------------------------------------------------------------------

def check_cluster_balance(
    clusters: list[list[str]],
    threshold: float = 0.70,
) -> bool:
    """Return True if clusters are balanced (no cluster has >threshold of nodes)."""
    total = sum(len(c) for c in clusters)
    if total == 0:
        return True
    for c in clusters:
        if len(c) / total > threshold:
            return False
    return True


# ---------------------------------------------------------------------------
# Job 2: Small QUBO via QAOA
# ---------------------------------------------------------------------------

def _qubo_to_ising(
    Q: dict[tuple[str, str], float],
    variable_names: list[str],
) -> list[tuple[list[tuple[int, str]], float]]:
    """Convert QUBO Q-matrix to Ising Hamiltonian terms.

    QUBO: H = Σ Q[i,j] xi xj  (xi ∈ {0,1})
    Ising: xi = (1 - Zi)/2

    Returns list of (pauli_ops, coeff) where pauli_ops = [(qubit, 'Z'), ...]
    """
    name_to_idx = {n: i for i, n in enumerate(variable_names)}
    n = len(variable_names)

    h_linear = [0.0] * n
    J_quad: dict[tuple[int, int], float] = {}
    offset = 0.0

    for (ni, nj), val in Q.items():
        if abs(val) < 1e-15:
            continue
        i = name_to_idx[ni]
        j = name_to_idx[nj]

        if i == j:
            offset += val / 2.0
            h_linear[i] -= val / 2.0
        else:
            offset += val / 4.0
            h_linear[i] -= val / 4.0
            h_linear[j] -= val / 4.0
            key = (min(i, j), max(i, j))
            J_quad[key] = J_quad.get(key, 0.0) + val / 4.0

    terms: list[tuple[list[tuple[int, str]], float]] = []
    terms.append(([], offset))

    for i, h in enumerate(h_linear):
        if abs(h) > 1e-15:
            terms.append(([(i, "Z")], h))

    for (i, j), Jij in J_quad.items():
        if abs(Jij) > 1e-15:
            terms.append(([(i, "Z"), (j, "Z")], Jij))

    return terms


def solve_small_qubo_qaoa(
    Q: dict[tuple[str, str], float],
    variable_names: list[str],
    p_layers: int = 3,
    shots: int = 4096,
    backend: str = "simulator",
) -> QAOAResult | None:
    """Solve a small QUBO using QAOA on QCloud.

    Builds the Ising Hamiltonian, constructs a QAOA circuit manually,
    and submits to QCloud. Scores by QUBO objective value, not frequency.

    Returns QAOAResult or None on failure.
    """
    n = len(variable_names)
    if n == 0:
        return QAOAResult(best_bitstring={}, objective_value=0.0,
                          optimal_energy=0.0, backend="none", shots=0)
    if n > 15:
        logger.warning("QUBO too large for QAOA (%d vars), skipping", n)
        return None

    try:
        from qpiai_quantum import Circuit

        ising_terms = _qubo_to_ising(Q, variable_names)
        name_to_idx = {nm: i for i, nm in enumerate(variable_names)}

        device = "QpiAI-QSV-Local"
        if backend == "qpu":
            device = "QpiAI-INDUS-QPU"

        gamma_params = [0.5] * p_layers
        beta_params = [0.5] * p_layers

        qc = Circuit(n)

        for i in range(n):
            qc.h(i)

        for layer in range(p_layers):
            gamma = gamma_params[layer]
            for ops, coeff in ising_terms:
                if len(ops) == 2:
                    i, j = ops[0][0], ops[1][0]
                    qc.rzz(i, j, 2.0 * gamma * coeff)
                elif len(ops) == 1:
                    i = ops[0][0]
                    qc.rz(i, 2.0 * gamma * coeff)

            beta = beta_params[layer]
            for i in range(n):
                qc.rx(i, 2.0 * beta)

        qc.measure_all()

        result = qc.run(
            shots=shots,
            device_name=device,
            experiment_name="MedBalance-SmallQUBO",
        )

        counts = result.get_counts() if hasattr(result, "get_counts") else {}

        from .qubo_builder import compute_objective_value

        best_h = float("inf")
        best_bits: dict[str, int] = {nm: 0 for nm in variable_names}

        for bs_str, freq in counts.items():
            bits = {}
            for idx, nm in enumerate(variable_names):
                bit_pos = -(idx + 1)
                bits[nm] = int(bs_str[bit_pos]) if abs(bit_pos) <= len(bs_str) else 0
            h = compute_objective_value(bits, Q)
            if h < best_h:
                best_h = h
                best_bits = bits

        return QAOAResult(
            best_bitstring=best_bits,
            objective_value=best_h,
            optimal_energy=best_h,
            counts=counts,
            backend=device,
            shots=shots,
        )

    except Exception as e:
        logger.error("QAOA small-QUBO failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Evidence logging
# ---------------------------------------------------------------------------

def log_qcloud_evidence(
    label: str,
    result: MaxCutResult | QAOAResult | dict,
    evidence_dir: str = "evidence/qcloud_runs",
) -> Path:
    """Save QCloud result to evidence directory."""
    edir = Path(evidence_dir)
    edir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = edir / f"{ts}_{label}.json"

    if isinstance(result, (MaxCutResult, QAOAResult)):
        data = result.__dict__
    else:
        data = result

    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("QCloud evidence saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_clustering_pipeline(
    facilities: list[Facility],
    routes: list[Route],
    route_map: dict[str, Route],
    api_key: str | None = None,
    p_layers: int = 3,
    shots: int = 4096,
) -> dict:
    """Run full clustering pipeline: QAOA Max-Cut + k-means fallback.

    1. Authenticate with QCloud
    2. Build facility graph
    3. Run QAOA Max-Cut (or None if auth fails)
    4. Run k-means
    5. Compare and select winner
    6. Check cluster balance → fall back to k-means if imbalanced
    7. Log evidence
    """
    authed = authenticate(api_key)
    G = build_facility_graph(facilities, routes)

    qaoa_result = None
    if authed:
        qaoa_result = solve_maxcut_qaoa(G, p_layers=p_layers, shots=shots)

    km_clusters = kmeans_clustering(facilities, n_clusters=2)

    qc_clusters = qaoa_result.clusters if qaoa_result else None
    comparison = compare_clusterings(qc_clusters, km_clusters, route_map)

    chosen = comparison["recommended_clusters"]

    if not check_cluster_balance(chosen):
        logger.warning("Cluster imbalance >70%%, falling back to k-means")
        chosen = km_clusters
        comparison["rebalanced"] = True

    pipeline_result = {
        "chosen_clusters": chosen,
        "qaoa_result": qaoa_result,
        "kmeans_clusters": km_clusters,
        "comparison": comparison,
        "authenticated": authed,
    }

    log_qcloud_evidence("clustering_pipeline", {
        "chosen_clusters": chosen,
        "kmeans_clusters": km_clusters,
        "comparison": comparison,
        "authenticated": authed,
        "qaoa_available": qaoa_result is not None,
    })

    return pipeline_result
