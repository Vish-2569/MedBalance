"""QUBO matrix builder: constructs the Q-matrix from filtered transfer
variables for quantum-inspired optimization.

H = -Σ reward(k,d) × x(k,d)                              [reward]
  + P1 × Σ_lot Σ_{d<d'} x(k,d) × x(k,d')                [lot exclusivity]
  + P2 × Σ_{(d,g)} crit(g) × (Σ ratio(k)×x(k,d) - 1)²   [shortage fill]

All variables are binary (0/1), so x²=x.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from .schemas import (
    InventoryData,
    InventorySnapshot,
    TransferVariable,
)

logger = logging.getLogger(__name__)

FEFO_BONUS_WEIGHT = 0.05


def build_q_matrix(
    variables: list[TransferVariable],
    inventory_snapshot: InventorySnapshot,
    data: InventoryData,
    p1_factor: float = 2.0,
    p2_weight: float = 1.5,
    gamma_factor: float = 0.3,
) -> dict[tuple[str, str], float]:
    """Build the QUBO Q-matrix from eligible transfer variables.

    Returns dict mapping (var_id_i, var_id_j) → coefficient.
    Diagonal (i,i) entries are linear terms; off-diagonal (i,j) are quadratic.
    Keys are canonicalized so i <= j lexicographically.
    """
    Q: dict[tuple[str, str], float] = {}

    if not variables:
        return Q

    needs = inventory_snapshot.needs
    criticality = {code: drug.criticality for code, drug in data.drugs.items()}

    # --- (a) Raw rewards with over-allocation guard ---
    raw_rewards: dict[str, float] = {}
    for v in variables:
        need_key = f"{v.destination_id}|{v.drug_code}"
        need = needs.get(need_key, 0)

        if v.usable_quantity > 0 and need > 0:
            cap_ratio = min(v.usable_quantity, need) / v.usable_quantity
        else:
            cap_ratio = 0.0

        effective_value = v.value_at_risk * cap_ratio
        raw_rewards[v.var_id] = max(0.0, effective_value - v.transport_cost)

    # --- (b) Normalize to [0, 1] ---
    max_reward = max(raw_rewards.values()) if raw_rewards else 1.0
    if max_reward <= 0:
        max_reward = 1.0

    p1 = p1_factor  # in normalized space, max reward = 1.0

    # --- (c) Group variables ---
    by_lot: dict[str, list[TransferVariable]] = defaultdict(list)
    by_dest_drug: dict[str, list[TransferVariable]] = defaultdict(list)

    for v in variables:
        by_lot[v.lot_id].append(v)
        by_dest_drug[f"{v.destination_id}|{v.drug_code}"].append(v)

    # --- (d) Diagonal reward + FEFO bonus ---
    for v in variables:
        norm_reward = raw_rewards[v.var_id] / max_reward
        fefo_bonus = FEFO_BONUS_WEIGHT / max(v.days_to_expiry, 1)
        _add(Q, v.var_id, v.var_id, -(norm_reward + fefo_bonus))

    # --- (e) P1: lot exclusivity ---
    for lot_id, group in by_lot.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                _add(Q, group[i].var_id, group[j].var_id, p1)

    # --- (f) P2: shortage fill deviation ---
    for key, group in by_dest_drug.items():
        need = needs.get(key, 0)
        if need <= 0:
            continue

        _, drug_code = key.split("|", 1)
        crit = criticality.get(drug_code, 1)
        p2 = p2_weight * crit

        ratios: dict[str, float] = {}
        for v in group:
            ratios[v.var_id] = v.usable_quantity / need

        # Expand (Σ r_k x_k - 1)²:
        #   diagonal: (r_k² - 2r_k) per variable
        #   off-diagonal: 2 × r_i × r_j per pair
        for v in group:
            r = ratios[v.var_id]
            _add(Q, v.var_id, v.var_id, p2 * (r * r - 2.0 * r))

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                ri = ratios[group[i].var_id]
                rj = ratios[group[j].var_id]
                _add(Q, group[i].var_id, group[j].var_id, 2.0 * p2 * ri * rj)

    # --- (g) γ: shared-trip bonus ---
    # Encourage co-selecting transfers from the same source to the same
    # destination (they can share a truck trip).
    if gamma_factor > 0 and variables:
        min_cost = min(v.transport_cost for v in variables)
        if min_cost > 0:
            gamma = gamma_factor * min_cost / max_reward

            by_src_dst: dict[tuple[str, str], list[TransferVariable]] = defaultdict(list)
            for v in variables:
                by_src_dst[(v.source_id, v.destination_id)].append(v)

            for (src, dst), group in by_src_dst.items():
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        vi, vj = group[i], group[j]
                        if vi.lot_id != vj.lot_id:
                            _add(Q, vi.var_id, vj.var_id, -gamma)

    logger.info(
        "QUBO built: %d variables, %d entries, max_reward=%.2f, P1=%.2f",
        len(variables), len(Q), max_reward, p1,
    )

    return Q


def _add(Q: dict[tuple[str, str], float], i: str, j: str, val: float) -> None:
    """Add val to Q[(i,j)] with canonical key ordering (i <= j)."""
    key = (i, j) if i <= j else (j, i)
    Q[key] = Q.get(key, 0.0) + val


# ---------------------------------------------------------------------------
# Conversion / decoding utilities
# ---------------------------------------------------------------------------

def q_matrix_to_dict(Q: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    """Convert Q-matrix to QpiAI Opt SDK format.

    Ensures canonical ordering (i <= j) with combined symmetric entries.
    Drops near-zero entries for sparsity.
    """
    combined: dict[tuple[str, str], float] = {}
    for (i, j), val in Q.items():
        key = (i, j) if i <= j else (j, i)
        combined[key] = combined.get(key, 0.0) + val

    return {k: v for k, v in combined.items() if abs(v) > 1e-12}


def decode_bitstring(
    bitstring: dict[str, int],
    variables: list[TransferVariable],
) -> list[TransferVariable]:
    """Convert a QUBO solution bitstring to selected transfers.

    Args:
        bitstring: {var_id: 0 or 1} for each variable.
        variables: the full list of eligible TransferVariables.

    Returns:
        List of TransferVariables where x=1 in the bitstring.
    """
    var_lookup = {v.var_id: v for v in variables}
    selected = []
    for var_id, val in bitstring.items():
        if val == 1 and var_id in var_lookup:
            selected.append(var_lookup[var_id])
    return selected


def compute_objective_value(
    bitstring: dict[str, int],
    Q: dict[tuple[str, str], float],
) -> float:
    """Evaluate the QUBO objective H = Σ Q[i,j] × x_i × x_j.

    For diagonal entries (i==j), x_i × x_i = x_i (binary).
    """
    total = 0.0
    for (i, j), coeff in Q.items():
        xi = bitstring.get(i, 0)
        xj = bitstring.get(j, 0)
        total += coeff * xi * xj
    return total
