"""Replay: 4-week simulation engine for benchmarking MedBalance-Q.

Facilities consume stock daily with random variation. At the end of each
week the planner runs (QUBO, greedy, or no-action) and transfers execute.
Stockouts, expirations, and costs are tracked per week.
"""

from __future__ import annotations

import copy
import csv
import logging
from dataclasses import dataclass, field
from math import floor
from pathlib import Path

import numpy as np

from .allocation import greedy_fefo_allocate
from .eligibility import create_transfer_variables
from .inventory import (
    build_inventory_snapshot,
    build_route_map,
    compute_facility_stock,
)
from .qubo_builder import build_q_matrix, compute_objective_value, decode_bitstring
from .schemas import (
    DrugTier,
    InventoryData,
    Lot,
    StorageCondition,
    TransferVariable,
)
from .validation import validate_plan

logger = logging.getLogger(__name__)

DAYS_PER_WEEK = 7


@dataclass
class WeekResult:
    week: int
    transfers: list[TransferVariable] = field(default_factory=list)
    stockout_facility_drug_days: int = 0
    stockout_lifesaving_days: int = 0
    expired_units: int = 0
    expired_value_inr: float = 0.0
    transport_cost_inr: float = 0.0
    net_value_inr: float = 0.0
    validation_violations: int = 0


# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------

def _clone_lots(lots: list[Lot]) -> list[Lot]:
    """Deep-copy lot list so mutations don't leak."""
    return [lot.model_copy(deep=True) for lot in lots]


def _simulate_day(
    lots: list[Lot],
    data_dict: dict,
    rng: np.random.RandomState,
    daily_variation: float,
) -> tuple[int, int, int, float]:
    """Simulate one day of consumption + expiry.

    Returns (stockout_fac_drug_count, stockout_lifesaving, expired_units, expired_value).
    """
    usage_rates = data_dict["usage_rates"]
    drugs = data_dict["drugs"]
    facilities = data_dict["facilities"]

    stockouts = 0
    stockouts_ls = 0
    expired_units = 0
    expired_value = 0.0

    for lot in lots:
        if lot.days_to_expiry > 0:
            lot.days_to_expiry -= 1

    newly_expired = [lot for lot in lots if lot.days_to_expiry <= 0 and lot.quantity > 0]
    for lot in newly_expired:
        drug = drugs.get(lot.drug_code)
        price = drug.unit_price_inr if drug else 0.0
        expired_units += lot.quantity
        expired_value += lot.quantity * price
        lot.quantity = 0

    lots_by_fac_drug: dict[str, list[Lot]] = {}
    for lot in lots:
        key = f"{lot.facility_id}|{lot.drug_code}"
        lots_by_fac_drug.setdefault(key, []).append(lot)

    consumed_keys: set[str] = set()
    for fac_id in facilities:
        for drug_code in drugs:
            rate_key = f"{fac_id}|{drug_code}"
            rate_obj = usage_rates.get(rate_key)
            if not rate_obj or rate_obj.units_per_day <= 0:
                continue

            variation = rng.uniform(-daily_variation, daily_variation)
            actual_usage = max(0.0, rate_obj.units_per_day * (1.0 + variation))
            to_consume = floor(actual_usage)
            if to_consume <= 0:
                continue

            fac_lots = lots_by_fac_drug.get(rate_key, [])
            fac_lots_live = [l for l in fac_lots if l.quantity > 0 and l.days_to_expiry > 0]
            fac_lots_live.sort(key=lambda l: (l.days_to_expiry, l.lot_id))

            remaining = to_consume
            for lot in fac_lots_live:
                if remaining <= 0:
                    break
                take = min(lot.quantity, remaining)
                lot.quantity -= take
                remaining -= take

            if remaining > 0:
                stockouts += 1
                drug = drugs.get(drug_code)
                if drug and drug.tier == DrugTier.life_saving:
                    stockouts_ls += 1

            consumed_keys.add(rate_key)

    return stockouts, stockouts_ls, expired_units, expired_value


def _execute_transfers(
    transfers: list[TransferVariable],
    lots: list[Lot],
) -> None:
    """Move lot quantities from source to destination."""
    lot_map: dict[str, Lot] = {}
    for lot in lots:
        lot_map.setdefault(lot.lot_id, lot)

    for t in transfers:
        src_lot = lot_map.get(t.lot_id)
        if src_lot and src_lot.quantity >= t.usable_quantity:
            src_lot.quantity -= t.usable_quantity

            dst_lot = None
            for lot in lots:
                if (lot.facility_id == t.destination_id
                        and lot.drug_code == t.drug_code
                        and lot.lot_id == t.lot_id):
                    dst_lot = lot
                    break

            if dst_lot:
                dst_lot.quantity += t.usable_quantity
            else:
                lots.append(Lot(
                    lot_id=t.lot_id,
                    facility_id=t.destination_id,
                    drug_code=t.drug_code,
                    batch_number=src_lot.batch_number,
                    quantity=t.usable_quantity,
                    days_to_expiry=max(0, t.days_to_expiry - t.transit_days),
                    storage_condition=src_lot.storage_condition,
                ))


def _build_data_dict_from_lots(lots: list[Lot], base_data: dict) -> dict:
    """Rebuild data dict with updated lots (keeps everything else)."""
    return {
        "facilities": base_data["facilities"],
        "drugs": base_data["drugs"],
        "lots": lots,
        "routes": base_data["routes"],
        "usage_rates": base_data["usage_rates"],
        "reserves": base_data["reserves"],
    }


def _build_inventory_data(data_dict: dict, snapshot) -> InventoryData:
    route_map = build_route_map(data_dict["routes"])
    return InventoryData(
        facilities=data_dict["facilities"],
        drugs=data_dict["drugs"],
        lots=data_dict["lots"],
        routes=data_dict["routes"],
        usage_rates=data_dict["usage_rates"],
        reserves=data_dict["reserves"],
        route_map=route_map,
        stock_by_facility_drug=snapshot.facility_stock,
        needs=snapshot.needs,
    )


# ---------------------------------------------------------------------------
# Replay Engine
# ---------------------------------------------------------------------------

class ReplayEngine:
    """Multi-week simulation with daily consumption variation."""

    def __init__(
        self,
        data: dict,
        n_weeks: int = 4,
        daily_variation: float = 0.2,
    ):
        self._base_data = data
        self.n_weeks = n_weeks
        self.daily_variation = daily_variation

    def _run_week(
        self,
        lots: list[Lot],
        rng: np.random.RandomState,
        mode: str = "none",
    ) -> tuple[WeekResult, list[TransferVariable]]:
        """Simulate 7 days of consumption, then optionally run planner.

        mode: "none" (no transfers), "greedy", "qubo"
        """
        week_stockouts = 0
        week_stockouts_ls = 0
        week_expired = 0
        week_expired_val = 0.0

        data_dict = _build_data_dict_from_lots(lots, self._base_data)

        for _ in range(DAYS_PER_WEEK):
            so, so_ls, exp_u, exp_v = _simulate_day(
                lots, data_dict, rng, self.daily_variation
            )
            week_stockouts += so
            week_stockouts_ls += so_ls
            week_expired += exp_u
            week_expired_val += exp_v

        transfers: list[TransferVariable] = []
        violations = 0
        transport_cost = 0.0

        if mode != "none":
            data_dict = _build_data_dict_from_lots(lots, self._base_data)
            snapshot = build_inventory_snapshot(data_dict)
            inv_data = _build_inventory_data(data_dict, snapshot)
            variables = create_transfer_variables(
                snapshot.surplus_chunks, inv_data
            )

            if variables:
                if mode == "greedy":
                    transfers = greedy_fefo_allocate(
                        variables, snapshot, inv_data
                    )
                elif mode == "qubo":
                    Q = build_q_matrix(variables, snapshot, inv_data)
                    var_names = [v.var_id for v in variables]

                    best_h = float("inf")
                    best_bits: dict[str, int] = {v: 0 for v in var_names}
                    for v in variables:
                        solo = {n: (1 if n == v.var_id else 0) for n in var_names}
                        h = compute_objective_value(solo, Q)
                        if h < best_h:
                            best_h = h
                            best_bits = solo

                    all_on = {n: 1 for n in var_names}
                    h_all = compute_objective_value(all_on, Q)
                    if h_all < best_h:
                        best_bits = all_on

                    transfers = decode_bitstring(best_bits, variables)

                if transfers:
                    val_result = validate_plan(
                        transfers, variables, snapshot, inv_data
                    )
                    violations = len(val_result.violations)

                    if val_result.is_valid:
                        _execute_transfers(transfers, lots)
                        transport_cost = sum(t.transport_cost for t in transfers)

        net_val = week_expired_val - transport_cost

        return WeekResult(
            week=0,
            transfers=transfers,
            stockout_facility_drug_days=week_stockouts,
            stockout_lifesaving_days=week_stockouts_ls,
            expired_units=week_expired,
            expired_value_inr=week_expired_val,
            transport_cost_inr=transport_cost,
            net_value_inr=net_val,
            validation_violations=violations,
        ), transfers

    def run_scenario(
        self,
        scenario_name: str,
        seed: int,
        mode: str = "qubo",
    ) -> list[WeekResult]:
        """Run n_weeks simulation with the given planning mode.

        mode: "none", "greedy", "qubo"
        """
        rng = np.random.RandomState(seed)
        lots = _clone_lots(self._base_data["lots"])
        results: list[WeekResult] = []

        for w in range(1, self.n_weeks + 1):
            wr, _ = self._run_week(lots, rng, mode=mode)
            wr.week = w
            results.append(wr)

        logger.info(
            "Replay %s (seed=%d, mode=%s): %d weeks, "
            "total stockouts=%d, expired=%d units",
            scenario_name, seed, mode,
            self.n_weeks,
            sum(r.stockout_facility_drug_days for r in results),
            sum(r.expired_units for r in results),
        )
        return results

    def run_scenario_no_action(
        self, scenario_name: str, seed: int
    ) -> list[WeekResult]:
        return self.run_scenario(scenario_name, seed, mode="none")

    def run_scenario_greedy(
        self, scenario_name: str, seed: int
    ) -> list[WeekResult]:
        return self.run_scenario(scenario_name, seed, mode="greedy")

    # ------------------------------------------------------------------
    # Scenario builders
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_scenario_mods(
        data: dict,
        scenario: str,
    ) -> dict:
        """Return a modified copy of data for a specific scenario."""
        data = {k: (copy.deepcopy(v) if k == "lots" else v) for k, v in data.items()}
        lots = data["lots"]

        if scenario == "normal_week":
            pass

        elif scenario == "shortage_heavy":
            for lot in lots:
                lot.quantity = max(0, lot.quantity // 3)

        elif scenario == "expiry_heavy":
            for lot in lots[:len(lots) // 2]:
                lot.days_to_expiry = min(lot.days_to_expiry, 10)

        elif scenario == "outbreak_spike":
            usage = copy.deepcopy(data["usage_rates"])
            drug_codes = list(data["drugs"].keys())
            if drug_codes:
                spike_drug = drug_codes[0]
                fac_ids = list(data["facilities"].keys())[:2]
                for fid in fac_ids:
                    key = f"{fid}|{spike_drug}"
                    if key in usage:
                        ur = usage[key]
                        from .schemas import UsageRate
                        usage[key] = UsageRate(
                            facility_id=ur.facility_id,
                            drug_code=ur.drug_code,
                            units_per_day=ur.units_per_day * 2.0,
                        )
            data = dict(data)
            data["usage_rates"] = usage

        elif scenario == "cold_chain_down":
            routes = [r for r in data["routes"] if r.has_cold_transport is False]
            data = dict(data)
            data["routes"] = routes

        return data

    SCENARIOS = [
        "normal_week",
        "shortage_heavy",
        "expiry_heavy",
        "outbreak_spike",
        "cold_chain_down",
    ]

    def run_all_scenarios(
        self,
        n_seeds: int = 20,
        evidence_dir: str = "evidence/benchmarks",
    ) -> dict:
        """Run all 5 scenarios × n_seeds, comparing QUBO vs greedy vs no-action.

        Returns dict[scenario][mode] → {mean, std, min, max} for key metrics.
        Saves summary CSV to evidence/benchmarks/.
        """
        all_results: dict[str, dict[str, list[dict]]] = {}
        rows: list[dict] = []

        for scenario in self.SCENARIOS:
            all_results[scenario] = {"qubo": [], "greedy": [], "none": []}
            mod_data = self._apply_scenario_mods(self._base_data, scenario)
            engine = ReplayEngine(mod_data, n_weeks=self.n_weeks,
                                  daily_variation=self.daily_variation)

            for seed in range(n_seeds):
                for mode in ("qubo", "greedy", "none"):
                    weeks = engine.run_scenario(scenario, seed, mode)
                    summary = {
                        "stockout_days": sum(w.stockout_facility_drug_days for w in weeks),
                        "expired_units": sum(w.expired_units for w in weeks),
                        "expired_value": sum(w.expired_value_inr for w in weeks),
                        "transport_cost": sum(w.transport_cost_inr for w in weeks),
                        "violations": sum(w.validation_violations for w in weeks),
                        "transfers": sum(len(w.transfers) for w in weeks),
                    }
                    all_results[scenario][mode].append(summary)
                    rows.append({
                        "scenario": scenario,
                        "mode": mode,
                        "seed": seed,
                        **summary,
                    })

        agg: dict[str, dict[str, dict[str, dict]]] = {}
        for scenario in self.SCENARIOS:
            agg[scenario] = {}
            for mode in ("qubo", "greedy", "none"):
                runs = all_results[scenario][mode]
                if not runs:
                    continue
                metrics_keys = runs[0].keys()
                agg[scenario][mode] = {}
                for mk in metrics_keys:
                    vals = [r[mk] for r in runs]
                    agg[scenario][mode][mk] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                    }

        out_dir = Path(evidence_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "scenario_comparison_table.csv"
        if rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            logger.info("Benchmark CSV saved: %s", csv_path)

        return agg
