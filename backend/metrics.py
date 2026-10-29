"""Metrics: computes KPIs comparing a transfer plan against a no-action
baseline over a multi-week replay.

All metrics are derived from WeekResult lists produced by replay.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import DrugTier

if False:  # TYPE_CHECKING
    from .replay import WeekResult


@dataclass
class MetricsReport:
    actual_stockout_reduction: float
    actual_stockout_reduction_lifesaving: float
    actual_expiry_avoided_units: int
    actual_expiry_avoided_inr: float
    expected_vs_actual_gap: float
    inventory_feasibility: int
    net_value_inr: float
    total_transport_cost: float
    total_trips: int
    total_km: float


def compute_metrics(
    plan_results: list[WeekResult],
    baseline_results: list[WeekResult],
    lifesaving_drug_codes: set[str] | None = None,
    avg_km_per_trip: float = 45.0,
) -> MetricsReport:
    """Compare plan vs no-action baseline over the same simulation period.

    Args:
        plan_results: WeekResult list from the plan run (QUBO or greedy).
        baseline_results: WeekResult list from the no-action run.
        lifesaving_drug_codes: drug codes with tier=life_saving. If None,
            the life-saving reduction is set to 0.
        avg_km_per_trip: estimated average km per transfer trip.
    """
    plan_stockout_days = sum(w.stockout_facility_drug_days for w in plan_results)
    base_stockout_days = sum(w.stockout_facility_drug_days for w in baseline_results)

    if base_stockout_days > 0:
        stockout_reduction = (
            (base_stockout_days - plan_stockout_days) / base_stockout_days
        )
    else:
        stockout_reduction = 0.0

    plan_ls_stockout = sum(
        w.stockout_lifesaving_days for w in plan_results
    )
    base_ls_stockout = sum(
        w.stockout_lifesaving_days for w in baseline_results
    )
    if base_ls_stockout > 0:
        stockout_reduction_ls = (
            (base_ls_stockout - plan_ls_stockout) / base_ls_stockout
        )
    else:
        stockout_reduction_ls = 0.0

    plan_expired = sum(w.expired_units for w in plan_results)
    base_expired = sum(w.expired_units for w in baseline_results)
    expiry_avoided_units = max(0, base_expired - plan_expired)

    plan_expired_val = sum(w.expired_value_inr for w in plan_results)
    base_expired_val = sum(w.expired_value_inr for w in baseline_results)
    expiry_avoided_inr = max(0.0, base_expired_val - plan_expired_val)

    plan_net = sum(w.net_value_inr for w in plan_results)
    base_net = sum(w.net_value_inr for w in baseline_results)
    if abs(plan_net) > 0:
        gap = abs(plan_net - base_net) / abs(plan_net)
    else:
        gap = 0.0

    violations = sum(w.validation_violations for w in plan_results)
    transport_cost = sum(w.transport_cost_inr for w in plan_results)
    total_trips = sum(len(w.transfers) for w in plan_results)
    total_km = total_trips * avg_km_per_trip
    net_value = expiry_avoided_inr - transport_cost

    return MetricsReport(
        actual_stockout_reduction=stockout_reduction,
        actual_stockout_reduction_lifesaving=stockout_reduction_ls,
        actual_expiry_avoided_units=expiry_avoided_units,
        actual_expiry_avoided_inr=expiry_avoided_inr,
        expected_vs_actual_gap=gap,
        inventory_feasibility=violations,
        net_value_inr=net_value,
        total_transport_cost=transport_cost,
        total_trips=total_trips,
        total_km=total_km,
    )
