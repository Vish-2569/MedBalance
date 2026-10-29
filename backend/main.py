"""FastAPI application for MedBalance-Q v2.

Ties the entire pipeline together: inventory → eligibility → QUBO → solver
→ validation → regret ledger → GenAI justifications → evidence.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .allocation import evaluate_greedy, greedy_fefo_allocate
from .eligibility import create_transfer_variables
from .genai_ingest import ParsedLot, parse_stock_report, validate_extraction
from .genai_narrate import generate_with_fallback
from .inventory import build_inventory_snapshot, build_route_map, load_all, load_data
from .metrics import MetricsReport, compute_metrics
from .qcloud import (
    build_facility_graph,
    compare_clusterings,
    kmeans_clustering,
    run_clustering_pipeline,
    solve_maxcut_qaoa,
)
from .qubo_builder import build_q_matrix, compute_objective_value, decode_bitstring
from .regret_ledger import RegretLedger, persist_ledger
from .replay import ReplayEngine, WeekResult
from .schemas import InventoryData, TransferOrder, TransferVariable
from .solver import MedBalanceOptSolver
from .validation import validate_plan

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="MedBalance-Q", version="2.0.0",
              description="Expiry-aware essential medicine redistribution")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_plans: dict[str, dict] = {}


def _get_data() -> tuple:
    """Load inventory data from CSVs. Returns (raw_data, snapshot, inv_data)."""
    raw = load_data()
    snapshot = build_inventory_snapshot(raw)
    route_map = build_route_map(raw["routes"])
    inv_data = InventoryData(
        facilities=raw["facilities"],
        drugs=raw["drugs"],
        lots=raw["lots"],
        routes=raw["routes"],
        usage_rates=raw["usage_rates"],
        reserves=raw["reserves"],
        route_map=route_map,
        stock_by_facility_drug=snapshot.facility_stock,
        needs=snapshot.needs,
    )
    return raw, snapshot, inv_data


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    facility_id: str
    raw_report: str

class PlanRequest(BaseModel):
    cover_days: int = 14
    use_quantum: bool = True
    solver: str = "opt"

class ApproveRequest(BaseModel):
    approved_order_ids: list[str]

class ClusteringRequest(BaseModel):
    method: str = "compare"

class ReplayRequest(BaseModel):
    scenario: str = "normal_week"
    n_seeds: int = 5
    n_weeks: int = 4


# ---------------------------------------------------------------------------
# 1. Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "version": "v2"}


# ---------------------------------------------------------------------------
# 2. Ingest
# ---------------------------------------------------------------------------

@app.post("/ingest")
def ingest(req: IngestRequest):
    logger.info("Ingest request for facility %s", req.facility_id)
    try:
        raw, _, inv_data = _get_data()
        drug_master = list(inv_data.drugs.values())
        lots = parse_stock_report(req.raw_report, req.facility_id, drug_master)
        accepted, flagged = validate_extraction(lots, drug_master)

        return {
            "accepted_lots": [_parsed_lot_dict(l) for l in accepted],
            "flagged_lots": [_parsed_lot_dict(l) for l in flagged],
            "confidence_scores": [
                {"lot_id": l.lot_id, "confidence": l.confidence} for l in lots
            ],
        }
    except Exception as e:
        logger.error("Ingest failed: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


def _parsed_lot_dict(p: ParsedLot) -> dict:
    return {
        "lot_id": p.lot_id, "facility_id": p.facility_id,
        "drug_code": p.drug_code, "drug_name_raw": p.drug_name_raw,
        "quantity": p.quantity, "batch_number": p.batch_number,
        "days_to_expiry": p.days_to_expiry,
        "storage_condition": p.storage_condition,
        "confidence": p.confidence, "flagged": p.flagged,
        "flag_reason": p.flag_reason,
    }


# ---------------------------------------------------------------------------
# 3. Inventory
# ---------------------------------------------------------------------------

@app.get("/inventory")
def get_inventory():
    try:
        raw, snapshot, inv_data = _get_data()
        return {
            "facility_stock": snapshot.facility_stock,
            "needs": snapshot.needs,
            "surplus_chunks": len(snapshot.surplus_chunks),
            "facilities": list(inv_data.facilities.keys()),
            "drugs": list(inv_data.drugs.keys()),
        }
    except Exception as e:
        logger.error("Inventory failed: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


# ---------------------------------------------------------------------------
# 4. Inventory per facility
# ---------------------------------------------------------------------------

@app.get("/inventory/{facility_id}")
def get_facility_inventory(facility_id: str):
    try:
        raw, snapshot, inv_data = _get_data()
        if facility_id not in inv_data.facilities:
            raise HTTPException(404, detail=f"Facility {facility_id} not found")

        stock = {
            k: v for k, v in snapshot.facility_stock.items()
            if k.startswith(f"{facility_id}|")
        }
        needs = {
            k: v for k, v in snapshot.needs.items()
            if k.startswith(f"{facility_id}|")
        }
        chunks = [
            c for c in snapshot.surplus_chunks
            if c.source_facility_id == facility_id
        ]
        return {
            "facility_id": facility_id,
            "stock": stock,
            "needs": needs,
            "surplus_chunks": [
                {
                    "chunk_id": c.chunk_id, "lot_id": c.lot_id,
                    "drug_code": c.drug_code, "quantity": c.quantity,
                    "days_to_expiry": c.days_to_expiry,
                    "value_at_risk": c.value_at_risk,
                }
                for c in chunks
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Facility inventory failed: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


# ---------------------------------------------------------------------------
# 5. Plan generate
# ---------------------------------------------------------------------------

@app.post("/plan/generate")
def generate_plan(req: PlanRequest):
    t0 = time.perf_counter()
    logger.info("Plan generation: solver=%s, quantum=%s", req.solver, req.use_quantum)

    try:
        raw, snapshot, inv_data = _get_data()
        variables = create_transfer_variables(snapshot.surplus_chunks, inv_data)

        if not variables:
            plan_id = str(uuid.uuid4())[:8]
            _plans[plan_id] = {
                "transfers": [], "variables": [], "validation": None,
                "ledger": RegretLedger(), "orders": {},
            }
            return {
                "plan_id": plan_id,
                "transfers": [],
                "validation": {"is_valid": True, "violations": [], "warnings": []},
                "metrics": {},
                "regret_ledger_summary": "No transfers needed.",
                "solver_used": req.solver,
                "wall_clock_s": time.perf_counter() - t0,
            }

        solver_used = req.solver
        selected: list[TransferVariable] = []
        opt_result = None
        Q = None

        if req.use_quantum and req.solver == "opt":
            Q = build_q_matrix(variables, snapshot, inv_data)
            var_names = [v.var_id for v in variables]
            solver = MedBalanceOptSolver()
            opt_result = solver.solve(Q, var_names)

            if opt_result:
                selected = decode_bitstring(opt_result.best, variables)
                solver_used = "opt"
            else:
                selected = greedy_fefo_allocate(variables, snapshot, inv_data)
                solver_used = "greedy (opt fallback)"
        else:
            selected = greedy_fefo_allocate(variables, snapshot, inv_data)
            solver_used = "greedy"

        val_result = validate_plan(selected, variables, snapshot, inv_data)

        ledger = RegretLedger()
        if opt_result and Q:
            ledger.record_alternatives(
                opt_result.top_k, Q, variables, snapshot, inv_data
            )

        orders: dict[str, dict] = {}
        for i, t in enumerate(selected):
            order_id = f"ORD-{i + 1:03d}"
            regret_alts = ledger.get_alternatives_for_transfer(t)
            regret_entry = regret_alts[0] if regret_alts else None
            justification = generate_with_fallback(t, regret_entry, inv_data)

            drug = inv_data.drugs.get(t.drug_code)
            orders[order_id] = {
                "order_id": order_id,
                "lot_id": t.lot_id,
                "source_id": t.source_id,
                "destination_id": t.destination_id,
                "drug_code": t.drug_code,
                "drug_name": drug.name if drug else t.drug_code,
                "quantity": t.usable_quantity,
                "days_to_expiry": t.days_to_expiry,
                "transport_cost": t.transport_cost,
                "justification": justification,
                "approved": False,
            }

        plan_id = str(uuid.uuid4())[:8]
        _plans[plan_id] = {
            "transfers": selected,
            "variables": variables,
            "validation": val_result,
            "ledger": ledger,
            "orders": orders,
            "snapshot": snapshot,
            "inv_data": inv_data,
            "Q": Q,
        }

        persist_ledger(ledger, plan_id)

        greedy_metrics = evaluate_greedy(selected, snapshot)
        wall = time.perf_counter() - t0

        return {
            "plan_id": plan_id,
            "transfers": [_transfer_dict(t) for t in selected],
            "validation": {
                "is_valid": val_result.is_valid,
                "violations": val_result.violations,
                "warnings": val_result.warnings,
                "fefo_deviations": val_result.fefo_deviations,
            },
            "metrics": greedy_metrics,
            "regret_ledger_summary": ledger.summary(),
            "solver_used": solver_used,
            "wall_clock_s": round(wall, 3),
            "orders": list(orders.values()),
        }

    except Exception as e:
        logger.error("Plan generation failed: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


def _transfer_dict(t: TransferVariable) -> dict:
    return {
        "var_id": t.var_id, "lot_id": t.lot_id,
        "source_id": t.source_id, "destination_id": t.destination_id,
        "drug_code": t.drug_code,
        "usable_quantity": t.usable_quantity,
        "days_to_expiry": t.days_to_expiry,
        "transit_days": t.transit_days,
        "transport_cost": t.transport_cost,
        "value_at_risk": t.value_at_risk,
    }


# ---------------------------------------------------------------------------
# 6. Plan alternatives
# ---------------------------------------------------------------------------

@app.get("/plan/{plan_id}/alternatives")
def get_alternatives(plan_id: str):
    plan = _plans.get(plan_id)
    if not plan:
        raise HTTPException(404, detail=f"Plan {plan_id} not found")

    ledger: RegretLedger = plan["ledger"]
    return {
        "plan_id": plan_id,
        "summary": ledger.summary(),
        "entries": [
            {
                "rank": e.plan_rank,
                "objective_value": e.objective_value,
                "delta_from_best": e.delta_from_best,
                "is_feasible": e.is_feasible,
                "n_transfers": len(e.transfers),
                "diff_from_chosen": e.diff_from_chosen,
                "blocking_constraints": e.blocking_constraints,
            }
            for e in ledger.entries
        ],
    }


# ---------------------------------------------------------------------------
# 7. Transfer order detail
# ---------------------------------------------------------------------------

@app.get("/plan/{plan_id}/order/{order_id}")
def get_order(plan_id: str, order_id: str):
    plan = _plans.get(plan_id)
    if not plan:
        raise HTTPException(404, detail=f"Plan {plan_id} not found")

    order = plan["orders"].get(order_id)
    if not order:
        raise HTTPException(404, detail=f"Order {order_id} not found in plan {plan_id}")

    return order


# ---------------------------------------------------------------------------
# 8. Approve orders
# ---------------------------------------------------------------------------

@app.post("/plan/{plan_id}/approve")
def approve_orders(plan_id: str, req: ApproveRequest):
    plan = _plans.get(plan_id)
    if not plan:
        raise HTTPException(404, detail=f"Plan {plan_id} not found")

    approved = []
    not_found = []
    for oid in req.approved_order_ids:
        if oid in plan["orders"]:
            plan["orders"][oid]["approved"] = True
            approved.append(oid)
        else:
            not_found.append(oid)

    return {
        "plan_id": plan_id,
        "approved": approved,
        "not_found": not_found,
    }


# ---------------------------------------------------------------------------
# 9. Clustering
# ---------------------------------------------------------------------------

@app.post("/clustering")
def clustering(req: ClusteringRequest):
    logger.info("Clustering request: method=%s", req.method)
    try:
        raw, _, inv_data = _get_data()
        facilities = list(inv_data.facilities.values())
        routes = inv_data.routes
        route_map = inv_data.route_map

        if req.method == "kmeans":
            clusters = kmeans_clustering(facilities, n_clusters=2)
            return {"method": "kmeans", "clusters": clusters}

        elif req.method == "qaoa":
            G = build_facility_graph(facilities, routes)
            result = solve_maxcut_qaoa(G)
            if result:
                return {
                    "method": "qaoa",
                    "clusters": result.clusters,
                    "cut_value": result.cut_value,
                    "backend": result.backend,
                }
            else:
                clusters = kmeans_clustering(facilities, n_clusters=2)
                return {
                    "method": "kmeans (qaoa failed)",
                    "clusters": clusters,
                }

        else:
            result = run_clustering_pipeline(facilities, routes, route_map)
            return {
                "method": "compare",
                "chosen_clusters": result["chosen_clusters"],
                "kmeans_clusters": result["kmeans_clusters"],
                "comparison": result["comparison"],
                "qaoa_available": result["qaoa_result"] is not None,
            }

    except Exception as e:
        logger.error("Clustering failed: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


# ---------------------------------------------------------------------------
# 10. Replay
# ---------------------------------------------------------------------------

@app.post("/replay")
def replay(req: ReplayRequest):
    logger.info("Replay: scenario=%s, seeds=%d, weeks=%d",
                req.scenario, req.n_seeds, req.n_weeks)
    try:
        raw = load_data()
        engine = ReplayEngine(raw, n_weeks=req.n_weeks, daily_variation=0.2)

        if req.scenario == "all":
            agg = engine.run_all_scenarios(n_seeds=req.n_seeds)
            return {"scenarios": agg}

        results: dict[str, list[dict]] = {}
        for mode in ("qubo", "greedy", "none"):
            mode_results = []
            for seed in range(req.n_seeds):
                weeks = engine.run_scenario(req.scenario, seed, mode=mode)
                mode_results.append({
                    "seed": seed,
                    "stockout_days": sum(w.stockout_facility_drug_days for w in weeks),
                    "expired_units": sum(w.expired_units for w in weeks),
                    "expired_value": sum(w.expired_value_inr for w in weeks),
                    "transport_cost": sum(w.transport_cost_inr for w in weeks),
                    "violations": sum(w.validation_violations for w in weeks),
                    "transfers": sum(len(w.transfers) for w in weeks),
                })
            results[mode] = mode_results

        return {
            "scenario": req.scenario,
            "n_seeds": req.n_seeds,
            "n_weeks": req.n_weeks,
            "results": results,
        }

    except Exception as e:
        logger.error("Replay failed: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


# ---------------------------------------------------------------------------
# 11. Evidence listing
# ---------------------------------------------------------------------------

@app.get("/evidence")
def list_evidence():
    evidence_dir = Path("evidence")
    if not evidence_dir.exists():
        return {"files": []}

    files = []
    for p in sorted(evidence_dir.rglob("*")):
        if p.is_file():
            files.append({
                "path": str(p.relative_to(evidence_dir)),
                "size_bytes": p.stat().st_size,
            })
    return {"files": files}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
