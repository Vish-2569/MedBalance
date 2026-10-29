MedBalance

**Expiry-aware essential medicine redistribution for district health networks, powered by quantum-inspired (QUBO) optimization.**

> GENAI-04: Design an AI-driven system that uses quantum computing to optimize the supply chain of essential medicines in a district-level health network, reducing expiry-based wastage while ensuring equitable distribution.

---

## Problem

Indian district health networks lose crores in expired medicines every year. Facilities hold surplus stock while neighboring facilities face stockouts of the same drugs. Manual redistribution is slow, ad-hoc, and ignores expiry timelines.

MedBalance solves this by modeling the redistribution problem as a **Quadratic Unconstrained Binary Optimization (QUBO)** — a formulation native to quantum annealers and QAOA circuits. Binary decision variables `x(k,d)` represent "send surplus chunk k to facility d", and the objective balances value recovery, FEFO compliance, shortage filling, and transport cost.

## Architecture

```
                         Streamlit Dashboard (dashboard/app.py)
                                    |
                              FastAPI Backend (main.py)
                                    |
        +----------+----------+-----+------+-----------+
        |          |          |            |           |
   inventory  eligibility  allocation   solver    validation
     .py        .py          .py        .py         .py
        |          |          |            |           |
        +-----+----+-----+---+            |           |
              |          |                 |           |
         qubo_builder   regret_ledger   qcloud.py     |
            .py            .py         (QpiAI QAOA)   |
              |                                       |
              +------------ replay.py --------+-------+
                            metrics.py        |
                                              |
                          genai_ingest.py   genai_narrate.py
                          (Gemini parsing)  (Gemini justifications)
```

### Module Summary

| Module | Purpose |
|---|---|
| `schemas.py` | Pydantic v2 data models — Facility, Drug, Lot, Route, TransferVariable, etc. |
| `inventory.py` | CSV loading, FEFO-ordered surplus chunks, shortage needs, stock indexing |
| `eligibility.py` | 5 hard-rule filter: drug match, route exists, cold-chain, expiry margin, reserve |
| `allocation.py` | Greedy FEFO baseline allocator |
| `qubo_builder.py` | Q-matrix construction: reward, lot exclusivity (P1), shortage fill (P2), FEFO bonus, shared-trip (gamma) |
| `solver.py` | QpiAI Opt SDK integration — QUBO solver with top-K samples |
| `qcloud.py` | QpiAI QCloud QAOA — Max-Cut facility clustering + small QUBO verification |
| `validation.py` | 8-check post-QUBO validation (lot exclusivity, cold-chain, FEFO, reserves, etc.) |
| `regret_ledger.py` | Records top-K alternative plans with regret deltas and feasibility |
| `genai_ingest.py` | Gemini-powered unstructured stock report parsing with drug matching |
| `genai_narrate.py` | Gemini-powered transfer justification generation with verification |
| `replay.py` | 4-week simulation engine — 5 scenarios, daily consumption with random variation |
| `metrics.py` | Stockout reduction, expiry avoidance, net value, transport cost metrics |
| `main.py` | FastAPI app — 11 endpoints tying the full pipeline together |
| `dashboard/app.py` | Streamlit UI — 6 tabs: Overview, Ingestion, Plan, Quantum, Evaluation, Evidence |

### QUBO Formulation

```
H = -sum(reward_k * x_kd)                    # maximize value recovery
  + P1 * sum_k (sum_d x_kd - 1)^2           # lot exclusivity: each chunk sent to at most 1 destination
  + P2 * sum_d (fill_d - target_d)^2         # shortage filling penalty
  - FEFO_bonus * sum(1/expiry_k * x_kd)     # prefer earliest-expiring lots
  - gamma * sum(shared_trip_ij * x_i * x_j)  # reward co-located transfers
```

## Setup

### Prerequisites

- Python 3.11+
- pip

### Install

```bash
git clone <repo-url>
cd medbalance-q
pip install -r requirements.txt
```

### API Keys (optional, in `.env`)

```
QPIAI_API_KEY=...          # QpiAI Opt solver (falls back to greedy without)
QPIAI_QCLOUD_API_KEY=...   # QpiAI QCloud QAOA (falls back to k-means without)
GEMINI_API_KEY=...          # Google Gemini (falls back to templates without)
```

The system is fully functional without any API keys — all three services degrade gracefully to classical/template alternatives.

## Running

### One-command demo

```bash
chmod +x make_demo.sh
./make_demo.sh
```

This runs all 231 tests, executes the full pipeline end-to-end, runs a 4-week replay simulation, and saves evidence artifacts.

### Backend + Dashboard

```bash
# Terminal 1: Start FastAPI backend
uvicorn backend.main:app --port 8000

# Terminal 2: Start Streamlit dashboard
streamlit run dashboard/app.py
```

### Tests only

```bash
pytest tests/ -v --tb=short
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/ingest` | GenAI stock report parsing |
| GET | `/inventory` | District-wide inventory snapshot |
| GET | `/inventory/{facility_id}` | Per-facility stock and needs |
| POST | `/plan/generate` | Run full pipeline (QUBO or greedy) |
| GET | `/plan/{id}/alternatives` | Regret ledger alternatives |
| GET | `/plan/{id}/order/{oid}` | Single transfer order detail |
| POST | `/plan/{id}/approve` | Approve selected transfer orders |
| POST | `/clustering` | QAOA Max-Cut vs k-means clustering |
| POST | `/replay` | Multi-week simulation replay |
| GET | `/evidence` | List all evidence artifacts |

## Dashboard Tabs

1. **District Overview** — Facility map, stock status table, drug-level breakdown
2. **Stock Ingestion** — Paste raw reports, GenAI extracts structured lots with confidence scores
3. **Generate Plan** — Quantum/greedy toggle, transfer orders with justifications, approval workflow
4. **Quantum Panel** — QAOA vs k-means clustering comparison with honest reporting
5. **Evaluation** — 5-scenario replay with comparison tables and bar charts
6. **Evidence** — Categorized listing of all pipeline artifacts

## Evidence Folder

All pipeline runs produce auditable artifacts in `evidence/`:

```
evidence/
  classical_baselines/     # Greedy allocation results
  regret_ledger/           # Alternative plan comparisons (JSON)
  benchmarks/              # Replay scenario CSVs
  qcloud/                  # QAOA clustering results
```

## Data

Synthetic district data in `backend/data/` representing a Bangalore-area health network:

- **7 facilities**: 1 district store (DS-01), 2 general hospitals (GH-01, GH-02), 2 community health centres (CHC-01, CHC-02), 2 primary health centres (PHC-01, PHC-02)
- **10 essential drugs**: Including insulin, metformin, amlodipine, amoxicillin (4 life-saving, 4 essential, 2 routine; 2 require cold chain)
- **40 lots** with varied expiry profiles (3 to 180 days)
- **20 transport routes** with cold-chain flags and trip costs
- **70 usage rates** and **70 reserve levels**

## Test Suite

231 tests across 11 test files covering every module:

| Test File | Tests | Coverage |
|---|---|---|
| `test_inventory.py` | 20 | CSV loading, stock computation, surplus chunks |
| `test_eligibility.py` | 23 | 5 hard rules, edge cases |
| `test_allocation.py` | 8 | Greedy FEFO allocator |
| `test_qubo_exact.py` | 11 | Brute-force QUBO verification |
| `test_solver_integration.py` | 13 | QpiAI Opt SDK integration |
| `test_qcloud_integration.py` | 24 | QAOA Max-Cut, Ising conversion, clustering |
| `test_genai_ingest.py` | 33 | Drug matching, report parsing, validation |
| `test_genai_narrate.py` | 17 | Justification generation, verification |
| `test_regret_ledger.py` | 16 | Alternative recording, persistence |
| `test_replay_metrics.py` | 26 | Simulation engine, scenario mods, metrics |
| `test_main.py` | 19 | All 11 FastAPI endpoints |
| **Total** | **231** | |

