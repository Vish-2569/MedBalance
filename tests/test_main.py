"""Tests for the FastAPI application.

Uses TestClient to verify all endpoints without starting a server.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)

_CANNED_JUSTIFICATION = "Test justification for transfer order."


def _mock_narrate(transfer, regret_entry, data, api_key=None):
    return _CANNED_JUSTIFICATION


# ---------------------------------------------------------------------------
# 1. Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["version"] == "v2"


# ---------------------------------------------------------------------------
# 2. Ingest
# ---------------------------------------------------------------------------

class TestIngest:
    def test_ingest_no_api_key(self):
        """Without Gemini key, returns empty accepted/flagged."""
        r = client.post("/ingest", json={
            "facility_id": "PHC2",
            "raw_report": "metformin 500mg 200 tablets",
        })
        assert r.status_code == 200
        data = r.json()
        assert "accepted_lots" in data
        assert "flagged_lots" in data
        assert isinstance(data["accepted_lots"], list)

    def test_ingest_with_mocked_gemini(self):
        mock_response = [{
            "drug_name": "Metformin 500mg",
            "molecule": "Metformin",
            "strength": "500mg",
            "form": "tablet",
            "quantity": 200,
            "batch_number": "B1",
            "expiry_date": "2027-06-01",
            "storage": "room_temp",
        }]
        with patch("backend.genai_ingest._call_gemini", return_value=mock_response), \
             patch("backend.genai_ingest._load_api_key", return_value="fake-key"):
            r = client.post("/ingest", json={
                "facility_id": "PHC2",
                "raw_report": "metformin 500mg 200 tabs",
            })
        assert r.status_code == 200
        data = r.json()
        assert len(data["accepted_lots"]) + len(data["flagged_lots"]) >= 1


# ---------------------------------------------------------------------------
# 3. Inventory
# ---------------------------------------------------------------------------

class TestInventory:
    def test_inventory(self):
        r = client.get("/inventory")
        assert r.status_code == 200
        data = r.json()
        assert "facility_stock" in data
        assert "needs" in data
        assert "surplus_chunks" in data
        assert isinstance(data["facilities"], list)

    def test_inventory_facility(self):
        r = client.get("/inventory")
        facs = r.json()["facilities"]
        if facs:
            r2 = client.get(f"/inventory/{facs[0]}")
            assert r2.status_code == 200
            data = r2.json()
            assert data["facility_id"] == facs[0]
            assert "stock" in data
            assert "needs" in data

    def test_inventory_facility_not_found(self):
        r = client.get("/inventory/NONEXISTENT_FAC")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 5. Plan generate
# ---------------------------------------------------------------------------

class TestPlanGenerate:
    @pytest.mark.timeout(60)
    def test_greedy_plan(self):
        with patch("backend.main.generate_with_fallback", side_effect=_mock_narrate):
            r = client.post("/plan/generate", json={
                "cover_days": 14,
                "use_quantum": False,
                "solver": "greedy",
            })
        assert r.status_code == 200
        data = r.json()
        assert "plan_id" in data
        assert "transfers" in data
        assert "validation" in data
        assert "solver_used" in data
        assert data["solver_used"] == "greedy"
        assert "wall_clock_s" in data

    @pytest.mark.timeout(60)
    def test_plan_with_opt_fallback(self):
        """Without API key, opt solver falls back to greedy."""
        with patch("backend.main.generate_with_fallback", side_effect=_mock_narrate):
            r = client.post("/plan/generate", json={
                "cover_days": 14,
                "use_quantum": True,
                "solver": "opt",
            })
        assert r.status_code == 200
        data = r.json()
        assert "greedy" in data["solver_used"].lower()

    @pytest.mark.timeout(60)
    def test_plan_has_orders(self):
        with patch("backend.main.generate_with_fallback", side_effect=_mock_narrate):
            r = client.post("/plan/generate", json={
                "use_quantum": False, "solver": "greedy",
            })
        data = r.json()
        if data["transfers"]:
            assert "orders" in data
            assert len(data["orders"]) > 0
            order = data["orders"][0]
            assert "justification" in order
            assert len(order["justification"]) > 0


# ---------------------------------------------------------------------------
# Helper: generate a plan with mocked narration for downstream tests
# ---------------------------------------------------------------------------

def _generate_plan(**overrides):
    payload = {"use_quantum": False, "solver": "greedy"}
    payload.update(overrides)
    with patch("backend.main.generate_with_fallback", side_effect=_mock_narrate):
        r = client.post("/plan/generate", json=payload)
    return r.json()


# ---------------------------------------------------------------------------
# 6. Plan alternatives
# ---------------------------------------------------------------------------

class TestPlanAlternatives:
    @pytest.mark.timeout(60)
    def test_alternatives(self):
        data = _generate_plan()
        plan_id = data["plan_id"]
        r2 = client.get(f"/plan/{plan_id}/alternatives")
        assert r2.status_code == 200
        assert "summary" in r2.json()

    def test_alternatives_not_found(self):
        r = client.get("/plan/NONEXISTENT/alternatives")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 7. Order detail
# ---------------------------------------------------------------------------

class TestOrderDetail:
    @pytest.mark.timeout(60)
    def test_order_detail(self):
        data = _generate_plan()
        plan_id = data["plan_id"]
        if data.get("orders"):
            oid = data["orders"][0]["order_id"]
            r2 = client.get(f"/plan/{plan_id}/order/{oid}")
            assert r2.status_code == 200
            assert r2.json()["order_id"] == oid

    @pytest.mark.timeout(60)
    def test_order_not_found(self):
        data = _generate_plan()
        plan_id = data["plan_id"]
        r2 = client.get(f"/plan/{plan_id}/order/NONEXISTENT")
        assert r2.status_code == 404


# ---------------------------------------------------------------------------
# 8. Approve
# ---------------------------------------------------------------------------

class TestApprove:
    @pytest.mark.timeout(60)
    def test_approve_orders(self):
        data = _generate_plan()
        plan_id = data["plan_id"]
        order_ids = [o["order_id"] for o in data.get("orders", [])]

        if order_ids:
            r2 = client.post(f"/plan/{plan_id}/approve", json={
                "approved_order_ids": order_ids[:1],
            })
            assert r2.status_code == 200
            assert order_ids[0] in r2.json()["approved"]

    def test_approve_not_found(self):
        r = client.post("/plan/NONEXISTENT/approve", json={
            "approved_order_ids": ["ORD-001"],
        })
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 9. Clustering
# ---------------------------------------------------------------------------

class TestClustering:
    def test_kmeans(self):
        r = client.post("/clustering", json={"method": "kmeans"})
        assert r.status_code == 200
        data = r.json()
        assert data["method"] == "kmeans"
        assert "clusters" in data

    def test_compare(self):
        r = client.post("/clustering", json={"method": "compare"})
        assert r.status_code == 200
        data = r.json()
        assert "chosen_clusters" in data
        assert "comparison" in data


# ---------------------------------------------------------------------------
# 10. Replay
# ---------------------------------------------------------------------------

class TestReplay:
    def test_replay_single_scenario(self):
        r = client.post("/replay", json={
            "scenario": "normal_week",
            "n_seeds": 1,
            "n_weeks": 1,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["scenario"] == "normal_week"
        assert "results" in data
        assert "qubo" in data["results"]
        assert "greedy" in data["results"]
        assert "none" in data["results"]


# ---------------------------------------------------------------------------
# 11. Evidence
# ---------------------------------------------------------------------------

class TestEvidence:
    def test_evidence_listing(self):
        r = client.get("/evidence")
        assert r.status_code == 200
        assert "files" in r.json()
