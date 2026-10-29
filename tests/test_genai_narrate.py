"""Tests for GenAI narrate module.

Tests template generation, verification, and fallback logic — all
without needing a real Gemini API key (LLM calls are mocked).
"""

import os
from unittest.mock import patch

import pytest

from backend.schemas import (
    Drug, DrugTier, Facility, FacilityType, InventoryData,
    Route, TransferVariable, UsageRate,
)
from backend.genai_narrate import (
    _template_justification,
    generate_justification,
    generate_with_fallback,
    verify_justification,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drug(code="d1", name="Metformin 500mg", molecule="Metformin",
          strength="500mg", form="tablet"):
    return Drug(code=code, name=name, molecule=molecule, strength=strength,
                form=form, tier=DrugTier.essential, criticality=2,
                requires_cold_chain=False, unit_price_inr=2.0)


def _fac(fid, name):
    return Facility(id=fid, name=name, type=FacilityType.phc,
                    latitude=0.0, longitude=0.0, has_cold_storage=False)


def _tv(var_id="x_0_F2", lot="L1", src="F1", dst="F2", drug="d1",
        usable=50, expiry=30, transit=1, val_risk=500.0, cost=300.0):
    return TransferVariable(
        var_id=var_id, chunk_index=0, lot_id=lot, source_id=src,
        destination_id=dst, drug_code=drug, chunk_quantity=100,
        usable_quantity=usable, days_to_expiry=expiry, transit_days=transit,
        value_at_risk=val_risk, transport_cost=cost,
    )


def _data(drug_code="d1", src="F1", dst="F2", dst_stock=0, usage=10.0):
    drug = _drug(code=drug_code)
    route = Route(source_id=src, destination_id=dst, transit_days=1,
                  trip_cost_inr=300.0, has_cold_transport=False)
    usage_rate = UsageRate(facility_id=dst, drug_code=drug_code,
                           units_per_day=usage)
    return InventoryData(
        facilities={src: _fac(src, f"PHC {src}"), dst: _fac(dst, f"GH {dst}")},
        drugs={drug_code: drug},
        lots=[],
        routes=[route],
        usage_rates={f"{dst}|{drug_code}": usage_rate},
        reserves={},
        route_map={f"{src}|{dst}": route},
        stock_by_facility_drug={
            f"{src}|{drug_code}": 200,
            f"{dst}|{drug_code}": dst_stock,
        },
        needs={f"{dst}|{drug_code}": 50},
    )


# ---------------------------------------------------------------------------
# Test: template justification
# ---------------------------------------------------------------------------

class TestTemplateJustification:
    def test_basic_template(self):
        tv = _tv(usable=50, expiry=30)
        data = _data(dst_stock=0, usage=10.0)
        text = _template_justification(tv, data)

        assert "50 units" in text
        assert "Metformin 500mg" in text
        assert "30 days" in text
        assert "GH F2" in text
        assert "PHC F1" in text
        assert "5 days" in text  # 50/10 = 5

    def test_template_with_regret(self):
        tv = _tv()
        data = _data()
        regret = {
            "next_best_destination": "PHC-C",
            "selection_reason": "transport cost is 40% lower",
        }
        text = _template_justification(tv, data, regret)
        assert "PHC-C" in text
        assert "40% lower" in text

    def test_template_zero_usage(self):
        tv = _tv()
        data = _data(usage=0.0)
        text = _template_justification(tv, data)
        assert "not available" in text

    def test_template_missing_facilities(self):
        tv = _tv()
        data = _data()
        data.facilities = {}
        text = _template_justification(tv, data)
        assert "F1" in text
        assert "F2" in text


# ---------------------------------------------------------------------------
# Test: verify_justification
# ---------------------------------------------------------------------------

class TestVerifyJustification:
    def test_correct_numbers_pass(self):
        tv = _tv(usable=50, expiry=30, transit=1, cost=300.0, val_risk=500.0)
        text = (
            "Lot L1 of Metformin 500mg (expires in 30 days) is being "
            "moved from PHC F1 to GH F2. The 50 units will last about "
            "5 days at 10 units/day. Transport cost is 300."
        )
        ok, reason = verify_justification(text, tv)
        assert ok is True
        assert reason == ""

    def test_wrong_quantity_fails(self):
        tv = _tv(usable=50, expiry=30)
        text = (
            "The 51 units of Metformin will be transferred. "
            "Expires in 30 days."
        )
        ok, reason = verify_justification(text, tv)
        assert ok is False
        assert "quantity" in reason.lower() or "50" in reason or "51" in reason

    def test_empty_text_fails(self):
        tv = _tv()
        ok, reason = verify_justification("", tv)
        assert ok is False
        assert "Empty" in reason

    def test_no_numbers_passes(self):
        """Text with no numeric content passes (nothing to mismatch)."""
        tv = _tv()
        ok, _ = verify_justification(
            "This transfer is justified by surplus at source and shortage at destination.",
            tv,
        )
        assert ok is True


# ---------------------------------------------------------------------------
# Test: generate_justification (mocked Gemini)
# ---------------------------------------------------------------------------

class TestGenerateJustification:
    def test_returns_llm_text(self):
        tv = _tv(usable=50, expiry=30)
        data = _data()
        llm_text = "50 units of Metformin moved because GH F2 has 0 stock."

        with patch("backend.genai_narrate._call_gemini", return_value=llm_text):
            result = generate_justification(tv, None, data, api_key="fake")

        assert result == llm_text

    def test_returns_empty_on_failure(self):
        tv = _tv()
        data = _data()

        with patch("backend.genai_narrate._call_gemini", return_value=None):
            result = generate_justification(tv, None, data, api_key="fake")

        assert result == ""

    def test_returns_empty_without_key(self):
        tv = _tv()
        data = _data()
        with patch("backend.genai_narrate._load_api_key", return_value=None):
            result = generate_justification(tv, None, data, api_key=None)
        assert result == ""


# ---------------------------------------------------------------------------
# Test: generate_with_fallback
# ---------------------------------------------------------------------------

class TestGenerateWithFallback:
    def test_uses_genai_when_verified(self):
        tv = _tv(usable=50, expiry=30)
        data = _data()
        llm_text = (
            "Lot L1 of Metformin 500mg (expires in 30 days) is being moved. "
            "The 50 units will supply GH F2 for about 5 days."
        )

        with patch("backend.genai_narrate._call_gemini", return_value=llm_text):
            result = generate_with_fallback(tv, None, data, api_key="fake")

        assert result == llm_text

    def test_falls_back_on_verification_failure(self):
        tv = _tv(usable=50, expiry=30)
        data = _data()
        bad_text = "The 99 units will be transferred in 30 days."

        with patch("backend.genai_narrate._call_gemini", return_value=bad_text):
            result = generate_with_fallback(tv, None, data, api_key="fake")

        assert "Transfer Order" in result
        assert "50 units" in result
        assert "30 days" in result

    def test_falls_back_on_llm_failure(self):
        tv = _tv(usable=50, expiry=30)
        data = _data()

        with patch("backend.genai_narrate._call_gemini", return_value=None):
            result = generate_with_fallback(tv, None, data, api_key="fake")

        assert "Transfer Order" in result
        assert "50 units" in result

    def test_falls_back_without_api_key(self):
        tv = _tv(usable=50, expiry=30)
        data = _data()
        result = generate_with_fallback(tv, None, data, api_key=None)
        assert "Transfer Order" in result
        assert "50 units" in result

    def test_fallback_includes_regret(self):
        tv = _tv(usable=50, expiry=30)
        data = _data()
        regret = {
            "next_best_destination": "PHC-C",
            "selection_reason": "lower cost",
        }

        with patch("backend.genai_narrate._call_gemini", return_value=None):
            result = generate_with_fallback(tv, regret, data, api_key="fake")

        assert "PHC-C" in result
        assert "lower cost" in result

    def test_never_returns_empty(self):
        tv = _tv()
        data = _data()

        with patch("backend.genai_narrate._call_gemini", return_value=None):
            result = generate_with_fallback(tv, None, data, api_key="fake")

        assert len(result) > 0
