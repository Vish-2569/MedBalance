"""Tests for GenAI ingest module.

Tests drug matching, validation, and extraction pipeline — all without
needing a real Gemini API key (LLM calls are mocked).
"""

from datetime import date
from unittest.mock import patch

import pytest

from backend.schemas import Drug, DrugTier
from backend.genai_ingest import (
    ParsedLot,
    _build_lookup,
    _levenshtein,
    _normalise,
    _validate_expiry,
    _validate_quantity,
    match_drug,
    parse_stock_report,
    validate_extraction,
    SAMPLE_REPORTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drug(code="d1", name="Drug 1", molecule="Mol", strength="500mg",
          form="tablet", tier=DrugTier.essential, cold=False, price=10.0):
    return Drug(code=code, name=name, molecule=molecule, strength=strength,
                form=form, tier=tier, criticality=2,
                requires_cold_chain=cold, unit_price_inr=price)


DRUG_MASTER = [
    _drug("insulin_40iu_ml_injection", "Insulin 40 IU/ml", "Insulin",
          "40 IU/ml", "injection", DrugTier.life_saving, cold=True, price=45.0),
    _drug("metformin_500mg_tablet", "Metformin 500mg", "Metformin",
          "500mg", "tablet", price=2.0),
    _drug("amoxicillin_500mg_capsule", "Amoxicillin 500mg", "Amoxicillin",
          "500mg", "capsule", price=5.0),
    _drug("paracetamol_500mg_tablet", "Paracetamol 500mg", "Paracetamol",
          "500mg", "tablet", DrugTier.routine, price=1.0),
    _drug("ors_sachet", "ORS Sachet", "ORS", "", "sachet", price=8.0),
    _drug("amlodipine_5mg_tablet", "Amlodipine 5mg", "Amlodipine",
          "5mg", "tablet", DrugTier.life_saving, price=3.0),
    _drug("salbutamol_100mcg_inhaler", "Salbutamol 100mcg", "Salbutamol",
          "100mcg", "inhaler", DrugTier.life_saving, price=85.0),
]


# ---------------------------------------------------------------------------
# Test: normalisation
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_basic(self):
        assert _normalise("  Insulin  ") == "insulin"

    def test_collapses_whitespace(self):
        assert _normalise("40  IU /  ml") == "40 iu / ml"


# ---------------------------------------------------------------------------
# Test: Levenshtein distance
# ---------------------------------------------------------------------------

class TestLevenshtein:
    def test_identical(self):
        assert _levenshtein("abc", "abc") == 0

    def test_one_edit(self):
        assert _levenshtein("abc", "ab") == 1

    def test_empty(self):
        assert _levenshtein("", "abc") == 3

    def test_symmetric(self):
        assert _levenshtein("kitten", "sitting") == _levenshtein("sitting", "kitten")


# ---------------------------------------------------------------------------
# Test: drug matching
# ---------------------------------------------------------------------------

class TestMatchDrug:
    def setup_method(self):
        self.lookup = _build_lookup(DRUG_MASTER)

    def test_exact_match(self):
        drug, conf = match_drug("Metformin", "500mg", "tablet",
                                self.lookup, DRUG_MASTER)
        assert drug is not None
        assert drug.code == "metformin_500mg_tablet"
        assert conf == 1.0

    def test_exact_match_case_insensitive(self):
        drug, conf = match_drug("INSULIN", "40 IU/ml", "INJECTION",
                                self.lookup, DRUG_MASTER)
        assert drug is not None
        assert drug.code == "insulin_40iu_ml_injection"
        assert conf == 1.0

    def test_synonym_expansion_form(self):
        """'tab' should expand to 'tablet'."""
        drug, conf = match_drug("Paracetamol", "500mg", "tab",
                                self.lookup, DRUG_MASTER)
        assert drug is not None
        assert drug.code == "paracetamol_500mg_tablet"
        assert conf == 1.0

    def test_synonym_expansion_molecule(self):
        """'MET' should expand to 'Metformin'."""
        drug, conf = match_drug("MET", "500mg", "tablet",
                                self.lookup, DRUG_MASTER)
        assert drug is not None
        assert drug.code == "metformin_500mg_tablet"
        assert conf == 1.0

    def test_fuzzy_match_misspelling(self):
        """'Metformn' (one char off) should fuzzy match."""
        drug, conf = match_drug("Metformn", "500mg", "tablet",
                                self.lookup, DRUG_MASTER)
        assert drug is not None
        assert drug.code == "metformin_500mg_tablet"
        assert 0.5 <= conf < 1.0

    def test_no_match(self):
        drug, conf = match_drug("Zylothine", "100mg", "tablet",
                                self.lookup, DRUG_MASTER)
        assert drug is None
        assert conf == 0.0

    def test_ors_no_strength(self):
        """ORS has no strength — should still match."""
        drug, conf = match_drug("ORS", "", "sachet",
                                self.lookup, DRUG_MASTER)
        assert drug is not None
        assert drug.code == "ors_sachet"


# ---------------------------------------------------------------------------
# Test: quantity validation
# ---------------------------------------------------------------------------

class TestValidateQuantity:
    def test_valid(self):
        ok, val = _validate_quantity(100)
        assert ok is True
        assert val == 100

    def test_zero(self):
        ok, val = _validate_quantity(0)
        assert ok is False
        assert val == 0

    def test_negative(self):
        ok, val = _validate_quantity(-5)
        assert ok is False

    def test_string_number(self):
        ok, val = _validate_quantity("42")
        assert ok is True
        assert val == 42

    def test_invalid_string(self):
        ok, val = _validate_quantity("abc")
        assert ok is False


# ---------------------------------------------------------------------------
# Test: expiry validation
# ---------------------------------------------------------------------------

class TestValidateExpiry:
    def test_valid_future(self):
        ok, days = _validate_expiry("2027-06-15", date(2026, 10, 29))
        assert ok is True
        assert days > 0

    def test_past_date(self):
        ok, days = _validate_expiry("2025-01-01", date(2026, 10, 29))
        assert ok is False
        assert days == 0

    def test_empty_string(self):
        ok, days = _validate_expiry("", date(2026, 10, 29))
        assert ok is False

    def test_invalid_format(self):
        ok, days = _validate_expiry("15/06/2027", date(2026, 10, 29))
        assert ok is False

    def test_today_is_expired(self):
        ok, days = _validate_expiry("2026-10-29", date(2026, 10, 29))
        assert ok is False


# ---------------------------------------------------------------------------
# Test: parse_stock_report with mocked Gemini
# ---------------------------------------------------------------------------

MOCK_GEMINI_RESPONSE = [
    {
        "drug_name": "Insulin 40 IU/ml",
        "molecule": "Insulin",
        "strength": "40 IU/ml",
        "form": "injection",
        "quantity": 3,
        "batch_number": "INS-2024-001",
        "expiry_date": "2027-06-15",
        "storage": "cold_2_8c",
    },
    {
        "drug_name": "Metformin 500mg",
        "molecule": "Metformin",
        "strength": "500mg",
        "form": "tablet",
        "quantity": 200,
        "batch_number": "MET-2024-050",
        "expiry_date": "2027-03-20",
        "storage": "room_temp",
    },
    {
        "drug_name": "Unknown Drug",
        "molecule": "Zylothine",
        "strength": "100mg",
        "form": "tablet",
        "quantity": 50,
        "batch_number": "UNK-001",
        "expiry_date": "2027-01-01",
        "storage": "room_temp",
    },
]


class TestParseStockReport:
    def test_parses_with_mocked_gemini(self):
        with patch("backend.genai_ingest._call_gemini", return_value=MOCK_GEMINI_RESPONSE):
            lots = parse_stock_report(
                "dummy text", "PHC2", DRUG_MASTER,
                api_key="fake-key", today=date(2026, 10, 29),
            )

        assert len(lots) == 3

        insulin = lots[0]
        assert insulin.drug_code == "insulin_40iu_ml_injection"
        assert insulin.quantity == 3
        assert insulin.confidence == 1.0
        assert insulin.storage_condition == "cold_2_8c"
        assert insulin.days_to_expiry > 0

        metformin = lots[1]
        assert metformin.drug_code == "metformin_500mg_tablet"
        assert metformin.quantity == 200
        assert metformin.confidence == 1.0

        unknown = lots[2]
        assert unknown.drug_code == "UNKNOWN"
        assert unknown.flagged is True
        assert "no_drug_match" in unknown.flag_reason

    def test_returns_empty_when_gemini_fails(self):
        with patch("backend.genai_ingest._call_gemini", return_value=None):
            lots = parse_stock_report(
                "dummy", "PHC2", DRUG_MASTER,
                api_key="fake-key", today=date(2026, 10, 29),
            )
        assert lots == []

    def test_returns_empty_without_api_key(self):
        lots = parse_stock_report(
            "dummy", "PHC2", DRUG_MASTER,
            api_key=None, today=date(2026, 10, 29),
        )
        assert lots == []

    def test_flags_low_confidence(self):
        fuzzy_response = [{
            "drug_name": "Metformn",
            "molecule": "Metformn",
            "strength": "500mg",
            "form": "tablet",
            "quantity": 100,
            "batch_number": "B1",
            "expiry_date": "2027-06-01",
            "storage": "room_temp",
        }]
        with patch("backend.genai_ingest._call_gemini", return_value=fuzzy_response):
            lots = parse_stock_report(
                "dummy", "F1", DRUG_MASTER,
                api_key="fake-key", today=date(2026, 10, 29),
            )
        assert len(lots) == 1
        assert lots[0].drug_code == "metformin_500mg_tablet"
        assert lots[0].confidence < 1.0

    def test_flags_invalid_quantity(self):
        bad_qty = [{
            "drug_name": "Metformin",
            "molecule": "Metformin",
            "strength": "500mg",
            "form": "tablet",
            "quantity": -5,
            "batch_number": "B1",
            "expiry_date": "2027-06-01",
            "storage": "room_temp",
        }]
        with patch("backend.genai_ingest._call_gemini", return_value=bad_qty):
            lots = parse_stock_report(
                "dummy", "F1", DRUG_MASTER,
                api_key="fake-key", today=date(2026, 10, 29),
            )
        assert lots[0].flagged is True
        assert "invalid_quantity" in lots[0].flag_reason

    def test_flags_expired_date(self):
        expired = [{
            "drug_name": "Metformin",
            "molecule": "Metformin",
            "strength": "500mg",
            "form": "tablet",
            "quantity": 100,
            "batch_number": "B1",
            "expiry_date": "2025-01-01",
            "storage": "room_temp",
        }]
        with patch("backend.genai_ingest._call_gemini", return_value=expired):
            lots = parse_stock_report(
                "dummy", "F1", DRUG_MASTER,
                api_key="fake-key", today=date(2026, 10, 29),
            )
        assert lots[0].flagged is True
        assert "invalid_expiry" in lots[0].flag_reason


# ---------------------------------------------------------------------------
# Test: validate_extraction
# ---------------------------------------------------------------------------

class TestValidateExtraction:
    def test_splits_accepted_and_flagged(self):
        lots = [
            ParsedLot("L1", "F1", "d1", "Drug", 100, "B1", 30,
                       "room_temp", 1.0, flagged=False),
            ParsedLot("L2", "F1", "d2", "Drug2", 50, "B2", 20,
                       "room_temp", 0.6, flagged=True, flag_reason="low_confidence"),
            ParsedLot("L3", "F1", "UNKNOWN", "X", 0, "", 0,
                       "room_temp", 0.0, flagged=True, flag_reason="no_drug_match"),
        ]
        accepted, flagged = validate_extraction(lots, DRUG_MASTER)
        assert len(accepted) == 1
        assert len(flagged) == 2
        assert accepted[0].lot_id == "L1"

    def test_all_accepted(self):
        lots = [
            ParsedLot("L1", "F1", "d1", "Drug", 100, "B1", 30,
                       "room_temp", 1.0),
        ]
        accepted, flagged = validate_extraction(lots, DRUG_MASTER)
        assert len(accepted) == 1
        assert len(flagged) == 0

    def test_empty(self):
        accepted, flagged = validate_extraction([], DRUG_MASTER)
        assert accepted == []
        assert flagged == []


# ---------------------------------------------------------------------------
# Test: sample reports exist
# ---------------------------------------------------------------------------

class TestSampleReports:
    def test_three_formats(self):
        assert "free_text" in SAMPLE_REPORTS
        assert "semi_structured" in SAMPLE_REPORTS
        assert "abbreviated" in SAMPLE_REPORTS
        for key, text in SAMPLE_REPORTS.items():
            assert len(text) > 20, f"{key} too short"
