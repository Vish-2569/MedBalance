"""GenAI ingest: uses Gemini to parse unstructured stock reports into
structured lot records, validated against the canonical drug master list.

SDK: google-genai 2.19+ (new Client API).
Model: gemini-3.8-flash, temperature=0.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime

from dotenv import load_dotenv

from .schemas import Drug

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ParsedLot:
    lot_id: str
    facility_id: str
    drug_code: str
    drug_name_raw: str
    quantity: int
    batch_number: str
    days_to_expiry: int
    storage_condition: str
    confidence: float
    flagged: bool = False
    flag_reason: str = ""


# ---------------------------------------------------------------------------
# Drug master helpers
# ---------------------------------------------------------------------------


def _normalise(s: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _build_lookup(drugs: list[Drug]) -> dict[str, Drug]:
    """Build canonical lookup keyed by normalised (molecule|strength|form)."""
    lookup: dict[str, Drug] = {}
    for d in drugs:
        key = f"{_normalise(d.molecule)}|{_normalise(d.strength)}|{_normalise(d.form)}"
        lookup[key] = d
    return lookup


# Common abbreviation synonyms
_FORM_SYNONYMS: dict[str, str] = {
    "tab": "tablet", "tabs": "tablet", "cap": "capsule", "caps": "capsule",
    "inj": "injection", "vial": "injection", "vials": "injection",
    "inh": "inhaler",
}

_MOLECULE_SYNONYMS: dict[str, str] = {
    "ins": "insulin", "met": "metformin", "amox": "amoxicillin",
    "para": "paracetamol", "pcm": "paracetamol", "amlo": "amlodipine",
    "oxy": "oxytocin", "ifa": "iron + folic acid", "azithro": "azithromycin",
    "salbu": "salbutamol",
}


def _expand_synonym(val: str, table: dict[str, str]) -> str:
    n = _normalise(val)
    return table.get(n, n)


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def match_drug(
    molecule: str,
    strength: str,
    form: str,
    lookup: dict[str, Drug],
    drugs: list[Drug],
) -> tuple[Drug | None, float]:
    """Match extracted drug info to canonical Drug.

    Returns (matched_drug, confidence). Confidence:
      1.0  = exact match on molecule+strength+form
      0.5-0.9 = fuzzy match
      0.0  = no match
    """
    mol_n = _expand_synonym(molecule, _MOLECULE_SYNONYMS)
    form_n = _expand_synonym(form, _FORM_SYNONYMS)
    str_n = _normalise(strength)

    exact_key = f"{mol_n}|{str_n}|{form_n}"
    if exact_key in lookup:
        return lookup[exact_key], 1.0

    without_strength = f"{mol_n}||{form_n}"
    for key, drug in lookup.items():
        parts = key.split("|")
        if parts[0] == mol_n and parts[2] == form_n and parts[1] == "":
            return drug, 0.95

    best_drug: Drug | None = None
    best_score = 0.0
    for drug in drugs:
        d_mol = _normalise(drug.molecule)
        d_form = _normalise(drug.form)
        d_str = _normalise(drug.strength)

        mol_dist = _levenshtein(mol_n, d_mol)
        max_len = max(len(mol_n), len(d_mol), 1)
        mol_sim = 1.0 - mol_dist / max_len

        if mol_sim < 0.6:
            continue

        form_match = 1.0 if form_n == d_form else 0.5
        str_match = 1.0 if str_n == d_str else (0.7 if str_n == "" or d_str == "" else 0.3)

        score = mol_sim * 0.6 + form_match * 0.2 + str_match * 0.2

        if score > best_score:
            best_score = score
            best_drug = drug

    if best_drug and best_score >= 0.5:
        confidence = min(0.9, best_score)
        return best_drug, confidence

    return None, 0.0


# ---------------------------------------------------------------------------
# Gemini LLM call
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """You are a pharmaceutical inventory parser. Extract lot-level records from this stock report.

Return a JSON array where each element has:
- drug_name: string (the drug name as written)
- molecule: string (active ingredient, e.g. "Metformin", "Insulin")
- strength: string (e.g. "500mg", "40 IU/ml", "" if not specified)
- form: string (e.g. "tablet", "capsule", "injection", "inhaler", "sachet")
- quantity: integer (number of units)
- batch_number: string (batch/lot number, or "" if not mentioned)
- expiry_date: string (YYYY-MM-DD format, or "" if not mentioned)
- storage: string ("room_temp" or "cold_2_8c")

Rules:
- Be exact — do not guess or hallucinate.
- If a field is not mentioned, use "" for strings, 0 for integers.
- Infer form from context (e.g. "vials" → injection, "tabs" → tablet, "caps" → capsule).
- Return ONLY the JSON array, no other text.

Stock report:
{report_text}"""


def _call_gemini(raw_text: str, api_key: str) -> list[dict] | None:
    """Call Gemini to extract structured records. Returns parsed JSON or None."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        prompt = _EXTRACTION_PROMPT.format(report_text=raw_text)

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=4096,
                response_mime_type="application/json",
                http_options=types.HttpOptions(timeout=30_000),
            ),
        )

        text = response.text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        records = json.loads(text)
        if not isinstance(records, list):
            logger.error("Gemini returned non-list: %s", type(records))
            return None
        return records

    except json.JSONDecodeError as e:
        logger.error("JSON parse failed on Gemini response: %s", e)
        return None
    except Exception as e:
        logger.error("Gemini call failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Offline extraction (no LLM)
# ---------------------------------------------------------------------------

def _call_gemini_offline(raw_text: str) -> list[dict] | None:
    """Stub for testing without LLM. Returns None (triggers manual queue)."""
    return None


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def parse_stock_report(
    raw_text: str,
    facility_id: str,
    drug_master: list[Drug],
    api_key: str | None = None,
    today: date | None = None,
) -> list[ParsedLot]:
    """Parse a raw stock report into structured lot records.

    Steps:
      1. Call Gemini to extract structured records
      2. Match each record to canonical drug code
      3. Validate quantities and dates
      4. Return ParsedLot list with confidence scores

    If Gemini fails, returns empty list (report queued for manual entry).
    """
    if today is None:
        today = date.today()

    key = api_key or _load_api_key()
    if key:
        records = _call_gemini(raw_text, key)
    else:
        logger.warning("No GEMINI_API_KEY — skipping LLM extraction")
        records = None

    if records is None:
        logger.warning("LLM extraction failed for %s; queuing for manual entry", facility_id)
        return []

    lookup = _build_lookup(drug_master)
    lots: list[ParsedLot] = []

    for i, rec in enumerate(records):
        molecule = str(rec.get("molecule", ""))
        strength = str(rec.get("strength", ""))
        form = str(rec.get("form", ""))
        drug_name_raw = str(rec.get("drug_name", ""))
        quantity = rec.get("quantity", 0)
        batch = str(rec.get("batch_number", ""))
        expiry_str = str(rec.get("expiry_date", ""))
        storage = str(rec.get("storage", "room_temp"))

        matched_drug, confidence = match_drug(molecule, strength, form, lookup, drug_master)

        if matched_drug is None:
            lots.append(ParsedLot(
                lot_id=f"GEN-{facility_id}-{i:03d}",
                facility_id=facility_id,
                drug_code="UNKNOWN",
                drug_name_raw=drug_name_raw,
                quantity=max(0, int(quantity)) if isinstance(quantity, (int, float)) else 0,
                batch_number=batch,
                days_to_expiry=0,
                storage_condition=storage,
                confidence=0.0,
                flagged=True,
                flag_reason="no_drug_match",
            ))
            continue

        qty_valid, qty_val = _validate_quantity(quantity)
        expiry_valid, days_exp = _validate_expiry(expiry_str, today)

        flag_reasons = []
        if not qty_valid:
            flag_reasons.append("invalid_quantity")
        if expiry_str and not expiry_valid:
            flag_reasons.append("invalid_expiry")
        if confidence < 0.8:
            flag_reasons.append("low_confidence")

        flagged = len(flag_reasons) > 0

        if storage not in ("room_temp", "cold_2_8c"):
            storage = "cold_2_8c" if matched_drug.requires_cold_chain else "room_temp"

        lots.append(ParsedLot(
            lot_id=f"GEN-{facility_id}-{i:03d}",
            facility_id=facility_id,
            drug_code=matched_drug.code,
            drug_name_raw=drug_name_raw,
            quantity=qty_val,
            batch_number=batch,
            days_to_expiry=days_exp,
            storage_condition=storage,
            confidence=confidence,
            flagged=flagged,
            flag_reason="; ".join(flag_reasons),
        ))

    return lots


def _validate_quantity(qty) -> tuple[bool, int]:
    """Validate and sanitise quantity. Returns (is_valid, cleaned_value)."""
    try:
        val = int(qty)
        if val <= 0:
            return False, 0
        return True, val
    except (ValueError, TypeError):
        return False, 0


def _validate_expiry(expiry_str: str, today: date) -> tuple[bool, int]:
    """Validate expiry date string. Returns (is_valid, days_to_expiry)."""
    if not expiry_str:
        return False, 0
    try:
        exp_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        delta = (exp_date - today).days
        if delta <= 0:
            return False, 0
        return True, delta
    except ValueError:
        return False, 0


# ---------------------------------------------------------------------------
# Validation / split
# ---------------------------------------------------------------------------


def validate_extraction(
    parsed_lots: list[ParsedLot],
    drug_master: list[Drug],
) -> tuple[list[ParsedLot], list[ParsedLot]]:
    """Split parsed lots into accepted and flagged.

    Accepted: confidence >= 0.8, valid quantity, valid or missing expiry.
    Flagged: everything else (needs human review).
    """
    accepted: list[ParsedLot] = []
    flagged: list[ParsedLot] = []

    for lot in parsed_lots:
        if lot.flagged:
            flagged.append(lot)
        else:
            accepted.append(lot)

    return accepted, flagged


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------


def _load_api_key() -> str | None:
    load_dotenv()
    key = os.environ.get("GEMINI_API_KEY")
    if not key or key.startswith("PASTE"):
        return None
    return key


# ---------------------------------------------------------------------------
# Sample stock reports (for testing / demo)
# ---------------------------------------------------------------------------

SAMPLE_REPORTS = {
    "free_text": (
        "PHC Rampur monthly stock report:\n"
        "Insulin 40 IU/ml - 3 vials remaining, stored in fridge (2-8°C). "
        "Batch INS-2024-001, expires 2026-06-15.\n"
        "Metformin 500mg tablets - 200 tablets in stock. "
        "Batch MET-2024-050, exp 2027-03-20.\n"
        "Amoxicillin 500mg - running low, only 15 capsules left. "
        "Batch AMX-2024-100, expiry 2026-12-01.\n"
        "Paracetamol 500mg tablets - 500 tablets. "
        "Batch PCM-2024-200, exp 2027-08-10."
    ),

    "semi_structured": (
        "FACILITY: CHC Deoria\n"
        "DATE: 2026-10-28\n"
        "┌─────────────────┬──────────┬─────┬────────────────┬────────────┬─────────┐\n"
        "│ Drug             │ Form     │ Qty │ Batch          │ Expiry     │ Storage │\n"
        "├─────────────────┼──────────┼─────┼────────────────┼────────────┼─────────┤\n"
        "│ Amlodipine 5mg   │ tablet   │ 120 │ AML-2025-010   │ 2027-05-01 │ room    │\n"
        "│ ORS              │ sachet   │  80 │ ORS-2025-005   │ 2027-01-15 │ room    │\n"
        "│ Oxytocin 5 IU    │ injection│  10 │ OXY-2025-003   │ 2026-08-20 │ cold    │\n"
        "│ Azithromycin 500 │ tablet   │  45 │ AZT-2025-020   │ 2027-09-30 │ room    │\n"
        "└─────────────────┴──────────┴─────┴────────────────┴────────────┴─────────┘"
    ),

    "abbreviated": (
        "PHC3 quick count: "
        "INS 40IU 3v cold B:INS-001 E:2026-06-15, "
        "MET 500 200t B:MET-050 E:2027-03-20, "
        "AMOX 500 15c B:AMX-100 E:2026-12-01, "
        "SALBU 100mcg 5inh B:SAL-010 E:2027-11-01"
    ),
}
