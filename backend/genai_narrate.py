"""GenAI narrate: generates human-readable transfer justifications.

Each transfer order carries a one-paragraph explanation produced by Gemini
(with deterministic template fallback on any failure).
"""

from __future__ import annotations

import logging
import os
import re

from dotenv import load_dotenv

from .schemas import InventoryData, TransferVariable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic template (always-available fallback)
# ---------------------------------------------------------------------------

def _template_justification(
    transfer: TransferVariable,
    data: InventoryData,
    regret_entry: dict | None = None,
) -> str:
    drug = data.drugs.get(transfer.drug_code)
    drug_name = drug.name if drug else transfer.drug_code

    src_name = transfer.source_id
    dst_name = transfer.destination_id
    if data.facilities:
        src_fac = data.facilities.get(transfer.source_id)
        dst_fac = data.facilities.get(transfer.destination_id)
        if src_fac:
            src_name = src_fac.name
        if dst_fac:
            dst_name = dst_fac.name

    dst_stock_key = f"{transfer.destination_id}|{transfer.drug_code}"
    dst_stock = data.stock_by_facility_drug.get(dst_stock_key, 0)

    rate_key = f"{transfer.destination_id}|{transfer.drug_code}"
    usage = data.usage_rates.get(rate_key)
    usage_per_day = usage.units_per_day if usage else 0.0

    if usage_per_day > 0:
        days_supply = transfer.usable_quantity / usage_per_day
        consumption_note = (
            f"Destination daily usage: {usage_per_day:.1f} units/day. "
            f"Transferred quantity will be consumed in approximately "
            f"{days_supply:.0f} days."
        )
    else:
        consumption_note = "Destination usage rate not available."

    regret_note = ""
    if regret_entry:
        alt = regret_entry.get("next_best_destination", "")
        reason = regret_entry.get("selection_reason", "")
        if alt:
            regret_note = (
                f" Next-best alternative was {alt}; "
                f"this destination was chosen because {reason}."
                if reason else
                f" Next-best alternative was {alt}."
            )

    return (
        f"Transfer Order: {transfer.usable_quantity} units of {drug_name} "
        f"(Lot {transfer.lot_id}, expires in {transfer.days_to_expiry} days) "
        f"from {src_name} to {dst_name}. "
        f"Destination current stock: {dst_stock} units. "
        f"{consumption_note} "
        f"Reason: {src_name} has surplus above reserve; "
        f"{dst_name} has shortage below cover threshold."
        f"{regret_note}"
    )


# ---------------------------------------------------------------------------
# Gemini LLM justification
# ---------------------------------------------------------------------------

_JUSTIFICATION_PROMPT = """You are a pharmaceutical supply-chain analyst. Write ONE paragraph (3-5 sentences) justifying this medicine transfer order. Be factual and cite the exact numbers provided. Do not add information that is not in the data below.

Transfer data:
- Lot ID: {lot_id}
- Drug: {drug_name} ({drug_code})
- Source: {src_name} (facility {src_id})
- Source stock of this drug: {src_stock} units
- Destination: {dst_name} (facility {dst_id})
- Destination stock of this drug: {dst_stock} units
- Quantity transferred: {quantity} units (usable before expiry)
- Days to expiry: {days_to_expiry}
- Transit days: {transit_days}
- Destination usage rate: {usage_per_day} units/day
- Days of supply this provides: {days_supply}
- Value at risk (waste if not transferred): ₹{value_at_risk:.0f}
- Transport cost: ₹{transport_cost:.0f}
{regret_context}

Write a clear, concise justification paragraph. Use the exact numbers above — do not round or approximate them differently. Do not use markdown formatting."""


def _build_prompt(
    transfer: TransferVariable,
    data: InventoryData,
    regret_entry: dict | None = None,
) -> str:
    drug = data.drugs.get(transfer.drug_code)
    drug_name = drug.name if drug else transfer.drug_code

    src_name = transfer.source_id
    dst_name = transfer.destination_id
    if data.facilities:
        src_fac = data.facilities.get(transfer.source_id)
        dst_fac = data.facilities.get(transfer.destination_id)
        if src_fac:
            src_name = src_fac.name
        if dst_fac:
            dst_name = dst_fac.name

    src_stock_key = f"{transfer.source_id}|{transfer.drug_code}"
    src_stock = data.stock_by_facility_drug.get(src_stock_key, 0)

    dst_stock_key = f"{transfer.destination_id}|{transfer.drug_code}"
    dst_stock = data.stock_by_facility_drug.get(dst_stock_key, 0)

    rate_key = f"{transfer.destination_id}|{transfer.drug_code}"
    usage = data.usage_rates.get(rate_key)
    usage_per_day = usage.units_per_day if usage else 0.0

    days_supply = (
        transfer.usable_quantity / usage_per_day if usage_per_day > 0 else 0.0
    )

    regret_context = ""
    if regret_entry:
        alt = regret_entry.get("next_best_destination", "")
        reason = regret_entry.get("selection_reason", "")
        if alt:
            regret_context = (
                f"- Next-best destination: {alt}\n"
                f"- Reason this destination was chosen over alternative: {reason}"
            )

    return _JUSTIFICATION_PROMPT.format(
        lot_id=transfer.lot_id,
        drug_name=drug_name,
        drug_code=transfer.drug_code,
        src_name=src_name,
        src_id=transfer.source_id,
        src_stock=src_stock,
        dst_name=dst_name,
        dst_id=transfer.destination_id,
        dst_stock=dst_stock,
        quantity=transfer.usable_quantity,
        days_to_expiry=transfer.days_to_expiry,
        transit_days=transfer.transit_days,
        usage_per_day=f"{usage_per_day:.1f}",
        days_supply=f"{days_supply:.0f}",
        value_at_risk=transfer.value_at_risk,
        transport_cost=transfer.transport_cost,
        regret_context=regret_context,
    )


def _call_gemini(prompt: str, api_key: str) -> str | None:
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=512,
                http_options=types.HttpOptions(timeout=30_000),
            ),
        )
        text = response.text.strip()
        if not text:
            return None
        return text

    except Exception as e:
        logger.error("Gemini narration call failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_justification(
    transfer: TransferVariable,
    regret_entry: dict | None,
    data: InventoryData,
    api_key: str | None = None,
) -> str:
    """Generate a one-paragraph justification via Gemini.

    Returns the LLM-generated text, or empty string on failure.
    """
    key = api_key or _load_api_key()
    if not key:
        logger.warning("No GEMINI_API_KEY — cannot generate LLM justification")
        return ""

    prompt = _build_prompt(transfer, data, regret_entry)
    text = _call_gemini(prompt, key)
    return text or ""


def verify_justification(
    justification: str,
    transfer: TransferVariable,
) -> tuple[bool, str]:
    """Verify that every number in the justification matches the transfer record.

    Extracts all integers and floats from the text, then checks each against
    known numeric fields. A number that is close to a known field but not
    equal triggers a mismatch.
    """
    if not justification:
        return False, "Empty justification"

    known = {
        "quantity": float(transfer.usable_quantity),
        "days_to_expiry": float(transfer.days_to_expiry),
        "transit_days": float(transfer.transit_days),
        "transport_cost": float(transfer.transport_cost),
        "value_at_risk": float(transfer.value_at_risk),
        "chunk_quantity": float(transfer.chunk_quantity),
    }

    text_numbers: list[float] = []
    for m in re.finditer(r"(?<![.\w])(\d+(?:\.\d+)?)(?![.\w])", justification):
        try:
            text_numbers.append(float(m.group(1)))
        except ValueError:
            continue

    known_set = set(known.values())
    mismatches: list[str] = []

    for n in text_numbers:
        if n in known_set or n == 0.0:
            continue
        for field_name, actual in known.items():
            if actual == 0.0:
                continue
            ratio = n / actual if actual != 0 else float("inf")
            if 0.8 < ratio < 1.2 and n != actual:
                mismatches.append(
                    f"Mismatch: stated {n} but actual {field_name} is {actual}"
                )
                break

    if mismatches:
        return False, "; ".join(mismatches)

    return True, ""


def generate_with_fallback(
    transfer: TransferVariable,
    regret_entry: dict | None,
    data: InventoryData,
    api_key: str | None = None,
) -> str:
    """Generate justification with automatic fallback to deterministic template.

    1. Try Gemini LLM
    2. Verify the LLM output against the transfer record
    3. If either step fails → use template

    Returns the final justification string (never empty).
    """
    genai_text = generate_justification(transfer, regret_entry, data, api_key)

    if genai_text:
        ok, reason = verify_justification(genai_text, transfer)
        if ok:
            logger.info("Justification for %s: GenAI (verified)", transfer.var_id)
            return genai_text
        else:
            logger.warning(
                "Justification verification failed for %s: %s — using template",
                transfer.var_id, reason,
            )

    template = _template_justification(transfer, data, regret_entry)
    logger.info("Justification for %s: template fallback", transfer.var_id)
    return template


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

def _load_api_key() -> str | None:
    load_dotenv()
    key = os.environ.get("GEMINI_API_KEY")
    if not key or key.startswith("PASTE"):
        return None
    return key
